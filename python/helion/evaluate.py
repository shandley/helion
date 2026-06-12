"""
Gene prediction evaluation following Burset & Guigo (1996) conventions.

Compares a predicted GFF3 against a reference GFF3 at two levels:

  Nucleotide: Sn, Sp, CC (Matthews correlation coefficient)
  Exon:       exact boundary match -- Sn, Sp, F1

Strand is ignored throughout: a coding position or exon matches
regardless of which strand it is annotated on. This is appropriate
until Helion's Viterbi emits strand-aware predictions.

Usage:
    from helion.evaluate import evaluate, format_report
    result = evaluate(ref_gff3, pred_gff3, genome_fai)
    print(format_report(result))
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class NucleotideMetrics:
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def sensitivity(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def specificity(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        sn, sp = self.sensitivity, self.specificity
        return 2 * sn * sp / (sn + sp) if (sn + sp) else 0.0

    @property
    def correlation(self) -> float:
        """Matthews Correlation Coefficient (CC)."""
        tp, fp, tn, fn = self.tp, self.fp, self.tn, self.fn
        denom_sq = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        if denom_sq == 0:
            return 0.0
        return (tp * tn - fp * fn) / math.sqrt(denom_sq)

    @property
    def total_ref(self) -> int:
        return self.tp + self.fn

    @property
    def total_pred(self) -> int:
        return self.tp + self.fp


@dataclass
class ExonMetrics:
    tp: int   # predicted exon matching reference (within tolerance)
    fp: int   # predicted exon not matching any reference exon
    fn: int   # reference exon not matched by any prediction
    tolerance: int = 0  # boundary tolerance in nt used to compute this

    @property
    def sensitivity(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def specificity(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        sn, sp = self.sensitivity, self.specificity
        return 2 * sn * sp / (sn + sp) if (sn + sp) else 0.0

    @property
    def total_ref(self) -> int:
        return self.tp + self.fn

    @property
    def total_pred(self) -> int:
        return self.tp + self.fp


@dataclass
class OverlapStats:
    """Boundary offset analysis for predicted exons vs nearest reference exons."""
    n_pred: int
    n_overlapping: int          # predicted exons overlapping any reference exon
    pct_overlapping: float      # n_overlapping / n_pred
    median_start_offset: float  # median |pred_start - nearest_ref_start| for overlapping
    median_end_offset: float    # median |pred_end   - nearest_ref_end  | for overlapping
    p90_start_offset: float     # 90th percentile start offset
    p90_end_offset: float       # 90th percentile end offset


@dataclass
class EvaluationResult:
    nucleotide: NucleotideMetrics
    exon: ExonMetrics
    overlap: Optional[OverlapStats] = None  # populated when compute_overlap=True


# ---------------------------------------------------------------------------
# GFF3 parsing
# ---------------------------------------------------------------------------

CdsMap = dict[str, list[tuple[int, int]]]


def load_cds_intervals(gff3_path: Path) -> CdsMap:
    """
    Extract all CDS intervals from a GFF3 file.

    Converts from GFF3's 1-based inclusive coordinates to 0-based
    half-open [start, end) intervals.
    """
    intervals: CdsMap = {}
    with gff3_path.open() as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip().split("\t")
            if len(parts) < 9:
                continue
            if parts[2] != "CDS":
                continue
            seqid = parts[0]
            start = int(parts[3]) - 1
            end = int(parts[4])
            intervals.setdefault(seqid, []).append((start, end))
    return intervals


def read_fai(fai_path: Path) -> dict[str, int]:
    """Parse a samtools .fai index and return {seqid: length}."""
    lengths: dict[str, int] = {}
    with fai_path.open() as f:
        for line in f:
            parts = line.split("\t")
            if len(parts) >= 2:
                lengths[parts[0]] = int(parts[1])
    return lengths


def _fai_path(genome: Path) -> Path:
    fai = Path(str(genome) + ".fai")
    if not fai.exists():
        raise FileNotFoundError(
            f"Genome index not found: {fai}\n"
            f"Run: python -c \"from pyfaidx import Fasta; Fasta('{genome}')\""
        )
    return fai


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def _nucleotide_metrics(
    ref: CdsMap,
    pred: CdsMap,
    chrom_lengths: dict[str, int],
) -> NucleotideMetrics:
    tp = fp = tn = fn = 0

    for seqid, length in chrom_lengths.items():
        ref_mask = np.zeros(length, dtype=bool)
        for s, e in ref.get(seqid, []):
            ref_mask[s:e] = True

        pred_mask = np.zeros(length, dtype=bool)
        for s, e in pred.get(seqid, []):
            pred_mask[s:e] = True

        tp += int(np.count_nonzero(ref_mask & pred_mask))
        fp += int(np.count_nonzero(~ref_mask & pred_mask))
        tn += int(np.count_nonzero(~ref_mask & ~pred_mask))
        fn += int(np.count_nonzero(ref_mask & ~pred_mask))

    return NucleotideMetrics(tp=tp, fp=fp, tn=tn, fn=fn)


def _exon_metrics(ref: CdsMap, pred: CdsMap, tolerance: int = 0) -> ExonMetrics:
    """
    Compute exon-level metrics.

    With tolerance=0 (default): exact boundary match required.
    With tolerance>0: both boundaries must be within tolerance nt of
    a reference exon's boundaries. Each reference exon can be matched
    at most once; each predicted exon counts at most once.
    """
    if tolerance == 0:
        ref_set: set[tuple[str, int, int]] = {
            (seqid, s, e)
            for seqid, ivs in ref.items()
            for s, e in ivs
        }
        pred_set: set[tuple[str, int, int]] = {
            (seqid, s, e)
            for seqid, ivs in pred.items()
            for s, e in ivs
        }
        tp = len(ref_set & pred_set)
        fn = len(ref_set - pred_set)
        fp = len(pred_set - ref_set)
        return ExonMetrics(tp=tp, fp=fp, fn=fn, tolerance=0)

    # Tolerance > 0: build sorted ref arrays per seqid for fast lookup
    import bisect

    ref_by_seqid: dict[str, list[tuple[int, int]]] = {
        seqid: sorted(ivs) for seqid, ivs in ref.items()
    }

    matched_refs: set[tuple[str, int, int]] = set()
    fp = 0

    for seqid, pred_ivs in pred.items():
        ref_ivs = ref_by_seqid.get(seqid, [])
        ref_starts = [s for s, e in ref_ivs]

        for ps, pe in pred_ivs:
            # Binary search for reference exons whose start is near ps
            lo = bisect.bisect_left(ref_starts, ps - tolerance)
            hi = bisect.bisect_right(ref_starts, ps + tolerance)
            found = False
            for i in range(lo, hi):
                rs, re = ref_ivs[i]
                key = (seqid, rs, re)
                if abs(ps - rs) <= tolerance and abs(pe - re) <= tolerance:
                    if key not in matched_refs:
                        matched_refs.add(key)
                        found = True
                        break
            if not found:
                fp += 1

    tp = len(matched_refs)
    total_ref = sum(len(ivs) for ivs in ref.values())
    fn = total_ref - tp
    return ExonMetrics(tp=tp, fp=fp, fn=fn, tolerance=tolerance)


def _overlap_stats(ref: CdsMap, pred: CdsMap) -> OverlapStats:
    """
    For each predicted exon, find the nearest overlapping reference exon
    and report boundary offset statistics.

    Helps diagnose whether predictions are in the right place but with
    imprecise boundaries, vs completely wrong locations.
    """
    import bisect

    ref_by_seqid: dict[str, list[tuple[int, int]]] = {
        seqid: sorted(ivs) for seqid, ivs in ref.items()
    }

    n_pred = sum(len(ivs) for ivs in pred.values())
    start_offsets: list[int] = []
    end_offsets: list[int] = []

    for seqid, pred_ivs in pred.items():
        ref_ivs = ref_by_seqid.get(seqid, [])
        if not ref_ivs:
            continue
        ref_starts = [s for s, e in ref_ivs]
        ref_ends = [e for s, e in ref_ivs]

        for ps, pe in pred_ivs:
            # Find reference exons that overlap [ps, pe]
            # An exon [rs, re] overlaps [ps, pe] if rs < pe and re > ps
            lo = bisect.bisect_left(ref_starts, pe) - 1
            # scan nearby exons for overlap
            best_start_off: Optional[int] = None
            best_end_off: Optional[int] = None

            for i in range(max(0, lo - 5), min(len(ref_ivs), lo + 10)):
                rs, re = ref_ivs[i]
                if rs >= pe:
                    break
                if re <= ps:
                    continue
                # overlaps
                soff = abs(ps - rs)
                eoff = abs(pe - re)
                if best_start_off is None or (soff + eoff) < (best_start_off + best_end_off):  # type: ignore[operator]
                    best_start_off = soff
                    best_end_off = eoff

            if best_start_off is not None:
                start_offsets.append(best_start_off)
                end_offsets.append(best_end_off)  # type: ignore[arg-type]

    n_overlapping = len(start_offsets)
    pct = n_overlapping / n_pred if n_pred > 0 else 0.0

    if start_offsets:
        arr_s = np.array(start_offsets)
        arr_e = np.array(end_offsets)
        med_s = float(np.median(arr_s))
        med_e = float(np.median(arr_e))
        p90_s = float(np.percentile(arr_s, 90))
        p90_e = float(np.percentile(arr_e, 90))
    else:
        med_s = med_e = p90_s = p90_e = 0.0

    return OverlapStats(
        n_pred=n_pred,
        n_overlapping=n_overlapping,
        pct_overlapping=pct,
        median_start_offset=med_s,
        median_end_offset=med_e,
        p90_start_offset=p90_s,
        p90_end_offset=p90_e,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate(
    ref_gff3: Path,
    pred_gff3: Path,
    genome: Path,
    tolerance: int = 0,
    compute_overlap: bool = False,
) -> EvaluationResult:
    """
    Evaluate a Helion prediction GFF3 against a reference annotation.

    Args:
        ref_gff3:        Reference annotation (Ensembl, GENCODE, etc.)
        pred_gff3:       Helion output GFF3
        genome:          Genome FASTA (must have a .fai index alongside it)
        tolerance:       Allow boundary offsets up to this many nt (0 = exact match)
        compute_overlap: If True, also compute overlap/offset statistics

    Returns:
        EvaluationResult with nucleotide, exon, and optional overlap metrics.
    """
    ref = load_cds_intervals(ref_gff3)
    pred = load_cds_intervals(pred_gff3)
    chrom_lengths = read_fai(_fai_path(genome))

    overlap = _overlap_stats(ref, pred) if compute_overlap else None

    return EvaluationResult(
        nucleotide=_nucleotide_metrics(ref, pred, chrom_lengths),
        exon=_exon_metrics(ref, pred, tolerance=tolerance),
        overlap=overlap,
    )


def format_report(result: EvaluationResult, label: str = "") -> str:
    """Return a plain-text evaluation report."""
    nt = result.nucleotide
    ex = result.exon
    header = f"Helion evaluation report{' -- ' + label if label else ''}"
    exon_label = (
        f"Exon level (±{ex.tolerance} nt boundary tolerance)"
        if ex.tolerance > 0
        else "Exon level (exact boundary match)"
    )
    lines = [
        header,
        "=" * max(42, len(header)),
        "",
        "Nucleotide level",
        f"  Sensitivity (Sn):  {nt.sensitivity:.4f}",
        f"  Specificity (Sp):  {nt.specificity:.4f}",
        f"  Correlation  (CC): {nt.correlation:.4f}",
        f"  F1:                {nt.f1:.4f}",
        f"  Reference coding nt:  {nt.total_ref:,}",
        f"  Predicted coding nt:  {nt.total_pred:,}",
        "",
        exon_label,
        f"  Sensitivity (Sn):  {ex.sensitivity:.4f}",
        f"  Specificity (Sp):  {ex.specificity:.4f}",
        f"  F1:                {ex.f1:.4f}",
        f"  Reference exons:   {ex.total_ref:,}",
        f"  Predicted exons:   {ex.total_pred:,}",
        f"  Correct (TP):      {ex.tp:,}",
        f"  Missing (FN):      {ex.fn:,}",
        f"  Wrong   (FP):      {ex.fp:,}",
    ]

    if result.overlap is not None:
        ov = result.overlap
        lines += [
            "",
            "Overlap analysis (predicted vs nearest reference exon)",
            f"  Predicted exons total:    {ov.n_pred:,}",
            f"  Overlapping any ref exon: {ov.n_overlapping:,} ({ov.pct_overlapping:.1%})",
            f"  Median start offset:      {ov.median_start_offset:.0f} nt",
            f"  Median end   offset:      {ov.median_end_offset:.0f} nt",
            f"  90th pct start offset:    {ov.p90_start_offset:.0f} nt",
            f"  90th pct end   offset:    {ov.p90_end_offset:.0f} nt",
        ]

    lines += ["", "Note: strand-agnostic (Helion currently predicts + strand only)"]
    return "\n".join(lines)
