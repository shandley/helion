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
        # Use Python ints to avoid float64 overflow on large genomes.
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
    tp: int   # predicted exon with exact start and end match in reference
    fp: int   # predicted exon not in reference ("wrong" exons)
    fn: int   # reference exon not predicted ("missing" exons)

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
class EvaluationResult:
    nucleotide: NucleotideMetrics
    exon: ExonMetrics


# ---------------------------------------------------------------------------
# GFF3 parsing
# ---------------------------------------------------------------------------

# CDS intervals keyed by seqid: {seqid: [(start_0based, end_exclusive), ...]}
CdsMap = dict[str, list[tuple[int, int]]]


def load_cds_intervals(gff3_path: Path) -> CdsMap:
    """
    Extract all CDS intervals from a GFF3 file.

    Converts from GFF3's 1-based inclusive coordinates to 0-based
    half-open [start, end) intervals. Works with both Ensembl and
    Helion output formats.
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
            start = int(parts[3]) - 1   # GFF3 1-based -> 0-based
            end = int(parts[4])          # GFF3 inclusive -> exclusive (no change)
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


def _exon_metrics(ref: CdsMap, pred: CdsMap) -> ExonMetrics:
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
    return ExonMetrics(tp=tp, fp=fp, fn=fn)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate(ref_gff3: Path, pred_gff3: Path, genome: Path) -> EvaluationResult:
    """
    Evaluate a Helion prediction GFF3 against a reference annotation.

    Args:
        ref_gff3:  Reference annotation (Ensembl, GENCODE, etc.)
        pred_gff3: Helion output GFF3
        genome:    Genome FASTA (must have a .fai index alongside it)

    Returns:
        EvaluationResult with nucleotide and exon metrics.
    """
    ref = load_cds_intervals(ref_gff3)
    pred = load_cds_intervals(pred_gff3)
    chrom_lengths = read_fai(_fai_path(genome))

    return EvaluationResult(
        nucleotide=_nucleotide_metrics(ref, pred, chrom_lengths),
        exon=_exon_metrics(ref, pred),
    )


def format_report(result: EvaluationResult) -> str:
    """Return a plain-text evaluation report."""
    nt = result.nucleotide
    ex = result.exon
    lines = [
        "Helion evaluation report",
        "=" * 42,
        "",
        "Nucleotide level",
        f"  Sensitivity (Sn):  {nt.sensitivity:.4f}",
        f"  Specificity (Sp):  {nt.specificity:.4f}",
        f"  Correlation  (CC): {nt.correlation:.4f}",
        f"  F1:                {nt.f1:.4f}",
        f"  Reference coding nt:  {nt.total_ref:,}",
        f"  Predicted coding nt:  {nt.total_pred:,}",
        "",
        "Exon level (exact boundary match)",
        f"  Sensitivity (Sn):  {ex.sensitivity:.4f}",
        f"  Specificity (Sp):  {ex.specificity:.4f}",
        f"  F1:                {ex.f1:.4f}",
        f"  Reference exons:   {ex.total_ref:,}",
        f"  Predicted exons:   {ex.total_pred:,}",
        f"  Correct (TP):      {ex.tp:,}",
        f"  Missing (FN):      {ex.fn:,}",
        f"  Wrong   (FP):      {ex.fp:,}",
        "",
        "Note: strand-agnostic (Helion currently predicts + strand only)",
    ]
    return "\n".join(lines)
