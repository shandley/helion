"""
Recall diagnostic: explain WHY Helion misses reference exons.

vertebrate_v3 recalls only ~44% of reference exons on chr22 (TP ~2,800 of
~6,400). This script classifies every *missed* reference exon into one of three
mutually-exclusive buckets so we know which intervention each loss implies:

  BOUNDARY_OFF  A predicted exon overlaps the reference exon but neither
                boundary matches within tolerance -- it is detected, but its
                start/end are wrong. Implies: splice-site sharpening / a more
                lenient boundary tolerance would recover it.

  UNDETECTED    No predicted exon overlaps the reference exon at all -- a true
                detection failure. Implies: CNN sensitivity (donor/acceptor/
                coding channels) or DAG node/edge thresholds are too strict.

  ISOFORM_ONLY  The missed exon's locus is already covered by a DIFFERENT
                transcript isoform whose exon WAS matched, so a
                single-transcript-per-locus decoder cannot recover it. Implies:
                a multi-isoform decoder or an isoform-aware evaluation change --
                not a sensitivity problem.

The "matched" definition mirrors python/helion/evaluate.py exactly (see
`_matched_ref_keys` below) so this diagnostic's TP set is identical to the real
evaluator's. Read-only: parses GFF3 and prints; never writes.

Run on a login node -- CPU only, no torch needed (pure parsing + intervals).

Usage:
    python scripts/recall_diagnostic.py \
        --ref   results/ref_vertebrate_chr22.gff3 \
        --pred  results/pred_vertebrate_v3_chr22_t0.3_gs0.0.gff3 \
        --genome results/vertebrate_chr22.fa
"""

from __future__ import annotations

import argparse
import bisect
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Mirror the sys.path pattern from scripts/score_dist.py so `helion` is importable
# whether this runs from the repo on the RIS cluster or anywhere else.
H = Path("/storage3/fs1/shandley/Active/helion")
if (H / "python").is_dir():
    sys.path.insert(0, str(H / "python"))
else:
    # Fall back to the repo this script lives in (scripts/ -> repo/python).
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

# evaluate.py imports only stdlib + numpy (no torch), so this is dependency-light.
from helion.evaluate import load_cds_intervals  # noqa: E402

# A reference exon is identified, like in evaluate.py, by (seqid, start, end, strand)
# in 0-based half-open coordinates. Strand is normalised to "." unless strand-aware.
ExonKey = tuple[str, int, int, str]

# Default inputs -- these live on the RIS cluster, not locally. Use the UNFILTERED
# prediction (best-recall case) so we measure the detection ceiling, not filtering loss.
DEFAULT_REF = H / "results/ref_vertebrate_chr22.gff3"
DEFAULT_PRED = H / "results/pred_vertebrate_v3_chr22_t0.3_gs0.0.gff3"
DEFAULT_GENOME = H / "results/vertebrate_chr22.fa"


# ---------------------------------------------------------------------------
# Reference exon -> transcript membership (needed for ISOFORM_ONLY)
#
# load_cds_intervals() discards transcript identity, so we re-parse the reference
# GFF3 here. This mirrors parse_gff3() in python/helion/signals/dataset.py
# (lines 27-58: mRNA keyed by ID, CDS keyed by Parent, 1-based inclusive ->
# 0-based half-open) but is inlined to avoid importing dataset.py, which pulls in
# torch/pyfaidx. Strand is normalised exactly as load_cds_intervals does so the
# keys built here are identical to the evaluator's.
# ---------------------------------------------------------------------------

def _extract_attr(attrs: str, key: str) -> str | None:
    """Mirror dataset.py:_extract_attr."""
    for part in attrs.split(";"):
        if part.startswith(f"{key}="):
            return part[len(key) + 1:]
    return None


def load_exon_transcripts(gff3_path: Path, strand_aware: bool) -> dict[ExonKey, set[str]]:
    """Map each reference exon key to the set of transcript IDs that contain it."""
    mrna_strand: dict[str, str] = {}
    cds_by_parent: dict[str, list[tuple[str, int, int]]] = defaultdict(list)

    with gff3_path.open() as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip().split("\t")
            if len(parts) < 9:
                continue
            feature = parts[2]
            if feature not in ("mRNA", "CDS"):
                continue
            seqid = parts[0]
            start = int(parts[3]) - 1  # 1-based inclusive -> 0-based half-open
            end = int(parts[4])
            strand = parts[6] if parts[6] in ("+", "-") else "."
            attrs = parts[8]
            if feature == "mRNA":
                tid = _extract_attr(attrs, "ID") or ""
                mrna_strand[tid] = strand
            else:  # CDS
                parent = _extract_attr(attrs, "Parent") or ""
                cds_by_parent[parent].append((seqid, start, end))

    membership: dict[ExonKey, set[str]] = defaultdict(set)
    for tid, cds_list in cds_by_parent.items():
        strand = mrna_strand.get(tid, ".")
        for seqid, s, e in cds_list:
            key: ExonKey = (seqid, s, e, strand if strand_aware else ".")
            membership[key].add(tid)
    return membership


# ---------------------------------------------------------------------------
# Matching -- mirrors evaluate.py:_exon_metrics (lines 256-290) exactly.
#
# The greedy bisect algorithm there generalises the exact-match set logic:
# with tolerance=0 it reproduces the exact-boundary TP count, with tolerance>0
# it allows boundary slack. Each predicted exon matches at most one reference
# exon; each reference exon is matched at most once. We return the matched
# reference KEYS (evaluate.py only returns counts) so we can classify per-exon.
# ---------------------------------------------------------------------------

def _to_keys(intervals: list[tuple[int, int, str]], seqid: str, strand_aware: bool) -> set[ExonKey]:
    return {
        (seqid, s, e, st if strand_aware else ".")
        for s, e, st in intervals
    }


def _matched_ref_keys(
    ref: dict[str, list[tuple[int, int, str]]],
    pred: dict[str, list[tuple[int, int, str]]],
    tolerance: int,
    strand_aware: bool,
) -> set[ExonKey]:
    """Return the set of reference exon keys matched by some prediction.

    Faithful re-implementation of evaluate.py:_exon_metrics' greedy matcher.
    """
    ref_by_seqid: dict[str, list[tuple[int, int, str]]] = {
        seqid: sorted(ivs) for seqid, ivs in ref.items()
    }

    matched: set[ExonKey] = set()
    for seqid, pred_ivs in pred.items():
        ref_ivs = ref_by_seqid.get(seqid, [])
        ref_starts = [s for s, _e, _st in ref_ivs]
        for ps, pe, pst in pred_ivs:
            lo = bisect.bisect_left(ref_starts, ps - tolerance)
            hi = bisect.bisect_right(ref_starts, ps + tolerance)
            for i in range(lo, hi):
                rs, re, rst = ref_ivs[i]
                if strand_aware and pst != rst:
                    continue
                key: ExonKey = (seqid, rs, re, rst if strand_aware else ".")
                if abs(ps - rs) <= tolerance and abs(pe - re) <= tolerance:
                    if key not in matched:
                        matched.add(key)
                        break
    return matched


# ---------------------------------------------------------------------------
# Overlap lookup helpers (overlap defined as in evaluate.py:_overlap_stats
# line 319: [a0,a1) overlaps [b0,b1) iff a0 < b1 and a1 > b0).
# ---------------------------------------------------------------------------

class IntervalIndex:
    """Per-seqid interval index for fast overlap queries.

    Stores intervals as (start, end, strand) sorted by start, with the maximum
    interval length per seqid so a bisect window captures every possible overlap.
    """

    def __init__(self, intervals: dict[str, list[tuple[int, int, str]]]) -> None:
        self._ivs: dict[str, list[tuple[int, int, str]]] = {}
        self._starts: dict[str, list[int]] = {}
        self._max_len: dict[str, int] = {}
        for seqid, ivs in intervals.items():
            s_ivs = sorted(ivs)
            self._ivs[seqid] = s_ivs
            self._starts[seqid] = [s for s, _e, _st in s_ivs]
            self._max_len[seqid] = max((e - s for s, e, _st in s_ivs), default=0)

    def overlaps(
        self, seqid: str, qs: int, qe: int, qstrand: str, strand_aware: bool
    ) -> list[tuple[int, int, str]]:
        ivs = self._ivs.get(seqid)
        if not ivs:
            return []
        starts = self._starts[seqid]
        max_len = self._max_len[seqid]
        # Any interval overlapping [qs, qe) has start in (qs - max_len, qe).
        lo = bisect.bisect_left(starts, qs - max_len)
        hi = bisect.bisect_left(starts, qe)
        out: list[tuple[int, int, str]] = []
        for i in range(lo, hi):
            s, e, st = ivs[i]
            if e <= qs:  # no overlap (interval ends at/before query start)
                continue
            if strand_aware and st != qstrand:
                continue
            out.append((s, e, st))
        return out


# ---------------------------------------------------------------------------
# Diagnostic
# ---------------------------------------------------------------------------

def _pct(n: int, d: int) -> str:
    return f"{(100.0 * n / d):.1f}%" if d else "n/a"


def run(ref_path: Path, pred_path: Path, tolerance: int, strand_aware: bool) -> str:
    ref = load_cds_intervals(ref_path)
    pred = load_cds_intervals(pred_path)

    # All distinct reference exon keys (the evaluator's exon unit; identical
    # exons shared by multiple transcripts collapse, matching evaluate.py's set).
    ref_keys: set[ExonKey] = set()
    for seqid, ivs in ref.items():
        ref_keys |= _to_keys(ivs, seqid, strand_aware)

    matched = _matched_ref_keys(ref, pred, tolerance, strand_aware) & ref_keys
    missed = ref_keys - matched

    membership = load_exon_transcripts(ref_path, strand_aware)
    pred_index = IntervalIndex(pred)
    ref_index = IntervalIndex(ref)

    n_boundary = n_undetected = n_isoform = 0
    abs_start_offsets: list[int] = []
    abs_end_offsets: list[int] = []
    # Per boundary-off exon: min over overlapping preds of max(|d_start|,|d_end|).
    # An exon is recoverable at tolerance t iff this value <= t.
    min_max_offsets: list[int] = []

    for seqid, rs, re, rst in missed:
        overlapping_preds = pred_index.overlaps(seqid, rs, re, rst, strand_aware)

        # ISOFORM_ONLY (checked first; takes precedence so buckets are disjoint
        # and sum to the missed total). The missed exon's locus is redundant with
        # a DIFFERENT transcript's exon that WAS matched -> structurally
        # unrecoverable by a single-transcript-per-locus decoder.
        this_tids = membership.get((seqid, rs, re, rst), set())
        is_isoform = False
        for ors, ore, orst in ref_index.overlaps(seqid, rs, re, rst, strand_aware):
            other_key: ExonKey = (seqid, ors, ore, orst)
            if other_key == (seqid, rs, re, rst):
                continue
            if other_key not in matched:
                continue
            # Belongs to a transcript that this exon does not -> different isoform.
            if membership.get(other_key, set()) - this_tids:
                is_isoform = True
                break

        if is_isoform:
            n_isoform += 1
            continue

        if overlapping_preds:
            n_boundary += 1
            # Best overlapping prediction = smallest total boundary error.
            best = min(overlapping_preds, key=lambda p: abs(p[0] - rs) + abs(p[1] - re))
            d_start = abs(best[0] - rs)
            d_end = abs(best[1] - re)
            abs_start_offsets.append(d_start)
            abs_end_offsets.append(d_end)
            # Smallest worst-boundary error across all overlapping preds.
            min_max = min(max(abs(p[0] - rs), abs(p[1] - re)) for p in overlapping_preds)
            min_max_offsets.append(min_max)
        else:
            n_undetected += 1

    return _format_report(
        ref_path=ref_path,
        pred_path=pred_path,
        tolerance=tolerance,
        strand_aware=strand_aware,
        n_ref=len(ref_keys),
        n_matched=len(matched),
        n_missed=len(missed),
        n_boundary=n_boundary,
        n_undetected=n_undetected,
        n_isoform=n_isoform,
        abs_start_offsets=abs_start_offsets,
        abs_end_offsets=abs_end_offsets,
        min_max_offsets=min_max_offsets,
    )


def _format_report(
    *,
    ref_path: Path,
    pred_path: Path,
    tolerance: int,
    strand_aware: bool,
    n_ref: int,
    n_matched: int,
    n_missed: int,
    n_boundary: int,
    n_undetected: int,
    n_isoform: int,
    abs_start_offsets: list[int],
    abs_end_offsets: list[int],
    min_max_offsets: list[int],
) -> str:
    lines: list[str] = []
    lines.append("Helion recall diagnostic -- why are reference exons missed?")
    lines.append("=" * 60)
    lines.append(f"  ref:   {ref_path}")
    lines.append(f"  pred:  {pred_path}")
    lines.append(
        f"  match: tolerance={tolerance} nt, "
        f"strand-{'aware' if strand_aware else 'agnostic'} "
        f"(mirrors evaluate.py:_exon_metrics)"
    )
    lines.append("")
    lines.append("Exon recall")
    lines.append(f"  Reference exons (distinct): {n_ref:,}")
    lines.append(f"  Matched (TP):               {n_matched:,}  ({_pct(n_matched, n_ref)})")
    lines.append(f"  Missed (FN):                {n_missed:,}  ({_pct(n_missed, n_ref)})")
    lines.append("")
    lines.append("Missed-exon breakdown (mutually exclusive; ISOFORM_ONLY takes precedence)")
    lines.append(
        f"  BOUNDARY_OFF: {n_boundary:,}  ({_pct(n_boundary, n_missed)})"
        "  -- detected but boundaries wrong"
    )
    lines.append(
        f"  UNDETECTED:   {n_undetected:,}  ({_pct(n_undetected, n_missed)})"
        "  -- no overlapping prediction"
    )
    lines.append(
        f"  ISOFORM_ONLY: {n_isoform:,}  ({_pct(n_isoform, n_missed)})"
        "  -- locus covered by a matched alt isoform"
    )
    lines.append("")

    # BOUNDARY_OFF offset distribution + tolerance-recovery.
    lines.append("BOUNDARY_OFF boundary-error distribution (best overlapping prediction)")
    if abs_start_offsets:
        arr_s = np.array(abs_start_offsets)
        arr_e = np.array(abs_end_offsets)
        lines.append(
            f"  |start offset|:  median {np.median(arr_s):.0f} nt   "
            f"p90 {np.percentile(arr_s, 90):.0f} nt"
        )
        lines.append(
            f"  |end   offset|:  median {np.median(arr_e):.0f} nt   "
            f"p90 {np.percentile(arr_e, 90):.0f} nt"
        )
        lines.append("")
        lines.append("  Recoverable as TP if boundary tolerance were relaxed:")
        mm = np.array(min_max_offsets)
        for t in (2, 5, 10):
            n_rec = int(np.count_nonzero(mm <= t))
            lines.append(
                f"    tolerance {t:>2} nt: +{n_rec:,} TP "
                f"({_pct(n_rec, len(min_max_offsets))} of BOUNDARY_OFF)"
            )
    else:
        lines.append("  (no BOUNDARY_OFF exons)")
    lines.append("")

    # Interpretation hints.
    lines.append("Interpretation")
    lines.append(
        "  BOUNDARY_OFF -> sharpen splice/start/stop channels or relax exon "
        "boundary tolerance; check phase handling."
    )
    lines.append(
        "  UNDETECTED   -> raise CNN sensitivity (donor/acceptor/coding) or "
        "lower DAG node/edge score thresholds; the locus is invisible."
    )
    lines.append(
        "  ISOFORM_ONLY -> not a sensitivity problem; needs a multi-isoform "
        "decoder or isoform-collapsed evaluation."
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify missed Helion reference exons (boundary/undetected/isoform).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ref", type=Path, default=DEFAULT_REF, help="Reference GFF3")
    parser.add_argument(
        "--pred", type=Path, default=DEFAULT_PRED, help="Prediction GFF3 (use unfiltered for ceiling)"
    )
    parser.add_argument(
        "--genome", type=Path, default=DEFAULT_GENOME,
        help="Genome FASTA (accepted for interface parity; unused for exon-level matching)",
    )
    parser.add_argument("--tolerance", type=int, default=0, help="Boundary tolerance in nt")
    parser.add_argument(
        "--strand-aware", action=argparse.BooleanOptionalAction, default=False,
        help="Require predicted and reference exons to share strand",
    )
    args = parser.parse_args()

    print(run(args.ref, args.pred, args.tolerance, args.strand_aware))


if __name__ == "__main__":
    main()
