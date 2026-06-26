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
# Splice-consensus audit (quantifies the expected gain from enforcing GT-AG).
#
# Boundary conventions, derived against dataset.py:_make_labels and graph.rs:
#   exon = 0-based half-open [s, e):  s = first coding base, e = first intronic
#   base (exclusive end). Every exon has a genomic-LOW boundary (examine
#   seq[s-2:s], the two bases just before the exon) and a genomic-HIGH boundary
#   (examine seq[e:e+2], the two bases just after it). Which is donor vs acceptor,
#   and the expected dinucleotide, depends on strand (minus strand is the reverse
#   complement):
#     + low  = acceptor -> AG          - low  = donor    -> AC  (rc of GT) / GC
#     + high = donor    -> GT / GC     - high = acceptor -> CT  (rc of AG)
# A "splice" boundary is any exon boundary shared with an adjacent intron, i.e.
# not the transcript's outermost ends (those are the start/stop codon, not a
# splice site, so GT-AG enforcement does not apply there).
# ---------------------------------------------------------------------------

# (seqid, s, e) -> (strand, low_is_splice, high_is_splice)
ExonRole = tuple[str, bool, bool]


def load_genome(path: Path) -> dict[str, str]:
    """Load a (possibly multi-record) FASTA into {seqid: uppercased sequence}."""
    genome: dict[str, str] = {}
    name: str | None = None
    chunks: list[str] = []
    with path.open() as f:
        for line in f:
            if line.startswith(">"):
                if name is not None:
                    genome[name] = "".join(chunks).upper()
                name = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line.strip())
    if name is not None:
        genome[name] = "".join(chunks).upper()
    return genome


def _seq_for(genome: dict[str, str], seqid: str) -> str | None:
    """Look up a chromosome, tolerating a chr/no-chr prefix mismatch."""
    if seqid in genome:
        return genome[seqid]
    alt = seqid[3:] if seqid.startswith("chr") else f"chr{seqid}"
    return genome.get(alt)


def _low_canonical(seq: str, s: int, strand: str, allow_gc: bool) -> bool:
    if s < 2:
        return False
    di = seq[s - 2:s]
    if strand == "+":  # acceptor AG
        return di == "AG"
    return di == "AC" or (allow_gc and di == "GC")  # minus-strand donor (rc of GT/GC)


def _high_canonical(seq: str, e: int, strand: str, allow_gc: bool) -> bool:
    if e + 2 > len(seq):
        return False
    di = seq[e:e + 2]
    if strand == "+":  # donor GT/GC
        return di == "GT" or (allow_gc and di == "GC")
    return di == "CT"  # minus-strand acceptor (rc of AG)


def _boundary_kind(strand: str, side: str) -> str:
    if (strand == "+" and side == "low") or (strand == "-" and side == "high"):
        return "acceptor"
    return "donor"


def load_exon_roles(gff3_path: Path) -> dict[tuple[str, int, int], ExonRole]:
    """For every exon, record its strand and whether each genomic boundary is a
    splice site (vs a transcript-terminal start/stop boundary).

    An exon's low boundary is a splice site iff it is not the genomic-lowest exon
    of a transcript; its high boundary iff not the genomic-highest. Flags are
    OR-ed across transcripts (a boundary counts as splice if it is one anywhere).
    """
    mrna_strand: dict[str, str] = {}
    cds_by_parent: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    with gff3_path.open() as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip().split("\t")
            if len(parts) < 9 or parts[2] not in ("mRNA", "CDS"):
                continue
            seqid = parts[0]
            start = int(parts[3]) - 1
            end = int(parts[4])
            strand = parts[6] if parts[6] in ("+", "-") else "."
            attrs = parts[8]
            if parts[2] == "mRNA":
                mrna_strand[_extract_attr(attrs, "ID") or ""] = strand
            else:
                cds_by_parent[_extract_attr(attrs, "Parent") or ""].append((seqid, start, end))

    roles: dict[tuple[str, int, int], ExonRole] = {}
    for tid, cds_list in cds_by_parent.items():
        strand = mrna_strand.get(tid, ".")
        ordered = sorted(set(cds_list))
        n = len(ordered)
        for idx, (seqid, s, e) in enumerate(ordered):
            low_sp, high_sp = idx > 0, idx < n - 1
            key = (seqid, s, e)
            if key in roles:
                st, ls, hs = roles[key]
                roles[key] = (st, ls or low_sp, hs or high_sp)
            else:
                roles[key] = (strand, low_sp, high_sp)
    return roles


def _consensus_tally(
    roles: dict[tuple[str, int, int], ExonRole], genome: dict[str, str], allow_gc: bool
) -> dict[str, list[int]]:
    """Count canonical/total at acceptor and donor splice boundaries."""
    tally: dict[str, list[int]] = {"acceptor": [0, 0], "donor": [0, 0]}
    for (seqid, s, e), (strand, low_sp, high_sp) in roles.items():
        if strand not in ("+", "-"):
            continue
        seq = _seq_for(genome, seqid)
        if seq is None:
            continue
        if low_sp:
            kind = _boundary_kind(strand, "low")
            tally[kind][1] += 1
            tally[kind][0] += int(_low_canonical(seq, s, strand, allow_gc))
        if high_sp:
            kind = _boundary_kind(strand, "high")
            tally[kind][1] += 1
            tally[kind][0] += int(_high_canonical(seq, e, strand, allow_gc))
    return tally


def _nearest_canonical(
    seq: str, pos: int, side: str, strand: str, allow_gc: bool, window: int
) -> int | None:
    """Nearest position to `pos` (within +/-window) whose `side` boundary is
    canonical. Ties resolve toward the smaller offset, then downstream."""
    check = _low_canonical if side == "low" else _high_canonical
    for d in range(window + 1):
        for cand in ((pos,) if d == 0 else (pos - d, pos + d)):
            if check(seq, cand, strand, allow_gc):
                return cand
    return None


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


def run(
    ref_path: Path,
    pred_path: Path,
    genome_path: Path,
    tolerance: int,
    strand_aware: bool,
    allow_gc: bool,
    snap_window: int,
) -> str:
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

    # Splice-consensus inputs for the expected-gain estimate.
    genome = load_genome(genome_path)
    ref_roles = load_exon_roles(ref_path)
    pred_roles = load_exon_roles(pred_path)
    ref_audit = _consensus_tally(ref_roles, genome, allow_gc)
    pred_audit = _consensus_tally(pred_roles, genome, allow_gc)

    n_boundary = n_undetected = n_isoform = 0
    abs_start_offsets: list[int] = []
    abs_end_offsets: list[int] = []
    signed_start_offsets: list[int] = []
    signed_end_offsets: list[int] = []
    # Per boundary-off exon: min over overlapping preds of max(|d_start|,|d_end|).
    # An exon is recoverable at tolerance t iff this value <= t.
    min_max_offsets: list[int] = []
    n_snap_recoverable = 0  # BOUNDARY_OFF exons made exact by snapping to GT-AG

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
            abs_start_offsets.append(abs(best[0] - rs))
            abs_end_offsets.append(abs(best[1] - re))
            signed_start_offsets.append(best[0] - rs)
            signed_end_offsets.append(best[1] - re)
            # Smallest worst-boundary error across all overlapping preds.
            min_max = min(max(abs(p[0] - rs), abs(p[1] - re)) for p in overlapping_preds)
            min_max_offsets.append(min_max)
            if _is_snap_recoverable(
                genome, ref_roles, seqid, rs, re, best[0], best[1], tolerance, allow_gc, snap_window
            ):
                n_snap_recoverable += 1
        else:
            n_undetected += 1

    return _format_report(
        ref_path=ref_path,
        pred_path=pred_path,
        tolerance=tolerance,
        strand_aware=strand_aware,
        allow_gc=allow_gc,
        snap_window=snap_window,
        n_ref=len(ref_keys),
        n_matched=len(matched),
        n_missed=len(missed),
        n_boundary=n_boundary,
        n_undetected=n_undetected,
        n_isoform=n_isoform,
        abs_start_offsets=abs_start_offsets,
        abs_end_offsets=abs_end_offsets,
        signed_start_offsets=signed_start_offsets,
        signed_end_offsets=signed_end_offsets,
        min_max_offsets=min_max_offsets,
        ref_audit=ref_audit,
        pred_audit=pred_audit,
        n_snap_recoverable=n_snap_recoverable,
    )


def _is_snap_recoverable(
    genome: dict[str, str],
    ref_roles: dict[tuple[str, int, int], ExonRole],
    seqid: str,
    rs: int,
    re: int,
    pred_s: int,
    pred_e: int,
    tolerance: int,
    allow_gc: bool,
    window: int,
) -> bool:
    """Would enforcing GT-AG make this BOUNDARY_OFF exon an exact match?

    Estimate: snap each predicted boundary to the nearest canonical dinucleotide
    (within +/-window) and ask whether it lands on the reference boundary. A
    splice boundary can snap; a transcript-terminal boundary (start/stop codon)
    cannot, so it must already match within tolerance. Optimistic: assumes the
    decoder would pick the nearest canonical site, which it may not.
    """
    role = ref_roles.get((seqid, rs, re))
    if role is None:
        return False
    strand, low_sp, high_sp = role
    if strand not in ("+", "-"):
        return False
    seq = _seq_for(genome, seqid)
    if seq is None:
        return False

    if low_sp:
        if _nearest_canonical(seq, pred_s, "low", strand, allow_gc, window) != rs:
            return False
    elif abs(pred_s - rs) > tolerance:
        return False

    if high_sp:
        if _nearest_canonical(seq, pred_e, "high", strand, allow_gc, window) != re:
            return False
    elif abs(pred_e - re) > tolerance:
        return False

    return True


def _audit_line(label: str, tally: dict[str, list[int]]) -> str:
    parts = []
    for kind in ("acceptor", "donor"):
        canon, total = tally[kind]
        parts.append(f"{kind} {_pct(canon, total)} (n={total:,})")
    return f"  {label:<26} {parts[0]}   {parts[1]}"


def _format_report(
    *,
    ref_path: Path,
    pred_path: Path,
    tolerance: int,
    strand_aware: bool,
    allow_gc: bool,
    snap_window: int,
    n_ref: int,
    n_matched: int,
    n_missed: int,
    n_boundary: int,
    n_undetected: int,
    n_isoform: int,
    abs_start_offsets: list[int],
    abs_end_offsets: list[int],
    signed_start_offsets: list[int],
    signed_end_offsets: list[int],
    min_max_offsets: list[int],
    ref_audit: dict[str, list[int]],
    pred_audit: dict[str, list[int]],
    n_snap_recoverable: int,
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
        lines.append(
            f"  signed start:    median {np.median(np.array(signed_start_offsets)):+.0f} nt   "
            f"mean {np.mean(np.array(signed_start_offsets)):+.1f} nt  (pred - ref)"
        )
        lines.append(
            f"  signed end:      median {np.median(np.array(signed_end_offsets)):+.0f} nt   "
            f"mean {np.mean(np.array(signed_end_offsets)):+.1f} nt  (pred - ref)"
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

    # Expected gain from enforcing the splice dinucleotide (GT-AG / GC-AG).
    gc_note = "GT/GC donor, AG acceptor" if allow_gc else "GT donor, AG acceptor"
    lines.append(f"Splice-consensus audit (canonical = {gc_note}; strand-aware)")
    lines.append(_audit_line("reference splice sites:", ref_audit))
    lines.append(_audit_line("predicted splice sites:", pred_audit))
    ref_total = sum(ref_audit[k][1] for k in ref_audit)
    ref_canon = sum(ref_audit[k][0] for k in ref_audit)
    pred_total = sum(pred_audit[k][1] for k in pred_audit)
    pred_canon = sum(pred_audit[k][0] for k in pred_audit)
    lines.append(
        f"  -> reference {_pct(ref_canon, ref_total)} canonical vs predicted "
        f"{_pct(pred_canon, pred_total)}: the gap is headroom the decoder ignores."
    )
    lines.append("")
    lines.append(
        f"Expected gain: snap predicted boundaries to nearest GT-AG (+/-{snap_window} nt)"
    )
    lines.append(
        f"  +{n_snap_recoverable:,} BOUNDARY_OFF exons become exact "
        f"({_pct(n_snap_recoverable, n_boundary)} of BOUNDARY_OFF, "
        f"{_pct(n_snap_recoverable, n_ref)} of all reference exons)"
    )
    recovered_recall = n_matched + n_snap_recoverable
    lines.append(
        f"  estimated recall: {_pct(n_matched, n_ref)} -> {_pct(recovered_recall, n_ref)} "
        f"({n_matched:,} -> {recovered_recall:,} TP)"
    )
    lines.append(
        "  NOTE optimistic: assumes the decoder picks the nearest canonical site; "
        "it re-scores, so treat as an upper-ish bound."
    )
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
        help="Genome FASTA (read for the splice-consensus / expected-gain audit)",
    )
    parser.add_argument("--tolerance", type=int, default=0, help="Boundary tolerance in nt")
    parser.add_argument(
        "--strand-aware", action=argparse.BooleanOptionalAction, default=False,
        help="Require predicted and reference exons to share strand",
    )
    parser.add_argument(
        "--allow-gc-donor", action=argparse.BooleanOptionalAction, default=True,
        help="Count GC (not just GT) as a canonical donor (vertebrate/plant)",
    )
    parser.add_argument(
        "--snap-window", type=int, default=30,
        help="Max nt to snap a predicted boundary to a canonical site in the gain estimate",
    )
    args = parser.parse_args()

    print(run(
        args.ref, args.pred, args.genome, args.tolerance,
        args.strand_aware, args.allow_gc_donor, args.snap_window,
    ))


if __name__ == "__main__":
    main()
