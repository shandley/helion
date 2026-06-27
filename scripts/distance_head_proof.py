"""Signed-distance regression head: does it localize splice sites more sharply?

Python-only (no Rust) proof for the Helion gene-prediction project. The signal
CNN now emits, per genomic position, both:

  (a) CLASSIFICATION channels -- softmax donor/acceptor probabilities
      (``scores.donor`` / ``scores.acceptor``). A predicted site is a local
      maximum of this probability above ``--threshold``.

  (b) REGRESSION channels -- a predicted signed distance (in nt) to the nearest
      donor/acceptor (``scores.d_donor`` / ``scores.d_acceptor``). Convention:
      ~0 AT a true site, NEGATIVE just upstream (position < site), POSITIVE just
      downstream, saturating around +/-32. A predicted site is a NEGATIVE->
      POSITIVE zero-crossing of this field.

Hypothesis: the regression field's zero-crossings land on true splice sites more
precisely (smaller median offset, higher within-2nt fraction) than the
classification peaks.

SCOPE: forward ("+") strand only. We score chr22 on the sense strand and compare
only against "+"-strand reference transcripts. This keeps the proof simple; the
minus strand (which inference handles via reverse-complement) is intentionally
out of scope and would need a separate RC pass. Per-position arrays from the
10kb/500nt overlapping windows are placed back into genomic coordinates with
last-write-wins on the 500nt overlap (the later window overwrites the shared
tail) -- documented, and immaterial to peak/crossing geometry.

Runs as a GPU sbatch job (needs torch); honours --device. Read-only on data;
only prints a comparative report and a one-line verdict.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import numpy.typing as npt

# Mirror the sys.path pattern from scripts/score_dist.py so `helion` imports
# whether this runs from the repo on the RIS cluster or anywhere else.
H = Path("/storage3/fs1/shandley/Active/helion")
if (H / "python").is_dir():
    sys.path.insert(0, str(H / "python"))
else:
    # Fall back to the repo this script lives in (scripts/ -> repo/python).
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from helion.io.fasta import read_windows  # noqa: E402
from helion.signals.dataset import parse_gff3  # noqa: E402
from helion.signals.model import SignalModel  # noqa: E402

# Window geometry mirrors helion/predict.py exactly.
WINDOW_SIZE = 10_000
OVERLAP = 500

DEFAULT_MODEL = H / "models/vertebrate_distance"
DEFAULT_GENOME = H / "results/vertebrate_chr22.fa"
DEFAULT_REF = H / "results/ref_vertebrate_chr22.gff3"


# ---------------------------------------------------------------------------
# 1. Score the genome on the forward strand, placed into genomic coordinates.
# ---------------------------------------------------------------------------

class GenomeScores:
    """Per-seqid forward-strand arrays in genomic coordinates.

    Probability arrays are initialised to 0 (so untouched positions can never be
    a peak above a positive threshold); distance arrays to NaN (so untouched
    positions are skipped by crossing detection).
    """

    def __init__(self, lengths: dict[str, int]) -> None:
        self.donor: dict[str, npt.NDArray[np.float64]] = {}
        self.acceptor: dict[str, npt.NDArray[np.float64]] = {}
        self.d_donor: dict[str, npt.NDArray[np.float64]] = {}
        self.d_acceptor: dict[str, npt.NDArray[np.float64]] = {}
        for seqid, length in lengths.items():
            self.donor[seqid] = np.zeros(length, dtype=np.float64)
            self.acceptor[seqid] = np.zeros(length, dtype=np.float64)
            self.d_donor[seqid] = np.full(length, np.nan, dtype=np.float64)
            self.d_acceptor[seqid] = np.full(length, np.nan, dtype=np.float64)


def score_genome(model: SignalModel, genome: Path) -> GenomeScores:
    """Score every window on the forward strand and place arrays genomically."""
    # pyfaidx is already a dependency (read_windows uses it); query lengths up
    # front so we can allocate full-chromosome arrays before filling them.
    from pyfaidx import Fasta  # type: ignore[import]

    fa = Fasta(str(genome))
    lengths = {seqid: len(fa[seqid]) for seqid in fa.keys()}
    out = GenomeScores(lengths)

    for window in read_windows(genome, window_size=WINDOW_SIZE, overlap=OVERLAP):
        scores = model.score(window.sequence)  # forward strand only

        d_donor = getattr(scores, "d_donor", None)
        d_acceptor = getattr(scores, "d_acceptor", None)
        if d_donor is None or d_acceptor is None:
            print(
                "ERROR: this model has no distance head "
                "(scores.d_donor / scores.d_acceptor are None). "
                "The signed-distance proof cannot run -- train/load a model with "
                "the regression head enabled.",
                file=sys.stderr,
            )
            sys.exit(1)

        s, e = window.start, window.end  # genomic, 0-based half-open
        out.donor[window.seqid][s:e] = scores.donor
        out.acceptor[window.seqid][s:e] = scores.acceptor
        out.d_donor[window.seqid][s:e] = d_donor
        out.d_acceptor[window.seqid][s:e] = d_acceptor

    return out


# ---------------------------------------------------------------------------
# 2. True forward-strand splice sites from the reference GFF3.
#
# Mirrors python/helion/signals/dataset.py: parse_gff3 yields exons in 0-based
# half-open genomic coordinates; _make_labels places the donor at the exon END
# (rel_e -> first intronic base after each non-LAST exon) and the acceptor at
# the exon START (rel_s -> first base of each non-FIRST exon). We keep genomic
# (not gene-relative) coordinates. Sites shared by isoforms collapse via set().
# ---------------------------------------------------------------------------

def true_sites(
    ref: Path,
) -> tuple[dict[str, npt.NDArray[np.int64]], dict[str, npt.NDArray[np.int64]]]:
    donor_by_seq: dict[str, set[int]] = defaultdict(set)
    acceptor_by_seq: dict[str, set[int]] = defaultdict(set)

    for gene in parse_gff3(ref):
        if gene.strand != "+":  # forward strand only
            continue
        exons = sorted(gene.exons)
        n = len(exons)
        for i, (exon_start, exon_end) in enumerate(exons):
            if i < n - 1:  # non-last exon -> donor at exon end
                donor_by_seq[gene.seqid].add(exon_end)
            if i > 0:  # non-first exon -> acceptor at exon start
                acceptor_by_seq[gene.seqid].add(exon_start)

    donors = {sid: np.array(sorted(v), dtype=np.int64) for sid, v in donor_by_seq.items()}
    acceptors = {sid: np.array(sorted(v), dtype=np.int64) for sid, v in acceptor_by_seq.items()}
    return donors, acceptors


# ---------------------------------------------------------------------------
# 3. Predicted-site detectors.
# ---------------------------------------------------------------------------

def classification_peaks(prob: npt.NDArray[np.float64], threshold: float) -> npt.NDArray[np.int64]:
    """Local maxima of a probability track above ``threshold``.

    A position i is a peak iff prob[i] > prob[i-1] and prob[i] >= prob[i+1] and
    prob[i] > threshold. The asymmetric comparison keeps a single index per
    plateau (the left-most), avoiding double counts.
    """
    if prob.size < 3:
        return np.array([], dtype=np.int64)
    left = prob[1:-1] > prob[:-2]
    right = prob[1:-1] >= prob[2:]
    above = prob[1:-1] > threshold
    idx = np.where(left & right & above)[0] + 1  # +1: shift back to full-array index
    return idx.astype(np.int64)


def regression_zero_crossings(dist: npt.NDArray[np.float64]) -> npt.NDArray[np.int64]:
    """Negative->positive zero-crossings of a signed-distance track.

    The distance-to-nearest-donor field is negative just upstream of a true site
    and positive just downstream, so a true site sits on a -> + crossing. (At the
    midpoint BETWEEN two sites the nearest site flips, giving a + -> - crossing,
    which we deliberately ignore.) For each adjacent pair (i, i+1) with valid,
    oppositely-signed values d[i] < 0 <= d[i+1], we linearly interpolate the
    sub-integer crossing x = i - d[i] / (d[i+1] - d[i]) and round to the nearest
    base. NaN positions (window-uncovered) are skipped.
    """
    d0 = dist[:-1]
    d1 = dist[1:]
    valid = ~np.isnan(d0) & ~np.isnan(d1)
    cross = valid & (d0 < 0.0) & (d1 >= 0.0)
    i = np.where(cross)[0]
    if i.size == 0:
        return np.array([], dtype=np.int64)
    denom = d1[i] - d0[i]  # strictly > 0 given opposite signs
    frac = -d0[i] / denom  # in [0, 1)
    pos = np.rint(i + frac).astype(np.int64)
    return np.unique(pos)


# ---------------------------------------------------------------------------
# 4. Match each true site to its nearest predicted site within max_offset.
# ---------------------------------------------------------------------------

def nearest_offsets(
    true_pos: npt.NDArray[np.int64],
    pred_pos: npt.NDArray[np.int64],
    max_offset: int,
) -> tuple[npt.NDArray[np.int64], int]:
    """Return (|offset| for each matched true site, n_true).

    For every true site we find the nearest predicted site; if it is within
    ``max_offset`` the true site is matched and its |offset| is returned,
    otherwise it is a miss (n_true - len(offsets)).
    """
    n_true = int(true_pos.size)
    if n_true == 0 or pred_pos.size == 0:
        return np.array([], dtype=np.int64), n_true

    pred_sorted = np.sort(pred_pos)
    idx = np.searchsorted(pred_sorted, true_pos)
    offsets: list[int] = []
    for t, j in zip(true_pos.tolist(), idx.tolist()):
        best = max_offset + 1
        if j < pred_sorted.size:
            best = min(best, abs(int(pred_sorted[j]) - t))
        if j > 0:
            best = min(best, abs(int(pred_sorted[j - 1]) - t))
        if best <= max_offset:
            offsets.append(best)
    return np.array(offsets, dtype=np.int64), n_true


# ---------------------------------------------------------------------------
# 5. Report.
# ---------------------------------------------------------------------------

class MethodStats:
    """Offset statistics for one (signal, method) pair, summed over seqids."""

    def __init__(self, offsets: npt.NDArray[np.int64], n_true: int) -> None:
        self.offsets = offsets
        self.n_true = n_true
        self.n_matched = int(offsets.size)
        self.n_missed = n_true - self.n_matched

    @property
    def median(self) -> float:
        return float(np.median(self.offsets)) if self.n_matched else float("inf")

    @property
    def p90(self) -> float:
        return float(np.percentile(self.offsets, 90)) if self.n_matched else float("inf")

    def within(self, k: int) -> float:
        if self.n_true == 0:
            return 0.0
        return float(np.count_nonzero(self.offsets <= k)) / self.n_true


def _collect(
    true_by_seq: dict[str, npt.NDArray[np.int64]],
    pred_by_seq: dict[str, npt.NDArray[np.int64]],
    max_offset: int,
) -> MethodStats:
    all_offsets: list[npt.NDArray[np.int64]] = []
    n_true = 0
    for seqid, true_pos in true_by_seq.items():
        pred_pos = pred_by_seq.get(seqid, np.array([], dtype=np.int64))
        offsets, n = nearest_offsets(true_pos, pred_pos, max_offset)
        all_offsets.append(offsets)
        n_true += n
    merged = np.concatenate(all_offsets) if all_offsets else np.array([], dtype=np.int64)
    return MethodStats(merged, n_true)


def _stat_block(name: str, cls: MethodStats, reg: MethodStats) -> list[str]:
    lines = [f"  {name}  (n_true = {cls.n_true:,})"]
    header = f"    {'metric':<22}{'classification':>16}{'regression':>16}"
    lines.append(header)
    lines.append(f"    {'median |offset| (nt)':<22}{cls.median:>16.2f}{reg.median:>16.2f}")
    lines.append(f"    {'p90 |offset| (nt)':<22}{cls.p90:>16.2f}{reg.p90:>16.2f}")
    for k in (0, 1, 2, 5):
        lines.append(
            f"    {f'within {k} nt':<22}"
            f"{cls.within(k):>15.1%} {reg.within(k):>15.1%}"
        )
    lines.append(
        f"    {'misses (>max_offset)':<22}{cls.n_missed:>16,}{reg.n_missed:>16,}"
    )
    return lines


def _better(cls: MethodStats, reg: MethodStats) -> bool:
    """Regression clearly sharper: smaller median AND higher within-2nt fraction."""
    return reg.median < cls.median and reg.within(2) > cls.within(2)


def run(
    model_dir: Path,
    genome: Path,
    ref: Path,
    device: str,
    threshold: float,
    max_offset: int,
) -> str:
    model = SignalModel.load(model_dir, organism="vertebrate", device=device)
    gscores = score_genome(model, genome)

    donor_true, acceptor_true = true_sites(ref)

    # Predicted sites per seqid for each (signal, method).
    donor_peaks = {s: classification_peaks(a, threshold) for s, a in gscores.donor.items()}
    acceptor_peaks = {s: classification_peaks(a, threshold) for s, a in gscores.acceptor.items()}
    donor_cross = {s: regression_zero_crossings(a) for s, a in gscores.d_donor.items()}
    acceptor_cross = {s: regression_zero_crossings(a) for s, a in gscores.d_acceptor.items()}

    donor_cls = _collect(donor_true, donor_peaks, max_offset)
    donor_reg = _collect(donor_true, donor_cross, max_offset)
    acceptor_cls = _collect(acceptor_true, acceptor_peaks, max_offset)
    acceptor_reg = _collect(acceptor_true, acceptor_cross, max_offset)

    lines: list[str] = []
    lines.append("Helion distance-head proof -- zero-crossings vs classification peaks")
    lines.append("=" * 70)
    lines.append(f"  model:      {model_dir}")
    lines.append(f"  genome:     {genome}")
    lines.append(f"  reference:  {ref}")
    lines.append(f"  device:     {device}")
    lines.append(
        f"  params:     threshold={threshold}  max_offset={max_offset} nt  "
        f"window={WINDOW_SIZE}/overlap={OVERLAP}"
    )
    lines.append("  scope:      FORWARD strand only (sense-strand scoring vs + transcripts)")
    lines.append("")
    lines.append("Localization sharpness (lower median / higher within-2nt = sharper)")
    lines.extend(_stat_block("DONOR", donor_cls, donor_reg))
    lines.append("")
    lines.extend(_stat_block("ACCEPTOR", acceptor_cls, acceptor_reg))
    lines.append("")

    donor_better = _better(donor_cls, donor_reg)
    acceptor_better = _better(acceptor_cls, acceptor_reg)
    supported = donor_better and acceptor_better
    verdict = "SUPPORTED" if supported else "NOT SUPPORTED"
    lines.append(
        f"VERDICT: hypothesis {verdict} -- regression zero-crossings are "
        f"{'sharper' if supported else 'NOT clearly sharper'} than classification "
        f"peaks for {'both' if supported else 'not both'} donor and acceptor "
        f"(donor sharper={donor_better}, acceptor sharper={acceptor_better})."
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate that the signed-distance head localizes splice sites "
        "more sharply than the classification channels (forward strand, Python-only).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Model directory")
    parser.add_argument("--genome", type=Path, default=DEFAULT_GENOME, help="Genome FASTA (chr22)")
    parser.add_argument("--ref", type=Path, default=DEFAULT_REF, help="Reference GFF3")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device")
    parser.add_argument(
        "--threshold", type=float, default=0.3, help="Classification peak threshold"
    )
    parser.add_argument("--max-offset", type=int, default=50, help="Max match distance (nt)")
    args = parser.parse_args()

    print(run(args.model, args.genome, args.ref, args.device, args.threshold, args.max_offset))


if __name__ == "__main__":
    main()
