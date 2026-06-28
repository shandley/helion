from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import numpy.typing as npt
import torch
from pyfaidx import Fasta  # type: ignore[import]
from torch.utils.data import Dataset

from helion.signals.model import DIST_W, N_CLASSES, _one_hot

_RC_TABLE = str.maketrans("ACGTNacgtn", "TGCANtgcan")


@dataclass
class GFF3Gene:
    seqid: str
    start: int   # 0-based
    end: int
    strand: str
    exons: list[tuple[int, int]]  # list of (start, end) 0-based


def parse_gff3(path: Path) -> list[GFF3Gene]:
    """Parse coding exon coordinates from a GFF3 annotation file."""
    genes: dict[str, GFF3Gene] = {}
    exons: dict[str, list[tuple[int, int]]] = {}

    with path.open() as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip().split("\t")
            if len(parts) < 9:
                continue
            feature = parts[2]
            if feature not in ("mRNA", "CDS"):
                continue

            seqid, _, _, start, end, _, strand, _, attrs = parts
            s, e = int(start) - 1, int(end)

            if feature == "mRNA":
                # Key by the transcript's own ID so CDS children (keyed by Parent) match
                gene_id = _extract_attr(attrs, "ID") or ""
                genes[gene_id] = GFF3Gene(seqid=seqid, start=s, end=e, strand=strand, exons=[])
            elif feature == "CDS":
                # Parent points to the transcript ID
                gene_id = _extract_attr(attrs, "Parent") or ""
                exons.setdefault(gene_id, []).append((s, e))

    for gid, gene in genes.items():
        gene.exons = sorted(exons.get(gid, []))

    return [g for g in genes.values() if g.exons]


_INTERGENIC_BUFFER = 2000  # bp around each gene boundary excluded from hard-negative sampling


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge a list of (start, end) intervals into non-overlapping sorted intervals."""
    if not intervals:
        return []
    lo, hi = sorted(intervals)[0]
    result: list[tuple[int, int]] = []
    for start, end in sorted(intervals)[1:]:
        if start <= hi:
            hi = max(hi, end)
        else:
            result.append((lo, hi))
            lo, hi = start, end
    result.append((lo, hi))
    return result


def _extract_attr(attrs: str, key: str) -> str | None:
    for part in attrs.split(";"):
        if part.startswith(f"{key}="):
            return part[len(key) + 1:]
    return None


def _reverse_complement(seq: str) -> str:
    return seq.translate(_RC_TABLE)[::-1]


def _rc_gene(gene: GFF3Gene) -> GFF3Gene:
    """Return a virtual + strand gene with exons mapped to RC coordinates.

    Used so _make_labels() always receives a sense-orientation gene,
    avoiding the mixed + /- training signal that caused class confusion.
    """
    L = gene.end - gene.start
    rc_exons: list[tuple[int, int]] = sorted(
        (gene.start + L - (e - gene.start), gene.start + L - (s - gene.start))
        for s, e in gene.exons
    )
    return GFF3Gene(seqid=gene.seqid, start=gene.start, end=gene.end, strand="+", exons=rc_exons)


class SignalDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """
    Windowed dataset for training the signal model.

    Each sample is a (one_hot_dna, label_tensor) pair where labels are
    per-position class indices matching the N_CLASSES output channels.

    Val split is chromosome-level to prevent data leakage. Training windows
    use splice-site-centered sampling (one window per signal position) to
    ensure every window contains at least one rare signal class. Val windows
    use the original sliding-window approach so val loss is comparable across
    training runs.
    """

    def __init__(
        self,
        genome: Path,
        annotations: Path,
        window_size: int = 5000,
        organism: str = "vertebrate",
        val_chromosomes: list[str] | None = None,
        split: str = "train",
        centered_sampling: bool = False,
        neg_fraction: float = 0.0,
        train_chromosomes: list[str] | None = None,
        feature_path: Path | None = None,
    ) -> None:
        self.genome = genome
        self.window_size = window_size
        self.organism = organism
        self.val_chromosomes = set(val_chromosomes or [])
        self.split = split
        self.centered_sampling = centered_sampling
        self._fasta = Fasta(str(genome))

        all_genes = parse_gff3(annotations)
        if self.val_chromosomes:
            if split == "train":
                self.genes = [g for g in all_genes if g.seqid not in self.val_chromosomes]
            else:
                self.genes = [g for g in all_genes if g.seqid in self.val_chromosomes]
        else:
            self.genes = all_genes

        # Optional: restrict TRAIN genes to a chromosome subset (cost control for
        # the DNA-embedding fusion A/B). Val is left untouched.
        if train_chromosomes and split == "train":
            keep = set(train_chromosomes)
            self.genes = [g for g in self.genes if g.seqid in keep]

        self.windows = list(self._build_windows())
        if neg_fraction > 0.0 and split == "train":
            n_neg = max(1, int(len(self.windows) * neg_fraction))
            neg_windows = self._sample_intergenic_windows(n_neg)
            self.windows.extend(neg_windows)
            print(
                f"Hard negatives: {len(neg_windows):,} intergenic windows "
                f"(target {n_neg:,}, {len(self.windows):,} total)",
                flush=True,
            )
        self._fasta.close()
        del self._fasta

        # Optional DNA-embedding fusion features: a precomputed (N, W, 6) array
        # index-aligned to self.windows (the window list is deterministic). Loaded
        # read-only via mmap; concatenated onto the one-hot input in __getitem__.
        self.features: npt.NDArray[np.float32] | None = None
        if feature_path is not None:
            feats = np.load(feature_path, mmap_mode="r")
            if feats.shape[0] != len(self.windows):
                raise ValueError(
                    f"feature cache {feature_path} has {feats.shape[0]} rows but "
                    f"dataset built {len(self.windows)} windows -- config mismatch"
                )
            if feats.shape[1] != self.window_size:
                raise ValueError(
                    f"feature cache window {feats.shape[1]} != window_size {self.window_size}"
                )
            self.features = feats

    def window_sequences(self) -> list[str]:
        """The oriented sequence string of every window, in index order (used by
        the feature-precompute script)."""
        return [seq for seq, _ in self.windows]

    def _build_windows(self) -> Iterator[tuple[str, npt.NDArray[np.int8]]]:
        half = self.window_size // 2
        for gene in self.genes:
            L = gene.end - gene.start
            seq = str(self._fasta[gene.seqid][gene.start:gene.end])
            # Normalize to sense orientation: RC both sequence and label array for
            # minus-strand genes so the model always sees donor=GT, acceptor=AG.
            if gene.strand == "-":
                seq = _reverse_complement(seq)
                labels = _make_labels(_rc_gene(gene))
            else:
                labels = _make_labels(gene)

            if self.centered_sampling:
                # One window per signal position (donor, acceptor, start, stop = classes 0-3).
                # Guarantees every training window contains at least one rare signal label.
                anchor_positions = sorted(set(int(i) for i in np.where(labels < 4)[0]))

                # Single-exon genes have no splice sites; fall back to gene midpoint so
                # the coding signal from these genes still contributes to training.
                if not anchor_positions:
                    anchor_positions = [L // 2]

                seen_starts: set[int] = set()
                for pos in anchor_positions:
                    win_start = max(0, pos - half)
                    win_end = win_start + self.window_size
                    if win_end > L:
                        win_end = L
                        win_start = max(0, win_end - self.window_size)
                    if win_end - win_start < self.window_size:
                        continue  # gene shorter than one window
                    if win_start in seen_starts:
                        continue
                    seen_starts.add(win_start)
                    yield seq[win_start:win_end], labels[win_start:win_end]
            else:
                # Sliding window (used for val and fungus random-split fallback).
                for i in range(0, L - self.window_size + 1, self.window_size // 2):
                    yield seq[i:i + self.window_size], labels[i:i + self.window_size]

    def _sample_intergenic_windows(self, n: int) -> list[tuple[str, npt.NDArray[np.int8]]]:
        """Sample n windows from intergenic regions, all positions labeled class 7."""
        rng = np.random.default_rng(seed=42)

        # Only sample from train chromosomes -- never fasta.keys(), which includes val chrom
        train_seqids = {g.seqid for g in self.genes}

        # Covered intervals: gene span ± buffer per chromosome
        covered: dict[str, list[tuple[int, int]]] = {sid: [] for sid in train_seqids}
        for gene in self.genes:
            covered[gene.seqid].append((
                max(0, gene.start - _INTERGENIC_BUFFER),
                gene.end + _INTERGENIC_BUFFER,
            ))

        # Enumerate all non-overlapping candidate window starts in uncovered gaps
        candidates: list[tuple[str, int]] = []  # (seqid, window_start)
        for seqid in sorted(train_seqids):
            chr_len = len(self._fasta[seqid])
            merged = _merge_intervals(covered.get(seqid, []))
            pos = 0
            for ivl_start, ivl_end in merged:
                while pos + self.window_size <= ivl_start:
                    candidates.append((seqid, pos))
                    pos += self.window_size
                pos = max(pos, ivl_end)
            while pos + self.window_size <= chr_len:
                candidates.append((seqid, pos))
                pos += self.window_size

        if not candidates:
            print("WARNING: no intergenic candidate windows found", flush=True)
            return []

        # Over-sample 2x to absorb N-rich windows (centromeric/telomeric gaps)
        n_draw = min(2 * n, len(candidates))
        drawn = rng.choice(len(candidates), size=n_draw, replace=False)

        intergenic_label = np.int8(N_CLASSES - 1)
        result: list[tuple[str, npt.NDArray[np.int8]]] = []
        for raw_idx in drawn:
            if len(result) >= n:
                break
            seqid, w_start = candidates[int(raw_idx)]
            seq = str(self._fasta[seqid][w_start:w_start + self.window_size])
            if len(seq) < self.window_size:
                continue
            # Skip windows predominantly composed of N (assembly gaps)
            seq_upper = seq.upper()
            acgt = (seq_upper.count("A") + seq_upper.count("C")
                    + seq_upper.count("G") + seq_upper.count("T"))
            if acgt / len(seq) < 0.9:
                continue
            labels = np.full(self.window_size, intergenic_label, dtype=np.int8)
            result.append((seq, labels))

        return result

    def compute_class_weights(self) -> torch.Tensor:
        """Class weights for weighted cross-entropy.

        Each class gets equal total contribution to the loss:
          weight[c] = total_positions / (N_CLASSES * count[c])
        This up-weights rare signal classes (donor/acceptor/start/stop) while
        leaving common classes (coding, intergenic) near their natural scale.
        Capped at 100x to prevent extreme gradients.
        """
        counts = np.zeros(N_CLASSES, dtype=np.int64)
        for _, labels in self.windows:
            counts += np.bincount(labels.astype(np.intp), minlength=N_CLASSES)
        counts = np.maximum(counts, 1)
        total = float(counts.sum())
        weights = total / (N_CLASSES * counts.astype(np.float64))
        weights = np.minimum(weights, 100.0)
        return torch.tensor(weights, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq, labels = self.windows[idx]
        x = _one_hot(seq)  # (4, L)
        if self.features is not None:
            # (L, 6) -> (6, L), concat onto one-hot -> (10, L)
            feat = torch.from_numpy(np.ascontiguousarray(self.features[idx])).T.float()
            x = torch.cat([x, feat], dim=0)
        dist = _distance_targets(labels)  # (2, L) normalised to [-1, 1]
        return (
            x,
            torch.tensor(labels, dtype=torch.long),
            torch.from_numpy(dist),
        )


def _signed_distance_field(positions: npt.NDArray[np.int64], length: int) -> npt.NDArray[np.float32]:
    """Signed distance (nt) from every position to the nearest site in `positions`.

    Convention: value is `idx - nearest_site`, so 0 at a site, negative upstream,
    positive downstream, clamped to [-DIST_W, DIST_W]. Empty `positions` (a window
    with no such boundary) yields a saturated +DIST_W everywhere.
    """
    if len(positions) == 0:
        return np.full(length, DIST_W, dtype=np.float32)
    pos = np.sort(positions)
    idx = np.arange(length)
    ins = np.searchsorted(pos, idx)
    left = np.clip(ins - 1, 0, len(pos) - 1)
    right = np.clip(ins, 0, len(pos) - 1)
    d_left = idx - pos[left]
    d_right = idx - pos[right]
    signed = np.where(np.abs(d_left) <= np.abs(d_right), d_left, d_right).astype(np.float32)
    return np.clip(signed, -DIST_W, DIST_W)


def _distance_targets(labels: npt.NDArray[np.int8]) -> npt.NDArray[np.float32]:
    """Signed-distance regression targets for donor (label 0) and acceptor (label 1),
    derived from the per-position class labels and normalised to [-1, 1]."""
    L = len(labels)
    donor_pos = np.nonzero(labels == 0)[0]
    acceptor_pos = np.nonzero(labels == 1)[0]
    d_donor = _signed_distance_field(donor_pos, L)
    d_acceptor = _signed_distance_field(acceptor_pos, L)
    return np.stack([d_donor, d_acceptor]) / DIST_W


def _make_labels(gene: GFF3Gene) -> npt.NDArray[np.int8]:
    """Assign per-position class labels from a gene annotation."""
    L = gene.end - gene.start
    labels = np.full(L, N_CLASSES - 1, dtype=np.int8)  # default: intergenic

    exons = sorted(gene.exons)

    frame_offset = 0  # reading frame carried across exons
    for i, (exon_start, exon_end) in enumerate(exons):
        rel_s = exon_start - gene.start
        rel_e = exon_end - gene.start
        cds_end = min(rel_e, L)

        # Coding phase: all positions in this exon get the same label (the
        # exon's phase). The DAG scores each candidate exon by averaging one
        # coding channel across all positions; uniform phase labeling ensures
        # the correct-phase DAG node scores higher than the other two.
        # Cycling labels (f0,f1,f2,...) would average to 1/3 for every phase
        # and give the DAG no discriminating signal.
        if rel_s < cds_end:
            labels[rel_s:cds_end] = frame_offset + 4
        frame_offset = (frame_offset + (rel_e - rel_s)) % 3

        # Donor: first intronic position after this exon end (skip last exon).
        # Use index rather than position: rel_e < L fails when 3' UTR separates
        # the last CDS exon from the mRNA boundary.
        if i < len(exons) - 1:
            labels[rel_e] = 0
        # Acceptor: first position of this exon, i.e. where the exon starts.
        # Placed at rel_s (not rel_s-2) so the DAG produces correct start coordinates.
        # Skipped for the first exon which has no upstream intron.
        if i > 0:
            labels[rel_s] = 1

    # Start codon: first 3 nt of first exon. Stop codon: last 3 nt of last exon.
    # Minus-strand genes are RC-mapped via _rc_gene() before calling this function,
    # so gene.strand is always "+" here.
    first_rel = exons[0][0] - gene.start
    last_rel_e = exons[-1][1] - gene.start
    labels[first_rel:min(first_rel + 3, L)] = 2   # start
    labels[max(0, last_rel_e - 3):last_rel_e] = 3  # stop

    return labels
