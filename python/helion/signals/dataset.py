from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import numpy.typing as npt
import torch
from pyfaidx import Fasta  # type: ignore[import]
from torch.utils.data import Dataset

from helion.signals.model import N_CLASSES, _one_hot

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

        self.windows = list(self._build_windows())
        self._fasta.close()
        del self._fasta

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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq, labels = self.windows[idx]
        return _one_hot(seq), torch.tensor(labels, dtype=torch.long)


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
