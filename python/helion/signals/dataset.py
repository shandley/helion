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


class SignalDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """
    Windowed dataset for training the signal model.

    Each sample is a (one_hot_dna, label_tensor) pair where labels are
    per-position class indices matching the N_CLASSES output channels.

    Val split is chromosome-level to prevent data leakage: if
    val_chromosomes is given, all transcripts on those chromosomes go to
    the validation set and are excluded from training. This is the
    correct approach since windows from the same gene share sequence
    context and must not appear in both splits.
    """

    def __init__(
        self,
        genome: Path,
        annotations: Path,
        window_size: int = 5000,
        organism: str = "vertebrate",
        val_chromosomes: list[str] | None = None,
        split: str = "train",
    ) -> None:
        self.genome = genome
        self.window_size = window_size
        self.organism = organism
        self.val_chromosomes = set(val_chromosomes or [])
        self.split = split
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

    def _build_windows(self) -> Iterator[tuple[str, npt.NDArray[np.int8]]]:
        for gene in self.genes:
            seq = str(self._fasta[gene.seqid][gene.start:gene.end])
            labels = _make_labels(gene)
            for i in range(0, len(seq) - self.window_size + 1, self.window_size // 2):
                yield seq[i:i + self.window_size], labels[i:i + self.window_size]

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq, labels = self.windows[idx]
        return _one_hot(seq), torch.tensor(labels, dtype=torch.long)


def _make_labels(gene: GFF3Gene) -> npt.NDArray[np.int8]:
    """Assign per-position class labels from a gene annotation."""
    L = gene.end - gene.start
    labels = np.full(L, N_CLASSES - 1, dtype=np.int8)  # default: intergenic

    for exon_start, exon_end in gene.exons:
        rel_s = exon_start - gene.start
        rel_e = exon_end - gene.start
        for pos in range(rel_s, min(rel_e, L)):
            frame = (pos - rel_s) % 3
            labels[pos] = 4 + frame  # coding_f0/f1/f2

        # splice donor (GT): 2 nt at exon end
        if rel_e + 2 <= L:
            labels[rel_e] = 0
        # splice acceptor (AG): 2 nt before exon start
        if rel_s - 2 >= 0:
            labels[rel_s - 2] = 1

    return labels
