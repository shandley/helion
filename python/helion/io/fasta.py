from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from pyfaidx import Fasta  # type: ignore[import]


@dataclass
class GenomicWindow:
    seqid: str
    start: int   # 0-based, genomic
    end: int
    sequence: str

    @property
    def offset(self) -> int:
        return self.start


def read_sequence(genome: Path, seqid: str, start: int, end: int) -> str:
    fa = Fasta(str(genome))
    return str(fa[seqid][start:end])


def read_windows(
    genome: Path,
    window_size: int = 10_000,
    overlap: int = 500,
) -> Iterator[GenomicWindow]:
    """Yield overlapping windows across every sequence in the genome."""
    fa = Fasta(str(genome))
    step = window_size - overlap

    for seqid in fa.keys():
        seq_len = len(fa[seqid])
        for start in range(0, seq_len, step):
            end = min(start + window_size, seq_len)
            sequence = str(fa[seqid][start:end])
            yield GenomicWindow(seqid=seqid, start=start, end=end, sequence=sequence)
