import tempfile
from pathlib import Path
import pytest
from helion.io.gff3 import write_gff3


class _MockExon:
    def __init__(self, start: int, end: int, frame: int = 0, score: float = 0.9) -> None:
        self.start = start
        self.end = end
        self.frame = frame
        self.score = score


class _MockModel:
    def __init__(self) -> None:
        self.seqid = "chr1"
        self.start = 100
        self.end = 500
        self.strand = "+"
        self.score = 0.95
        self.exons = [_MockExon(100, 200), _MockExon(300, 500)]

    def with_offset(self, offset: int) -> "_MockModel":
        return self


def test_write_gff3_creates_file() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "test.gff3"
        write_gff3([_MockModel()], out)  # type: ignore[list-item]
        assert out.exists()
        content = out.read_text()
        assert content.startswith("##gff-version 3")
        assert "chr1" in content
        assert "gene_000001" in content
        assert "CDS" in content


def test_write_gff3_coordinates_are_one_based() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "test.gff3"
        write_gff3([_MockModel()], out)  # type: ignore[list-item]
        lines = [l for l in out.read_text().splitlines() if "\tCDS\t" in l]
        assert len(lines) == 2
        first = lines[0].split("\t")
        # 0-based 100 -> GFF3 1-based 101
        assert first[3] == "101"
