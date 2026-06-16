from __future__ import annotations

from pathlib import Path
from typing import IO, TYPE_CHECKING

if TYPE_CHECKING:
    from helion._core import PyGeneModel as GeneModel


_GFF3_HEADER = "##gff-version 3\n"


def write_gff3(models: list[GeneModel], path: Path) -> None:
    with path.open("w") as f:
        f.write(_GFF3_HEADER)
        for i, model in enumerate(models, start=1):
            gene_id = f"gene_{i:06d}"
            mrna_id = f"mrna_{i:06d}"
            n_exons = len(model.exons)
            mean_score = model.score / n_exons if n_exons else 0.0
            _write_record(
                f, model.seqid, "gene", model.start, model.end, model.strand, gene_id,
                score=mean_score,
            )
            _write_record(
                f, model.seqid, "mRNA", model.start, model.end, model.strand, mrna_id,
                parent=gene_id, score=model.score,
            )
            for j, exon in enumerate(model.exons, start=1):
                exon_id = f"exon_{i:06d}_{j}"
                _write_record(
                    f, model.seqid, "CDS", exon.start, exon.end, model.strand, exon_id,
                    parent=mrna_id, phase=exon.frame,
                )


def _write_record(
    f: IO[str],
    seqid: str,
    feature: str,
    start: int,
    end: int,
    strand: str,
    id_: str,
    parent: str | None = None,
    score: float | None = None,
    phase: int = 0,
) -> None:
    score_str = f"{score:.4f}" if score is not None else "."
    attrs = f"ID={id_}"
    if parent:
        attrs += f";Parent={parent}"
    line = "\t".join([
        seqid, "helion", feature,
        str(start + 1), str(end),  # GFF3 is 1-based inclusive
        score_str, strand, str(phase), attrs,
    ])
    f.write(line + "\n")
