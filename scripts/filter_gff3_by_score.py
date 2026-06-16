#!/usr/bin/env python3
"""Filter a helion GFF3 by mean per-exon score stored in the gene feature's score column.

Usage:
    python filter_gff3_by_score.py <input.gff3> <output.gff3> <min_score>

helion write_gff3 places mean_exon_score in the score column (col 6) of gene features.
This script groups records by gene ID and drops all records (gene + mRNA + CDS) for
genes whose score is below min_score, without re-running prediction.
"""

import sys
from pathlib import Path


def filter_gff3(input_path: Path, output_path: Path, min_score: float) -> tuple[int, int]:
    """Return (genes_kept, genes_dropped)."""
    lines = input_path.read_text().splitlines(keepends=True)

    # First pass: collect mean_score per gene_id from gene feature lines.
    gene_scores: dict[str, float] = {}
    for line in lines:
        if line.startswith("#"):
            continue
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 9 or parts[2] != "gene":
            continue
        score_str = parts[5]
        try:
            score = float(score_str)
        except ValueError:
            score = 0.0
        attrs = parts[8]
        gene_id = ""
        for attr in attrs.split(";"):
            if attr.startswith("ID="):
                gene_id = attr[3:]
                break
        if gene_id:
            gene_scores[gene_id] = score

    kept = {gid for gid, s in gene_scores.items() if s >= min_score}
    dropped = len(gene_scores) - len(kept)

    # Second pass: write kept records.
    with output_path.open("w") as out:
        for line in lines:
            if line.startswith("#"):
                out.write(line)
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                out.write(line)
                continue
            attrs = parts[8]
            # Determine which gene this record belongs to.
            gene_id = ""
            for attr in attrs.split(";"):
                if attr.startswith("ID=gene_"):
                    gene_id = attr[3:]
                    break
                if attr.startswith("Parent=gene_"):
                    gene_id = attr[7:]
                    break
                # CDS/exon features have Parent=mrna_XXXXXX; find gene via prefix.
                if attr.startswith("Parent=mrna_"):
                    mrna_id = attr[7:]
                    gene_id = "gene_" + mrna_id[5:]  # mrna_000001 -> gene_000001
                    break
            if gene_id and gene_id not in kept:
                continue
            out.write(line)

    return len(kept), dropped


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <input.gff3> <output.gff3> <min_score>", file=sys.stderr)
        sys.exit(1)
    inp = Path(sys.argv[1])
    out = Path(sys.argv[2])
    threshold = float(sys.argv[3])
    kept, dropped = filter_gff3(inp, out, threshold)
    print(f"Kept {kept} genes, dropped {dropped} (min_score={threshold})")
