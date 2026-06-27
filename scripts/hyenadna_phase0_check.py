"""
Phase 0 sanity check for the DNA-embedding fusion experiment.

Two questions, no training:
  1. ALIGNMENT  -- does the vendored HyenaDNA feature track line up 1:1 with
     Helion's nucleotide coordinates? (len(features) == len(sequence))
  2. SIGNAL     -- does the offset-3 inversion actually separate coding from
     non-coding positions on a real chr22 gene, in Helion's coordinate frame?

Picks a multi-exon, plus-strand, protein-coding gene on chr22 from the reference
GFF3, scores its locus with HyenaDNA, and compares the inversion feature inside
CDS exons vs introns/flanks. If coding >> non-coding, the signal is present and
correctly aligned -- greenlight to build the fusion dataset/model (Phase 1).

Run as a GPU sbatch job (needs torch + transformers + the HyenaDNA weights).
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

H = Path("/storage3/fs1/shandley/Active/helion")
if (H / "python").is_dir():
    sys.path.insert(0, str(H / "python"))
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from helion.features.hyenadna import compute_features, load_hyenadna  # noqa: E402

DEFAULT_GENOME = H / "results/vertebrate_chr22.fa"
DEFAULT_REF = H / "results/ref_vertebrate_chr22.gff3"


def _attr(attrs: str, key: str) -> str | None:
    for part in attrs.split(";"):
        if part.startswith(f"{key}="):
            return part[len(key) + 1 :]
    return None


def pick_gene(ref: Path, min_exons: int, min_span: int, max_span: int) -> tuple[str, int, int, list[tuple[int, int]]]:
    """Find a plus-strand transcript with >= min_exons CDS exons whose locus span
    is in [min_span, max_span]. Returns (seqid, start, end, cds_intervals) in
    0-based half-open coords. cds_intervals are relative to the locus start."""
    cds_by_tx: dict[str, list[tuple[str, int, int, str]]] = defaultdict(list)
    with ref.open() as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.rstrip().split("\t")
            if len(p) < 9 or p[2] != "CDS":
                continue
            parent = _attr(p[8], "Parent") or ""
            cds_by_tx[parent].append((p[0], int(p[3]) - 1, int(p[4]), p[6]))
    for tx, cds in sorted(cds_by_tx.items()):
        if len({c[0] for c in cds}) != 1 or any(c[3] != "+" for c in cds):
            continue
        if len(cds) < min_exons:
            continue
        seqid = cds[0][0]
        start = min(c[1] for c in cds)
        end = max(c[2] for c in cds)
        span = end - start
        if not (min_span <= span <= max_span):
            continue
        rel = sorted((c[1] - start, c[2] - start) for c in cds)
        return seqid, start, end, rel
    raise SystemExit("No suitable plus-strand multi-exon gene found in range.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 0: HyenaDNA feature alignment + signal check")
    ap.add_argument("--genome", type=Path, default=DEFAULT_GENOME)
    ap.add_argument("--ref", type=Path, default=DEFAULT_REF)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--min-exons", type=int, default=4)
    ap.add_argument("--min-span", type=int, default=3000)
    ap.add_argument("--max-span", type=int, default=15000)
    args = ap.parse_args()

    from pyfaidx import Fasta  # type: ignore[import]

    seqid, start, end, cds_rel = pick_gene(args.ref, args.min_exons, args.min_span, args.max_span)
    fasta = Fasta(str(args.genome))
    locus = str(fasta[seqid][start:end]).upper()
    L = len(locus)
    print(f"Gene locus: {seqid}:{start}-{end}  ({L} bp, {len(cds_rel)} CDS exons, + strand)")

    print(f"Loading HyenaDNA on {args.device} ...", flush=True)
    model, tokenizer = load_hyenadna(device=args.device)
    print("Computing offset-cosine features ...", flush=True)
    feats = compute_features(locus, model, tokenizer, device=args.device)

    # 1. ALIGNMENT
    print("\n=== ALIGNMENT ===")
    print(f"  sequence length: {L}")
    print(f"  feature shape:   {feats.shape}")
    aligned = feats.shape == (L, 6)
    print(f"  aligned 1:1:     {aligned}")

    # 2. SIGNAL: coding vs non-coding inversion (feature column 2)
    coding = np.zeros(L, dtype=bool)
    for s, e in cds_rel:
        coding[s:e] = True
    inv = feats[:, 2]
    # ignore the tail where offset cosines are undefined (last 3 positions)
    valid = np.ones(L, dtype=bool)
    valid[-3:] = False
    c = inv[coding & valid]
    nc = inv[~coding & valid]
    print("\n=== SIGNAL (offset-3 inversion = cos3 - cos1) ===")
    print(f"  coding positions:     {c.size:,}   mean inversion {c.mean():+.4f}")
    print(f"  non-coding positions: {nc.size:,}   mean inversion {nc.mean():+.4f}")
    print(f"  separation (coding - non-coding): {c.mean() - nc.mean():+.4f}")
    print(f"  coding fraction with inversion>0:     {(c > 0).mean():.1%}")
    print(f"  non-coding fraction with inversion>0: {(nc > 0).mean():.1%}")

    signal = c.mean() > nc.mean()
    print("\n=== VERDICT ===")
    print(f"  ALIGNED:        {'PASS' if aligned else 'FAIL'}")
    print(f"  SIGNAL PRESENT: {'PASS' if signal else 'FAIL'} (coding inversion higher than non-coding)")
    print("  -> Phase 1 greenlit." if (aligned and signal) else "  -> investigate before Phase 1.")


if __name__ == "__main__":
    main()
