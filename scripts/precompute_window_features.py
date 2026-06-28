"""
Precompute HyenaDNA offset-cosine features for Helion training windows.

Builds the SignalDataset train + val window lists (deterministic) for a given
config, computes the 6 offset features for each window's oriented sequence via
HyenaDNA, and saves index-aligned (N, W, 6) arrays:
    <feature_dir>/train_features.npy
    <feature_dir>/val_features.npy

The dataset loads these by index, so the config here MUST match the training run
(window size, val/train chromosomes, neg fraction). Features are computed on the
exact oriented window sequence, so minus-strand windows are handled for free (no
coordinate/RC mapping).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

H = Path("/storage3/fs1/shandley/Active/helion")
if (H / "python").is_dir():
    sys.path.insert(0, str(H / "python"))
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from helion.features.hyenadna import compute_features_batch, load_hyenadna  # noqa: E402
from helion.signals.dataset import SignalDataset  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Precompute HyenaDNA window features")
    ap.add_argument("--genome", type=Path, required=True)
    ap.add_argument("--annotations", type=Path, required=True)
    ap.add_argument("--feature-dir", type=Path, required=True)
    ap.add_argument("--window-size", type=int, default=5000)
    ap.add_argument("--organism", type=str, default="vertebrate")
    ap.add_argument("--val-chromosomes", nargs="*", default=["chr22", "22"])
    ap.add_argument("--train-chromosomes", nargs="*", default=None)
    ap.add_argument("--neg-fraction", type=float, default=0.5)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    args.feature_dir.mkdir(parents=True, exist_ok=True)

    print("Building train windows ...", flush=True)
    train_ds = SignalDataset(
        args.genome, args.annotations, window_size=args.window_size, organism=args.organism,
        val_chromosomes=args.val_chromosomes, split="train", centered_sampling=True,
        neg_fraction=args.neg_fraction, train_chromosomes=args.train_chromosomes,
    )
    print("Building val windows ...", flush=True)
    val_ds = SignalDataset(
        args.genome, args.annotations, window_size=args.window_size, organism=args.organism,
        val_chromosomes=args.val_chromosomes, split="val", centered_sampling=False,
    )

    print(f"Loading HyenaDNA on {args.device} ...", flush=True)
    model, tokenizer = load_hyenadna(device=args.device)

    for name, ds in (("train", train_ds), ("val", val_ds)):
        seqs = ds.window_sequences()
        print(f"{name}: {len(seqs):,} windows -> features ...", flush=True)
        feats = compute_features_batch(
            seqs, model, tokenizer, device=args.device, batch_size=args.batch_size
        )
        out = args.feature_dir / f"{name}_features.npy"
        np.save(out, feats)
        print(f"  saved {out}  shape={feats.shape}", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
