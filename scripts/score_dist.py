"""
Score distribution diagnostic: load each model, score chr2L (drosophila)
and chr1 (plant), and report percentile stats for each channel.
Run on a login node -- CPU only, short sequence.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

H = Path("/storage3/fs1/shandley/Active/helion")
sys.path.insert(0, str(H / "python"))

from helion.signals.model import SignalModel  # noqa: E402

MODELS = [
    ("drosophila", H / "models/drosophila", H / "results/drosophila_chr2L.fa"),
    ("plant_w2000", H / "models/plant_w2000", H / "results/plant_chr1.fa"),
]

CHANNELS = ["donor", "acceptor", "start", "stop", "coding_f0", "coding_f1", "coding_f2", "intergenic"]
PERCENTILES = [50, 90, 95, 99, 99.9, 100]


def load_fasta_first_50k(fa_path: Path) -> str:
    lines = []
    with fa_path.open() as f:
        for line in f:
            if line.startswith(">"):
                continue
            lines.append(line.strip())
            if sum(len(l) for l in lines) >= 50_000:
                break
    return "".join(lines)[:50_000]


def main() -> None:
    for name, model_path, fa_path in MODELS:
        print(f"\n{'='*60}")
        print(f"Model: {name}  ({model_path})")
        print(f"Sequence: {fa_path.name} (first 50kb)")

        model = SignalModel.load(model_path, device="cpu")
        seq = load_fasta_first_50k(fa_path)
        scores = model.score(seq)

        # Build (N_CLASSES, L) array
        arr = np.stack([
            scores.donor,
            scores.acceptor,
            scores.start,
            scores.stop,
            scores.coding[:, 0],
            scores.coding[:, 1],
            scores.coding[:, 2],
            1.0 - scores.donor - scores.acceptor - scores.start - scores.stop
            - scores.coding[:, 0] - scores.coding[:, 1] - scores.coding[:, 2],
        ], axis=0)  # (8, L)

        print(f"\n{'Channel':<14}", end="")
        for p in PERCENTILES:
            print(f"  p{p:>5}", end="")
        print(f"  {'max>0.1':>8}  {'max>0.01':>9}")

        for i, ch in enumerate(CHANNELS):
            row = arr[i]
            pcts = np.percentile(row, PERCENTILES)
            n_above_01 = int((row > 0.1).sum())
            n_above_001 = int((row > 0.01).sum())
            print(f"{ch:<14}", end="")
            for v in pcts:
                print(f"  {v:>7.4f}", end="")
            print(f"  {n_above_01:>8}  {n_above_001:>9}")


if __name__ == "__main__":
    main()
