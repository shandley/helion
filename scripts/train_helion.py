"""
Training entry point for Helion signal models.

Called by the SLURM training scripts. Works without the compiled Rust
extension -- only the signals and io modules are needed for training.

Usage:
    python train_helion.py \\
        --annotations /path/to/annotation.gff3 \\
        --genome /path/to/genome.fa \\
        --output /path/to/models/drosophila \\
        --organism insect \\
        --epochs 50 \\
        --batch-size 128 \\
        --device cuda
"""

import argparse
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a Helion signal model")
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--genome", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--organism",
        default="insect",
        choices=["vertebrate", "insect", "plant", "fungus"],
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--window-size", type=int, default=5000)
    p.add_argument("--channels", type=int, default=256)
    p.add_argument("--device", default="cuda")
    p.add_argument("--workers", type=int, default=4)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Validate inputs before importing torch (faster failure)
    if not args.annotations.exists():
        sys.exit(f"Annotations not found: {args.annotations}")
    if not args.genome.exists():
        sys.exit(f"Genome not found: {args.genome}")

    import torch
    from helion.signals.train import train_model

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU", flush=True)
        device = "cpu"

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB", flush=True)

    print(f"Organism:   {args.organism}", flush=True)
    print(f"Epochs:     {args.epochs}", flush=True)
    print(f"Batch size: {args.batch_size}", flush=True)
    print(f"Device:     {device}", flush=True)
    print(f"Genome:     {args.genome}", flush=True)
    print(f"GFF3:       {args.annotations}", flush=True)
    print(f"Output:     {args.output}", flush=True)
    print("", flush=True)

    t0 = time.time()

    train_model(
        annotations=args.annotations,
        genome=args.genome,
        output=args.output,
        organism=args.organism,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        window_size=args.window_size,
        channels=args.channels,
        device=device,
    )

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed / 3600:.2f}h", flush=True)
    print(f"Weights saved to: {args.output}", flush=True)


if __name__ == "__main__":
    main()
