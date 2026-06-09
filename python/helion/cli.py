import argparse
import sys
from pathlib import Path

from helion.predict import predict


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="helion",
        description="Gene prediction using deep signal models and protein homology.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    pred = subparsers.add_parser("predict", help="Predict genes in a genome")
    pred.add_argument("genome", type=Path, help="Repeat-masked genome FASTA")
    pred.add_argument("output", type=Path, help="Output GFF3 file")
    pred.add_argument("--model", type=Path, required=True, help="Model weights directory")
    pred.add_argument("--protein", type=Path, default=None, help="Homologous protein FASTA")
    pred.add_argument(
        "--organism",
        default="vertebrate",
        choices=["vertebrate", "insect", "plant", "fungus"],
    )
    pred.add_argument("--device", default="cpu", help="Torch device (cpu, cuda, mps)")
    pred.add_argument("--batch-size", type=int, default=4)
    pred.add_argument("--threshold", type=float, default=0.1)

    train = subparsers.add_parser("train", help="Train a signal model")
    train.add_argument("annotations", type=Path, help="GFF3 annotation file")
    train.add_argument("genome", type=Path, help="Genome FASTA matching annotations")
    train.add_argument("output", type=Path, help="Directory to save model weights")
    train.add_argument("--organism", default="vertebrate")
    train.add_argument("--epochs", type=int, default=50)
    train.add_argument("--lr", type=float, default=1e-4)
    train.add_argument("--device", default="cpu")

    ev = subparsers.add_parser("evaluate", help="Evaluate predictions against a reference")
    ev.add_argument("reference", type=Path, help="Reference annotation GFF3")
    ev.add_argument("predictions", type=Path, help="Helion prediction GFF3")
    ev.add_argument("genome", type=Path, help="Genome FASTA (must have .fai index)")

    args = parser.parse_args()

    if args.command == "predict":
        predict(
            genome=args.genome,
            output=args.output,
            model_weights=args.model,
            protein=args.protein,
            organism=args.organism,
            device=args.device,
            batch_size=args.batch_size,
            threshold=args.threshold,
        )
    elif args.command == "train":
        from helion.signals.train import train_model
        train_model(
            annotations=args.annotations,
            genome=args.genome,
            output=args.output,
            organism=args.organism,
            epochs=args.epochs,
            lr=args.lr,
            device=args.device,
        )
    elif args.command == "evaluate":
        from helion.evaluate import evaluate, format_report
        result = evaluate(
            ref_gff3=args.reference,
            pred_gff3=args.predictions,
            genome=args.genome,
        )
        print(format_report(result))
    else:
        parser.print_help()
        sys.exit(1)
