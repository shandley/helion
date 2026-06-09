from pathlib import Path

import torch  # type: ignore[import]
import torch.nn as nn  # type: ignore[import]
from torch.utils.data import DataLoader  # type: ignore[import]

from helion.signals.dataset import SignalDataset
from helion.signals.model import SignalModel

# Chromosome hold-out conventions per organism.
# These are small, well-annotated chromosomes that make clean eval sets.
_DEFAULT_VAL_CHROMS: dict[str, list[str]] = {
    "insect":     ["chr4", "4"],           # Drosophila chr4 (tiny, ~1.3Mb)
    "vertebrate": ["chr22", "22"],         # Human chr22 (small, well-annotated)
    "plant":      ["Chr4", "4"],           # Arabidopsis Chr4
    "fungus":     [],                      # too few chromosomes -- fall back to random
}


def train_model(
    annotations: Path,
    genome: Path,
    output: Path,
    organism: str = "vertebrate",
    epochs: int = 50,
    lr: float = 1e-4,
    batch_size: int = 32,
    val_fraction: float = 0.1,
    val_chromosomes: list[str] | None = None,
    window_size: int = 5000,
    channels: int = 256,
    device: str = "cpu",
) -> SignalModel:
    """
    Train a Helion signal model.

    Validation split is chromosome-level by default to prevent data
    leakage from windows that share sequence context. Pass
    val_chromosomes explicitly to override the per-organism defaults,
    or pass val_chromosomes=[] to fall back to random window splitting
    (not recommended except for small genomes / fungi).
    """
    if val_chromosomes is None:
        val_chromosomes = _DEFAULT_VAL_CHROMS.get(organism, [])

    if val_chromosomes:
        train_ds = SignalDataset(
            genome, annotations, window_size=window_size, organism=organism,
            val_chromosomes=val_chromosomes, split="train",
        )
        val_ds = SignalDataset(
            genome, annotations, window_size=window_size, organism=organism,
            val_chromosomes=val_chromosomes, split="val",
        )
        print(f"Val chromosomes: {val_chromosomes}", flush=True)
        print(f"Train windows: {len(train_ds):,}  Val windows: {len(val_ds):,}", flush=True)
    else:
        # Fallback: random window split (leaky but acceptable for fungi)
        from torch.utils.data import random_split
        dataset = SignalDataset(genome, annotations, window_size=window_size, organism=organism)
        n_val = max(1, int(len(dataset) * val_fraction))
        train_ds, val_ds = random_split(dataset, [len(dataset) - n_val, n_val])
        print(f"Random split -- Train: {len(train_ds):,}  Val: {len(val_ds):,}", flush=True)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=(device != "cpu"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        num_workers=4, pin_memory=(device != "cpu"),
    )

    model = SignalModel(channels=channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        val_loss = _val_epoch(model, val_loader, loss_fn, device)
        scheduler.step()

        print(
            f"epoch {epoch:3d}/{epochs}  "
            f"train={train_loss / len(train_loader):.4f}  "
            f"val={val_loss:.4f}",
            flush=True,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save(output)
            print(f"  saved to {output}", flush=True)

    return model


def _val_epoch(
    model: SignalModel,
    loader: DataLoader,
    loss_fn: nn.CrossEntropyLoss,
    device: str,
) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            total += loss.item()
    return total / max(len(loader), 1)
