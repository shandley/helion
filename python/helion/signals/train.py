from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from helion.signals.dataset import SignalDataset
from helion.signals.model import SignalModel


def train_model(
    annotations: Path,
    genome: Path,
    output: Path,
    organism: str = "vertebrate",
    epochs: int = 50,
    lr: float = 1e-4,
    batch_size: int = 32,
    val_fraction: float = 0.1,
    device: str = "cpu",
) -> SignalModel:
    dataset = SignalDataset(genome, annotations, organism=organism)

    n_val = max(1, int(len(dataset) * val_fraction))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=4)

    model = SignalModel().to(device)
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
            # logits: (B, N_CLASSES, L), y: (B, L)
            loss = loss_fn(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        val_loss = _evaluate(model, val_loader, loss_fn, device)
        scheduler.step()

        print(
            f"epoch {epoch:3d}/{epochs}  "
            f"train={train_loss / len(train_loader):.4f}  "
            f"val={val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save(output)
            print(f"  saved to {output}")

    return model


def _evaluate(
    model: SignalModel,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
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
