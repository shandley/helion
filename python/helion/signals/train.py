from pathlib import Path

import torch  # type: ignore[import]
import torch.nn as nn  # type: ignore[import]
import torch.nn.functional as F  # type: ignore[import]
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


def _boundary_weighted_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    class_weights: torch.Tensor,
    emphasis: float,
    radius: int,
    exclude_labels: bool = False,
) -> torch.Tensor:
    """
    Per-position weighted cross-entropy that upweights boundary neighborhoods.

    With emphasis <= 0 this is bit-identical to nn.CrossEntropyLoss(
    weight=class_weights) default-mean reduction. With emphasis > 0, every
    position within +/-radius of a boundary label (donor=0, acceptor=1,
    start=2, stop=3) is weighted by (1 + emphasis), forcing the model to
    localize boundaries sharply.

    F.cross_entropy(..., reduction="none") already folds in the class weight,
    so per_pos = class_weight[y] * CE. nn.CrossEntropyLoss(reduction="mean")
    normalizes by the summed true-class weights (not the position count), so we
    replicate that denominator to keep emphasis=0 identical and the emphasis>0
    case a proper weighted average.
    """
    per_pos = F.cross_entropy(logits, y, weight=class_weights, reduction="none")  # (B, L)
    cw_y = class_weights[y]  # (B, L) true-class weight at each position
    if emphasis <= 0.0:
        return per_pos.sum() / cw_y.sum().clamp_min(1e-8)
    is_bnd = (y <= 3).float()  # boundary labels (donor/acceptor/start/stop) -> (B, L)
    dilated = F.max_pool1d(
        is_bnd.unsqueeze(1), kernel_size=2 * radius + 1, stride=1, padding=radius
    ).squeeze(1)  # (B, L): 1 within +/-radius of any boundary label
    if exclude_labels:
        # Emphasize only the coding/intergenic TRANSITION positions adjacent to a
        # boundary, not the splice/codon labels themselves -- those already carry
        # large inverse-frequency class weights, and double-weighting them makes
        # the boundary channels hyperactive (FP explosion at emphasis=5). This
        # targets the coding->intron cutoff that must turn off sharply.
        mask = dilated * (1.0 - is_bnd)
    else:
        mask = dilated
    w = 1.0 + emphasis * mask
    return (w * per_pos).sum() / (w * cw_y).sum().clamp_min(1e-8)


def _compute_loss(
    probs: torch.Tensor,
    dist_pred: torch.Tensor | None,
    y: torch.Tensor,
    dist_target: torch.Tensor,
    class_weights: torch.Tensor,
    emphasis: float,
    radius: int,
    exclude_labels: bool,
    distance_weight: float,
) -> torch.Tensor:
    """Classification (boundary-weighted CE) plus optional signed-distance
    regression (smooth-L1 against the [-1, 1] distance targets)."""
    loss = _boundary_weighted_loss(probs, y, class_weights, emphasis, radius, exclude_labels)
    if dist_pred is not None:
        loss = loss + distance_weight * F.smooth_l1_loss(dist_pred, dist_target)
    return loss


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
    workers: int = 4,
    neg_fraction: float = 0.0,
    boundary_emphasis: float = 0.0,
    boundary_radius: int = 3,
    boundary_exclude_labels: bool = False,
    distance_head: bool = False,
    distance_weight: float = 1.0,
    train_chromosomes: list[str] | None = None,
    feature_dir: Path | None = None,
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

    train_feat = (feature_dir / "train_features.npy") if feature_dir else None
    val_feat = (feature_dir / "val_features.npy") if feature_dir else None
    if val_chromosomes:
        train_ds = SignalDataset(
            genome, annotations, window_size=window_size, organism=organism,
            val_chromosomes=val_chromosomes, split="train",
            centered_sampling=True, neg_fraction=neg_fraction,
            train_chromosomes=train_chromosomes, feature_path=train_feat,
        )
        val_ds = SignalDataset(
            genome, annotations, window_size=window_size, organism=organism,
            val_chromosomes=val_chromosomes, split="val",
            centered_sampling=False, feature_path=val_feat,
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

    # Class weights from actual training windows (inverse frequency, capped 100x).
    # Compensates for rare signal classes (donor/acceptor/start/stop) being swamped
    # by intergenic and coding positions in the gradient.
    base_ds: SignalDataset = train_ds if isinstance(train_ds, SignalDataset) else train_ds.dataset  # type: ignore[assignment]
    class_weights = base_ds.compute_class_weights().to(device)
    print(f"Class weights: {[f'{w:.2f}' for w in class_weights.tolist()]}", flush=True)
    print(
        f"Boundary emphasis: {boundary_emphasis:.2f}  radius: {boundary_radius}"
        + (f"  exclude_labels={boundary_exclude_labels}" if boundary_emphasis > 0.0 else "  (disabled)"),
        flush=True,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=(device != "cpu"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        num_workers=workers, pin_memory=(device != "cpu"),
    )

    print(
        f"Distance head: {distance_head}"
        + (f"  weight={distance_weight:.2f}" if distance_head else ""),
        flush=True,
    )

    in_channels = 10 if feature_dir is not None else 4
    print(f"Input channels: {in_channels}" + ("  (one-hot + 6 DNA-embedding features)" if in_channels == 10 else ""), flush=True)
    model = SignalModel(
        channels=channels, use_distance_head=distance_head, in_channels=in_channels
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y, dist in train_loader:
            x, y, dist = x.to(device), y.to(device), dist.to(device)
            optimizer.zero_grad()
            probs, dist_pred = model(x)
            loss = _compute_loss(
                probs, dist_pred, y, dist, class_weights,
                boundary_emphasis, boundary_radius, boundary_exclude_labels,
                distance_weight,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        val_loss = _val_epoch(
            model, val_loader, class_weights, boundary_emphasis, boundary_radius,
            boundary_exclude_labels, distance_weight, device
        )
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
    class_weights: torch.Tensor,
    boundary_emphasis: float,
    boundary_radius: int,
    boundary_exclude_labels: bool,
    distance_weight: float,
    device: str,
) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for x, y, dist in loader:
            x, y, dist = x.to(device), y.to(device), dist.to(device)
            probs, dist_pred = model(x)
            loss = _compute_loss(
                probs, dist_pred, y, dist, class_weights,
                boundary_emphasis, boundary_radius, boundary_exclude_labels,
                distance_weight,
            )
            total += loss.item()
    return total / max(len(loader), 1)
