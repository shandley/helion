from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn


# Per-position output channels (sense-strand only; RC inference handles minus strand)
# Order: donor, acceptor, start, stop, coding_f0, coding_f1, coding_f2, intergenic
N_CLASSES = 8
CODING_CHANNELS = slice(4, 7)  # coding_f0, f1, f2
INTERGENIC_CHANNEL = 7

# Signed-distance regression head: clamp window (nt) for distance-to-boundary
# targets. The field is ~0 at a splice site, negative upstream, positive
# downstream, saturating at +/-DIST_W. 32 nt is wide enough to distinguish the
# true site from a wrong GT/AG tens of nt away while keeping a sharp 0-crossing.
DIST_W = 32


@dataclass
class SignalScores:
    """Per-nucleotide signal probabilities for one strand."""

    donor: npt.NDArray[np.float32]      # (seq_len,)
    acceptor: npt.NDArray[np.float32]   # (seq_len,)
    start: npt.NDArray[np.float32]      # (seq_len,)
    stop: npt.NDArray[np.float32]       # (seq_len,)
    coding: npt.NDArray[np.float32]     # (seq_len, 3)  -- frame 0/1/2
    intergenic: npt.NDArray[np.float32] # (seq_len,)
    # Signed-distance regression head outputs (nt), or None if the model has no
    # distance head. ~0 at a true donor/acceptor, negative upstream, positive
    # downstream (denormalised back to nt from the [-1, 1] training target).
    d_donor: npt.NDArray[np.float32] | None = None
    d_acceptor: npt.NDArray[np.float32] | None = None


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=padding)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=padding)
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.relu(x + residual)


class SignalModel(nn.Module):
    """
    Dilated residual CNN for per-nucleotide gene signal prediction.

    Processes one-hot DNA (4 channels) and outputs per-position probabilities
    for splice donors, acceptors, start/stop codons, and coding frame.

    Dilation schedule captures signals at multiple scales:
      - 1, 2, 4: local splice site consensus (~10-20 nt)
      - 8, 16:   branch point and polypyrimidine tract (~50-100 nt)
      - 32, 64:  exon-level compositional context (~200-400 nt)
    """

    DILATION_SCHEDULE = [1, 2, 4, 8, 16, 32, 64, 1, 2, 4, 8, 16, 32, 64]

    def __init__(
        self, channels: int = 256, kernel_size: int = 11, use_distance_head: bool = False,
        in_channels: int = 4,
    ) -> None:
        super().__init__()
        # in_channels = 4 (one-hot DNA) by default; 10 when DNA-embedding offset
        # features are fused in (4 one-hot + 6 HyenaDNA offset-cosine channels).
        self.in_channels = in_channels
        self.embed = nn.Conv1d(in_channels, channels, kernel_size=1)
        self.blocks = nn.ModuleList([
            ResidualBlock(channels, kernel_size, d) for d in self.DILATION_SCHEDULE
        ])
        self.head = nn.Conv1d(channels, N_CLASSES, kernel_size=1)
        # Optional signed-distance regression head: 2 channels (donor, acceptor),
        # raw output trained against [-1, 1]-normalised distance targets.
        self.use_distance_head = use_distance_head
        self.dist_head = nn.Conv1d(channels, 2, kernel_size=1) if use_distance_head else None

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        return x

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        # x: (batch, 4, seq_len) one-hot DNA
        feat = self._features(x)
        probs = torch.softmax(self.head(feat), dim=1)  # (batch, N_CLASSES, seq_len)
        dist = self.dist_head(feat) if self.dist_head is not None else None  # (batch, 2, seq_len)
        return probs, dist

    def score(
        self, sequence: str, features: npt.NDArray[np.float32] | None = None
    ) -> SignalScores:
        """Run inference on a single sequence string, return per-position scores.

        `features` is an optional (L, n_extra) DNA-embedding feature track that is
        concatenated onto the one-hot input (required for fused 10-channel models).
        """
        x = _one_hot(sequence)  # (4, L)
        if features is not None:
            feat = torch.from_numpy(np.ascontiguousarray(features)).T.float()  # (n_extra, L)
            x = torch.cat([x, feat], dim=0)
        x = x.unsqueeze(0)  # (1, in_channels, L)
        device = next(self.parameters()).device
        x = x.to(device)
        with torch.no_grad():
            probs_t, dist_t = self.forward(x)
        probs = probs_t[0].cpu().numpy()  # (N_CLASSES, L)
        d_donor = d_acceptor = None
        if dist_t is not None:
            dist = dist_t[0].cpu().numpy()  # (2, L), normalised [-1, 1]
            d_donor = dist[0] * DIST_W      # denormalise to nt
            d_acceptor = dist[1] * DIST_W
        return SignalScores(
            donor=probs[0],
            acceptor=probs[1],
            start=probs[2],
            stop=probs[3],
            coding=probs[4:7].T,  # (L, 3)
            intergenic=probs[7],
            d_donor=d_donor,
            d_acceptor=d_acceptor,
        )

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "weights.pt")

    @classmethod
    def load(cls, path: Path, organism: str = "vertebrate", device: str = "cpu") -> SignalModel:
        weights_path = path / "weights.pt"
        state = torch.load(weights_path, map_location=device, weights_only=True)
        # Auto-detect the head and input width from the checkpoint so old (v3/v4)
        # checkpoints load with strict matching and new ones build their head /
        # fused input conv.
        use_distance_head = any(k.startswith("dist_head") for k in state)
        in_channels = state["embed.weight"].shape[1] if "embed.weight" in state else 4
        model = cls(use_distance_head=use_distance_head, in_channels=in_channels)
        model.load_state_dict(state)
        model.eval()
        return model.to(device)


_DNA_MAP: dict[str, int] = {"A": 0, "C": 1, "G": 2, "T": 3}


def _one_hot(sequence: str) -> torch.Tensor:
    seq = sequence.upper()
    t = torch.zeros(4, len(seq), dtype=torch.float32)
    for i, base in enumerate(seq):
        idx = _DNA_MAP.get(base, -1)
        if idx >= 0:
            t[idx, i] = 1.0
    return t
