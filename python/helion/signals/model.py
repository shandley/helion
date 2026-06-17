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


@dataclass
class SignalScores:
    """Per-nucleotide signal probabilities for one strand."""

    donor: npt.NDArray[np.float32]      # (seq_len,)
    acceptor: npt.NDArray[np.float32]   # (seq_len,)
    start: npt.NDArray[np.float32]      # (seq_len,)
    stop: npt.NDArray[np.float32]       # (seq_len,)
    coding: npt.NDArray[np.float32]     # (seq_len, 3)  -- frame 0/1/2
    intergenic: npt.NDArray[np.float32] # (seq_len,)


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

    def __init__(self, channels: int = 256, kernel_size: int = 11) -> None:
        super().__init__()
        self.embed = nn.Conv1d(4, channels, kernel_size=1)
        self.blocks = nn.ModuleList([
            ResidualBlock(channels, kernel_size, d) for d in self.DILATION_SCHEDULE
        ])
        self.head = nn.Conv1d(channels, N_CLASSES, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 4, seq_len) one-hot DNA
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        return torch.softmax(self.head(x), dim=1)  # (batch, N_CLASSES, seq_len)

    def score(self, sequence: str) -> SignalScores:
        """Run inference on a single sequence string, return per-position scores."""
        x = _one_hot(sequence).unsqueeze(0)  # (1, 4, L)
        device = next(self.parameters()).device
        x = x.to(device)
        with torch.no_grad():
            probs = self.forward(x)[0].cpu().numpy()  # (N_CLASSES, L)
        return SignalScores(
            donor=probs[0],
            acceptor=probs[1],
            start=probs[2],
            stop=probs[3],
            coding=probs[4:7].T,  # (L, 3)
            intergenic=probs[7],
        )

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "weights.pt")

    @classmethod
    def load(cls, path: Path, organism: str = "vertebrate", device: str = "cpu") -> SignalModel:
        model = cls()
        weights_path = path / "weights.pt"
        state = torch.load(weights_path, map_location=device, weights_only=True)
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
