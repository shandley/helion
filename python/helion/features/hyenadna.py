"""
HyenaDNA offset-cosine features for the DNA-embedding fusion experiment.

Vendored from the dna-embedding-gene-discovery project (the validated
"offset-3 inversion" signal). A DNA foundation model trained on next-nucleotide
prediction learns codon periodicity geometrically: in coding DNA, embeddings 3
positions apart (one codon) are more similar than adjacent ones. These 6
per-position features expose that signal as extra CNN input channels.

HyenaDNA (LongSafari/hyenadna-small-32k-seqlen-hf, ~3.6M params) is the open,
local, no-API model used for the cheap proof; the same features can later come
from Evo2 (stronger signal). Extraction mirrors test_hyenadna_inversion.py and
the genescan tool (tool/src/main.rs) exactly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    import torch

_MODEL_NAME = "LongSafari/hyenadna-small-32k-seqlen-hf"

# Per-position feature channels, in order (matches probe_model.json).
FEATURE_NAMES = ("cos1", "cos3", "inversion", "local_mean", "local_std", "local_gap")
N_FEATURES = len(FEATURE_NAMES)
_DEFAULT_WINDOW = 15


def load_hyenadna(device: str = "cpu") -> tuple[Any, Any]:
    """Load HyenaDNA model + tokenizer (requires transformers, trust_remote_code)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import]

    tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(_MODEL_NAME, trust_remote_code=True)
    model.eval()
    model.to(device)
    return model, tokenizer


def extract_embeddings(
    model: Any, tokenizer: Any, sequence: str, device: str = "cpu"
) -> npt.NDArray[np.float32]:
    """Per-position HyenaDNA embeddings, aligned 1:1 with `sequence`.

    Mirrors test_hyenadna_inversion.py: take the last hidden layer and strip the
    leading special token so row i corresponds exactly to nucleotide i.
    """
    import torch  # type: ignore[import]

    inputs = tokenizer(sequence, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
    emb = outputs.hidden_states[-1].squeeze(0).cpu().numpy()  # (n_tokens, hidden)
    if emb.shape[0] > len(sequence):
        emb = emb[1 : len(sequence) + 1]
    return emb.astype(np.float32)


def _cosine_offset(emb: npt.NDArray[np.float32], offset: int) -> npt.NDArray[np.float32]:
    """Per-position cosine similarity between embedding i and i+offset (0 where
    undefined: the last `offset` positions and any zero-norm rows)."""
    n = emb.shape[0]
    out = np.zeros(n, dtype=np.float32)
    if offset >= n:
        return out
    norms = np.linalg.norm(emb, axis=1)
    a, b = emb[: n - offset], emb[offset:]
    na, nb = norms[: n - offset], norms[offset:]
    denom = na * nb
    dots = np.einsum("ij,ij->i", a, b)
    valid = denom > 0
    head = out[: n - offset]
    head[valid] = dots[valid] / denom[valid]
    return out


def _windowed_means(x: npt.NDArray[np.float32], half: int) -> tuple[
    npt.NDArray[np.float32], npt.NDArray[np.float32]
]:
    """Per-position mean and count over the clamped window [i-half, i+half] using
    a prefix sum (exact, edge-correct, O(n))."""
    n = x.size
    csum = np.concatenate(([0.0], np.cumsum(x, dtype=np.float64)))
    idx = np.arange(n)
    s = np.maximum(0, idx - half)
    e = np.minimum(n, idx + half + 1)
    count = (e - s).astype(np.float64)
    total = csum[e] - csum[s]
    return (total / count).astype(np.float32), count.astype(np.float32)


def offset_features(
    emb: npt.NDArray[np.float32], window: int = _DEFAULT_WINDOW
) -> npt.NDArray[np.float32]:
    """(L, 6) per-position features. Mirrors genescan tool/src/main.rs:predict,
    vectorized via prefix sums.

    cos1, cos3: per-position offset cosines.  inversion: cos3 - cos1.
    local_mean/local_std: windowed mean/std of inversion (clamped window).
    local_gap: mean(cos3) - mean(cos1) over the window (== local_mean by algebra,
    kept for fidelity with the trained probe).
    """
    cos1 = _cosine_offset(emb, 1)
    cos3 = _cosine_offset(emb, 3)
    inv = cos3 - cos1
    half = window // 2
    local_mean, _ = _windowed_means(inv, half)
    mean_sq, _ = _windowed_means(inv * inv, half)
    local_std = np.sqrt(np.maximum(mean_sq - local_mean * local_mean, 0.0)).astype(np.float32)
    mean_c3, _ = _windowed_means(cos3, half)
    mean_c1, _ = _windowed_means(cos1, half)
    local_gap = (mean_c3 - mean_c1).astype(np.float32)
    return np.stack([cos1, cos3, inv, local_mean, local_std, local_gap], axis=1).astype(
        np.float32
    )


def compute_features(
    sequence: str, model: Any, tokenizer: Any, device: str = "cpu",
    window: int = _DEFAULT_WINDOW,
) -> npt.NDArray[np.float32]:
    """End-to-end: DNA string -> (len(sequence), 6) offset-cosine feature track."""
    emb = extract_embeddings(model, tokenizer, sequence, device)
    return offset_features(emb, window)


def compute_features_batch(
    sequences: list[str], model: Any, tokenizer: Any, device: str = "cpu",
    window: int = _DEFAULT_WINDOW, batch_size: int = 16,
) -> npt.NDArray[np.float32]:
    """Features for many equal-length windows. Returns (N, W, 6).

    Sequences must all be the same length W (training windows are). Batches the
    HyenaDNA forward pass; strips the leading special token per row.
    """
    import torch  # type: ignore[import]

    if not sequences:
        return np.zeros((0, 0, N_FEATURES), dtype=np.float32)
    w = len(sequences[0])
    out = np.zeros((len(sequences), w, N_FEATURES), dtype=np.float32)
    for b0 in range(0, len(sequences), batch_size):
        batch = sequences[b0 : b0 + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True)
        input_ids = enc["input_ids"].to(device)
        with torch.no_grad():
            hidden = model(input_ids, output_hidden_states=True).hidden_states[-1]
        hidden = hidden.cpu().numpy().astype(np.float32)  # (B, toks, hidden)
        for j, seq in enumerate(batch):
            emb = hidden[j]
            if emb.shape[0] > len(seq):
                emb = emb[1 : len(seq) + 1]
            out[b0 + j] = offset_features(emb, window)
    return out
