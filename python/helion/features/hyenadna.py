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


def offset_features(
    emb: npt.NDArray[np.float32], window: int = _DEFAULT_WINDOW
) -> npt.NDArray[np.float32]:
    """(L, 6) per-position features. Mirrors genescan tool/src/main.rs:predict.

    cos1, cos3: per-position offset cosines.  inversion: cos3 - cos1.
    local_mean/local_std: windowed mean/std of inversion.
    local_gap: mean(cos3) - mean(cos1) over the window (== local_mean by algebra,
    kept for fidelity with the trained probe).
    """
    n = emb.shape[0]
    cos1 = _cosine_offset(emb, 1)
    cos3 = _cosine_offset(emb, 3)
    inv = cos3 - cos1
    half = window // 2
    local_mean = np.zeros(n, dtype=np.float32)
    local_std = np.zeros(n, dtype=np.float32)
    local_gap = np.zeros(n, dtype=np.float32)
    for i in range(n):
        s, e = max(0, i - half), min(n, i + half + 1)
        win_inv = inv[s:e]
        local_mean[i] = win_inv.mean()
        local_std[i] = win_inv.std()
        local_gap[i] = cos3[s:e].mean() - cos1[s:e].mean()
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
