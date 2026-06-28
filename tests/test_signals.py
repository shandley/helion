import torch
import pytest
from helion.signals.model import SignalModel, SignalScores, _one_hot, N_CLASSES


def test_one_hot_shape() -> None:
    t = _one_hot("ACGT")
    assert t.shape == (4, 4)
    assert t[0, 0] == 1.0  # A
    assert t[1, 1] == 1.0  # C
    assert t[2, 2] == 1.0  # G
    assert t[3, 3] == 1.0  # T


def test_one_hot_unknown_base() -> None:
    t = _one_hot("N")
    assert t.sum() == 0.0


def test_model_output_shape() -> None:
    model = SignalModel(channels=32)
    model.eval()
    L = 200
    x = torch.zeros(1, 4, L)
    with torch.no_grad():
        probs, dist = model(x)
    assert probs.shape == (1, N_CLASSES, L)
    assert dist is None  # no distance head by default


def test_model_output_is_probability() -> None:
    model = SignalModel(channels=32)
    model.eval()
    x = torch.zeros(2, 4, 100)
    with torch.no_grad():
        probs, _ = model(x)
    sums = probs.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_fused_input_channels() -> None:
    model = SignalModel(channels=32, in_channels=10)
    model.eval()
    L = 120
    x = torch.zeros(1, 10, L)
    with torch.no_grad():
        probs, _ = model(x)
    assert probs.shape == (1, N_CLASSES, L)
    # score() with a (L, 6) feature track builds the 10-channel input
    import numpy as np
    feats = np.zeros((L, 6), dtype=np.float32)
    scores = model.score("ACGT" * 30, features=feats)
    assert scores.donor.shape == (L,)


def test_distance_head_output_shape() -> None:
    model = SignalModel(channels=32, use_distance_head=True)
    model.eval()
    L = 200
    x = torch.zeros(1, 4, L)
    with torch.no_grad():
        probs, dist = model(x)
    assert probs.shape == (1, N_CLASSES, L)
    assert dist is not None and dist.shape == (1, 2, L)


def test_score_returns_signal_scores() -> None:
    model = SignalModel(channels=32)
    model.eval()
    scores = model.score("ATGCATGCATGC" * 10)
    assert isinstance(scores, SignalScores)
    assert scores.donor.shape == (120,)
    assert scores.coding.shape == (120, 3)
    assert scores.d_donor is None  # no distance head


def test_score_with_distance_head() -> None:
    model = SignalModel(channels=32, use_distance_head=True)
    model.eval()
    scores = model.score("ATGCATGCATGC" * 10)
    assert scores.d_donor is not None and scores.d_donor.shape == (120,)
    assert scores.d_acceptor is not None and scores.d_acceptor.shape == (120,)
