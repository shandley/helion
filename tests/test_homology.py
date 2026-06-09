import numpy as np
import pytest
from helion.homology.align import translate, score_exon_homology
from helion.signals.model import SignalScores


def test_translate_basic() -> None:
    assert translate("ATG") == "M"
    assert translate("TAA") == "*"
    assert translate("ATGTAA") == "M*"


def test_translate_unknown_codon() -> None:
    result = translate("NNN")
    assert result == "X"


def test_translate_partial_codon_ignored() -> None:
    result = translate("ATGAT")
    assert result == "M"  # partial codon dropped


def test_score_exon_homology_shape() -> None:
    L = 300
    seq = "ATGC" * (L // 4)
    scores = SignalScores(
        donor=np.zeros(L, dtype=np.float32),
        acceptor=np.zeros(L, dtype=np.float32),
        start=np.zeros(L, dtype=np.float32),
        stop=np.zeros(L, dtype=np.float32),
        coding=np.zeros((L, 3), dtype=np.float32),
    )
    n_residues = 50
    embed_dim = 320
    protein_embedding = np.random.randn(n_residues, embed_dim).astype(np.float32)

    result = score_exon_homology(seq, scores, protein_embedding)
    assert result.shape == (L,)
    assert result.dtype == np.float32
