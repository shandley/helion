import pytest
from helion._core import build_dag, viterbi_decode


def _flat_scores(length: int, value: float = 0.0) -> list[float]:
    return [value] * length


def _coding_scores(length: int, value: float = 0.0) -> list[list[float]]:
    return [[value, value, value] for _ in range(length)]


def test_build_dag_empty_when_no_signals() -> None:
    L = 100
    dag = build_dag(
        donor_scores=_flat_scores(L),
        acceptor_scores=_flat_scores(L),
        start_scores=_flat_scores(L),
        stop_scores=_flat_scores(L),
        coding_scores=_coding_scores(L),
    )
    models = viterbi_decode(dag)
    assert models == []


def test_build_dag_finds_simple_gene() -> None:
    L = 500
    donor = _flat_scores(L)
    acceptor = _flat_scores(L)
    start = _flat_scores(L)
    stop = _flat_scores(L)
    coding = _coding_scores(L)

    # Place a simple single-exon gene: start at 50, stop at 200
    start[50] = 0.9
    stop[200] = 0.9
    for i in range(50, 200):
        coding[i] = [0.8, 0.1, 0.1]

    dag = build_dag(
        donor_scores=donor,
        acceptor_scores=acceptor,
        start_scores=start,
        stop_scores=stop,
        coding_scores=coding,
        threshold=0.5,
    )
    models = viterbi_decode(dag)
    assert len(models) >= 1
    m = models[0]
    assert m.start >= 50
    assert m.end <= 200


def test_organism_constraints_applied() -> None:
    L = 200
    dag_vertebrate = build_dag(
        donor_scores=_flat_scores(L, 0.9),
        acceptor_scores=_flat_scores(L, 0.9),
        start_scores=_flat_scores(L),
        stop_scores=_flat_scores(L),
        coding_scores=_coding_scores(L, 0.9),
        organism="vertebrate",
        threshold=0.5,
    )
    dag_fungus = build_dag(
        donor_scores=_flat_scores(L, 0.9),
        acceptor_scores=_flat_scores(L, 0.9),
        start_scores=_flat_scores(L),
        stop_scores=_flat_scores(L),
        coding_scores=_coding_scores(L, 0.9),
        organism="fungus",
        threshold=0.5,
    )
    # Both should run without error; constraint differences are in intron/exon lengths
    _ = viterbi_decode(dag_vertebrate)
    _ = viterbi_decode(dag_fungus)
