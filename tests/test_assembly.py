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


def test_consensus_filter_rejects_non_codon_start() -> None:
    """With a sequence, a start candidate must carry an ATG (or acceptor AG);
    otherwise the single-exon gene cannot be assembled."""
    L = 500
    start = _flat_scores(L)
    stop = _flat_scores(L)
    coding = _coding_scores(L)
    start[50] = 0.9
    stop[200] = 0.9
    for i in range(50, 200):
        coding[i] = [0.8, 0.1, 0.1]

    # Sequence with a real start codon at 50 and a stop codon at [197, 200).
    seq = list("C" * L)
    seq[50:53] = list("ATG")
    seq[197:200] = list("TAA")

    kwargs = dict(
        donor_scores=_flat_scores(L),
        acceptor_scores=_flat_scores(L),
        start_scores=start,
        stop_scores=stop,
        coding_scores=coding,
        threshold=0.5,
    )

    dag = build_dag(sequence="".join(seq), **kwargs)
    assert len(viterbi_decode(dag)) >= 1

    # Break the start codon: no ATG (and no AG acceptor) at position 50 -> rejected.
    seq[50:53] = list("CCC")
    dag_bad = build_dag(sequence="".join(seq), **kwargs)
    assert viterbi_decode(dag_bad) == []


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
