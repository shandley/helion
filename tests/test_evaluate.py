"""
Tests for the evaluation framework.

The critical invariant: evaluate(reference, reference) == perfect scores.
If that fails the framework is wrong, not the model.
"""

import tempfile
from pathlib import Path

import pytest

from helion.evaluate import (
    EvaluationResult,
    ExonMetrics,
    NucleotideMetrics,
    _exon_metrics,
    _nucleotide_metrics,
    load_cds_intervals,
    format_report,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_gff3(path: Path, records: list[tuple[str, int, int]]) -> None:
    """Write minimal GFF3 with CDS records. coords are 0-based half-open."""
    with path.open("w") as f:
        f.write("##gff-version 3\n")
        for i, (seqid, start, end) in enumerate(records, 1):
            # GFF3 is 1-based inclusive
            f.write(f"{seqid}\ttest\tCDS\t{start + 1}\t{end}\t.\t+\t0\tID=cds{i}\n")


def _write_fai(path: Path, lengths: dict[str, int]) -> None:
    """Write a minimal .fai index."""
    with path.open("w") as f:
        for seqid, length in lengths.items():
            f.write(f"{seqid}\t{length}\t0\t60\t61\n")


# ---------------------------------------------------------------------------
# NucleotideMetrics properties
# ---------------------------------------------------------------------------

def test_nucleotide_sensitivity() -> None:
    m = NucleotideMetrics(tp=80, fp=20, tn=800, fn=20)
    assert abs(m.sensitivity - 80 / 100) < 1e-9


def test_nucleotide_specificity() -> None:
    m = NucleotideMetrics(tp=80, fp=20, tn=800, fn=20)
    assert abs(m.specificity - 80 / 100) < 1e-9


def test_nucleotide_cc_perfect() -> None:
    m = NucleotideMetrics(tp=100, fp=0, tn=900, fn=0)
    assert abs(m.correlation - 1.0) < 1e-9


def test_nucleotide_cc_zero() -> None:
    # FP*FN == TP*TN => CC == 0
    m = NucleotideMetrics(tp=10, fp=10, tn=10, fn=10)
    assert abs(m.correlation) < 1e-9


def test_nucleotide_zero_denom() -> None:
    m = NucleotideMetrics(tp=0, fp=0, tn=0, fn=0)
    assert m.sensitivity == 0.0
    assert m.specificity == 0.0
    assert m.correlation == 0.0


# ---------------------------------------------------------------------------
# ExonMetrics properties
# ---------------------------------------------------------------------------

def test_exon_sensitivity() -> None:
    m = ExonMetrics(tp=7, fp=3, fn=3)
    assert abs(m.sensitivity - 7 / 10) < 1e-9


def test_exon_specificity() -> None:
    m = ExonMetrics(tp=7, fp=3, fn=3)
    assert abs(m.specificity - 7 / 10) < 1e-9


# ---------------------------------------------------------------------------
# load_cds_intervals
# ---------------------------------------------------------------------------

def test_load_cds_intervals_basic() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        gff = Path(tmp) / "test.gff3"
        _write_gff3(gff, [("chr1", 100, 200), ("chr1", 300, 400), ("chr2", 50, 150)])
        intervals = load_cds_intervals(gff)
        assert sorted(intervals["chr1"]) == [(100, 200), (300, 400)]
        assert intervals["chr2"] == [(50, 150)]


def test_load_cds_intervals_ignores_other_features() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        gff = Path(tmp) / "test.gff3"
        with gff.open("w") as f:
            f.write("##gff-version 3\n")
            f.write("chr1\ttest\tgene\t101\t500\t.\t+\t.\tID=g1\n")
            f.write("chr1\ttest\tmRNA\t101\t500\t.\t+\t.\tID=t1;Parent=g1\n")
            f.write("chr1\ttest\tCDS\t101\t200\t.\t+\t0\tID=c1;Parent=t1\n")
        intervals = load_cds_intervals(gff)
        assert "chr1" in intervals
        assert (100, 200) in intervals["chr1"]
        assert len(intervals["chr1"]) == 1


# ---------------------------------------------------------------------------
# _nucleotide_metrics
# ---------------------------------------------------------------------------

def test_nucleotide_self_eval_perfect() -> None:
    """evaluate(reference, reference) must give perfect scores."""
    ref = {"chr1": [(100, 200), (300, 400)]}
    lengths = {"chr1": 1000}
    m = _nucleotide_metrics(ref, ref, lengths)
    assert m.tp == 200  # 100 + 100 coding positions
    assert m.fp == 0
    assert m.fn == 0
    assert abs(m.sensitivity - 1.0) < 1e-9
    assert abs(m.specificity - 1.0) < 1e-9
    assert abs(m.correlation - 1.0) < 1e-9


def test_nucleotide_empty_prediction() -> None:
    ref = {"chr1": [(100, 200)]}
    lengths = {"chr1": 1000}
    m = _nucleotide_metrics(ref, {}, lengths)
    assert m.tp == 0
    assert m.fn == 100
    assert m.sensitivity == 0.0


def test_nucleotide_perfect_miss() -> None:
    ref = {"chr1": [(100, 200)]}
    pred = {"chr1": [(300, 400)]}
    lengths = {"chr1": 1000}
    m = _nucleotide_metrics(ref, pred, lengths)
    assert m.tp == 0
    assert m.fn == 100
    assert m.fp == 100


def test_nucleotide_partial_overlap() -> None:
    ref = {"chr1": [(100, 200)]}
    pred = {"chr1": [(150, 250)]}
    lengths = {"chr1": 1000}
    m = _nucleotide_metrics(ref, pred, lengths)
    assert m.tp == 50   # 150-200
    assert m.fn == 50   # 100-150
    assert m.fp == 50   # 200-250


def test_nucleotide_or_semantics_overlapping_transcripts() -> None:
    """Overlapping CDS intervals: position is coding if covered by any."""
    ref = {"chr1": [(100, 200), (150, 300)]}  # 100-300 all coding
    lengths = {"chr1": 500}
    pred = {"chr1": [(100, 300)]}
    m = _nucleotide_metrics(ref, pred, lengths)
    assert m.tp == 200  # 100 to 300
    assert m.fp == 0
    assert m.fn == 0


def test_nucleotide_multiple_chroms() -> None:
    ref = {"chr1": [(0, 100)], "chr2": [(0, 50)]}
    lengths = {"chr1": 200, "chr2": 100}
    m = _nucleotide_metrics(ref, ref, lengths)
    assert m.tp == 150
    assert m.fp == 0
    assert m.fn == 0


# ---------------------------------------------------------------------------
# _exon_metrics
# ---------------------------------------------------------------------------

def test_exon_self_eval_perfect() -> None:
    """evaluate(reference, reference) must give perfect exon scores."""
    ref = {"chr1": [(100, 200), (300, 400)]}
    m = _exon_metrics(ref, ref)
    assert m.tp == 2
    assert m.fp == 0
    assert m.fn == 0
    assert abs(m.sensitivity - 1.0) < 1e-9
    assert abs(m.specificity - 1.0) < 1e-9


def test_exon_exact_match_required() -> None:
    """Off-by-one in boundary = no match."""
    ref = {"chr1": [(100, 200)]}
    pred = {"chr1": [(100, 201)]}  # end off by 1
    m = _exon_metrics(ref, pred)
    assert m.tp == 0
    assert m.fn == 1
    assert m.fp == 1


def test_exon_partial_prediction() -> None:
    ref = {"chr1": [(100, 200), (300, 400), (500, 600)]}
    pred = {"chr1": [(100, 200), (300, 400)]}  # missing last exon
    m = _exon_metrics(ref, pred)
    assert m.tp == 2
    assert m.fn == 1
    assert m.fp == 0
    assert abs(m.sensitivity - 2 / 3) < 1e-9
    assert abs(m.specificity - 1.0) < 1e-9


def test_exon_cross_chrom() -> None:
    ref = {"chr1": [(0, 100)], "chr2": [(0, 100)]}
    pred = {"chr1": [(0, 100)], "chr2": [(0, 99)]}  # chr2 off by 1
    m = _exon_metrics(ref, pred)
    assert m.tp == 1
    assert m.fn == 1
    assert m.fp == 1


# ---------------------------------------------------------------------------
# format_report
# ---------------------------------------------------------------------------

def test_format_report_smoke() -> None:
    result = EvaluationResult(
        nucleotide=NucleotideMetrics(tp=500, fp=50, tn=9000, fn=50),
        exon=ExonMetrics(tp=8, fp=2, fn=2),
    )
    report = format_report(result)
    assert "Sensitivity" in report
    assert "Correlation" in report
    assert "Missing" in report
    assert "Wrong" in report
