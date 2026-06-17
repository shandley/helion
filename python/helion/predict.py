from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

from helion.io.fasta import read_windows
from helion.io.gff3 import write_gff3
from helion.signals.model import SignalModel
from helion.homology.embed import ProteinEmbedder
from helion.homology.align import score_exon_homology

if TYPE_CHECKING:
    from helion._core import PyGeneModel

_RC_TABLE = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def _reverse_complement(seq: str) -> str:
    return seq.translate(_RC_TABLE)[::-1]


def _mean_exon_score(model: PyGeneModel) -> float:
    """Mean per-exon score for gene-level quality filtering.

    Each exon is scored as donor + acceptor + mean_coding (all softmax probs in
    [0, 1]), so the mean ranges from ~0.2 (noise) to ~1.8 (strong signal).
    Filtering on this rather than raw model.score avoids bias toward longer genes.
    """
    exons = model.exons
    return model.score / len(exons) if exons else 0.0


def predict(
    genome: Path,
    output: Path,
    model_weights: Path,
    protein: Optional[Path] = None,
    organism: str = "vertebrate",
    device: str = "cpu",
    batch_size: int = 4,
    threshold: float = 0.1,
    min_gene_score: float = 0.0,
) -> list[PyGeneModel]:
    from helion._core import PyGeneModel as _PyGeneModel
    from helion._core import build_dag, viterbi_decode

    _ = batch_size  # reserved for future batched window inference
    signal_model = SignalModel.load(model_weights, organism=organism, device=device)

    embedder = None
    protein_embedding = None
    if protein is not None:
        embedder = ProteinEmbedder(device=device)
        protein_embedding = embedder.embed(protein.read_text().strip())

    all_models: list[_PyGeneModel] = []

    for window in read_windows(genome, window_size=10_000, overlap=500):
        win_len = len(window.sequence)

        for seq, strand in [(window.sequence, "+"), (_reverse_complement(window.sequence), "-")]:
            scores = signal_model.score(seq)

            homology_scores = None
            if protein_embedding is not None:
                homology_scores = score_exon_homology(seq, scores, protein_embedding)

            dag = build_dag(
                donor_scores=scores.donor.tolist(),
                acceptor_scores=scores.acceptor.tolist(),
                start_scores=scores.start.tolist(),
                stop_scores=scores.stop.tolist(),
                coding_scores=scores.coding.tolist(),
                intergenic_scores=scores.intergenic.tolist(),
                homology_scores=homology_scores.tolist() if homology_scores is not None else None,
                organism=organism,
                threshold=threshold,
            )

            models = viterbi_decode(dag, strand)
            if strand == "-":
                models = [m.with_rc_flip(win_len) for m in models]
            all_models.extend(
                m.with_seqid(window.seqid).with_offset(window.offset) for m in models
            )

    if min_gene_score > 0.0:
        all_models = [m for m in all_models if _mean_exon_score(m) >= min_gene_score]

    write_gff3(all_models, output)
    return all_models
