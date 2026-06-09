from __future__ import annotations

import numpy as np
import numpy.typing as npt

from helion.signals.model import SignalScores

# fmt: off
# Canonical BLOSUM62 substitution matrix (NCBI).
# Rows and columns ordered by _AA_ORDER below.
_AA_ORDER = "ARNDCQEGHILKMFPSTWYV"

_BLOSUM62_RAW: list[list[int]] = [
 # A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V
 [ 4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0],  # A
 [-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3],  # R
 [-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3],  # N
 [-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3],  # D
 [ 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],  # C
 [-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2],  # Q
 [-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2],  # E
 [ 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3],  # G
 [-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3],  # H
 [-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3],  # I
 [-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1],  # L
 [-1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2],  # K
 [-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1],  # M
 [-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1],  # F
 [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2],  # P
 [ 1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2],  # S
 [ 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0],  # T
 [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3],  # W
 [-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1],  # Y
 [ 0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4],  # V
]
# fmt: on

_AA_INDEX: dict[str, int] = {aa: i for i, aa in enumerate(_AA_ORDER)}
_BLOSUM62_MATRIX = np.array(_BLOSUM62_RAW, dtype=np.float32)


# Genetic code (standard)
_CODON_TABLE: dict[str, str] = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}


def translate(dna: str) -> str:
    aa = []
    dna = dna.upper()
    for i in range(0, len(dna) - 2, 3):
        codon = dna[i:i + 3]
        aa.append(_CODON_TABLE.get(codon, "X"))
    return "".join(aa)


def _blosum62_vector(aa: str) -> npt.NDArray[np.float32]:
    idx = _AA_INDEX.get(aa, -1)
    if idx < 0:
        return np.zeros(20, dtype=np.float32)
    return _BLOSUM62_MATRIX[idx]


def _peptide_embedding(peptide: str) -> npt.NDArray[np.float32]:
    """Mean BLOSUM62 row vector over all residues in the peptide."""
    if not peptide:
        return np.zeros(20, dtype=np.float32)
    vecs = np.stack([_blosum62_vector(aa) for aa in peptide])
    return vecs.mean(axis=0)


def _ungapped_slide_score(peptide: str, protein: str) -> float:
    """Best ungapped alignment score of peptide slid along protein."""
    if not peptide or not protein:
        return 0.0
    pep_len = len(peptide)
    best = -1e9
    for start in range(max(1, len(protein) - pep_len + 1)):
        window = protein[start:start + pep_len]
        score = sum(
            int(_BLOSUM62_MATRIX[_AA_INDEX[p], _AA_INDEX[q]])
            if p in _AA_INDEX and q in _AA_INDEX else -4
            for p, q in zip(peptide, window)
        )
        if score > best:
            best = score
    return float(best)


def score_exon_homology_blosum(
    sequence: str,
    protein_sequence: str,
) -> npt.NDArray[np.float32]:
    """
    Per-nucleotide homology score using BLOSUM62 sliding-window alignment.
    Translates 20-aa windows in all 3 frames and scores each against the
    query protein via ungapped alignment. Returns scores in [0, 1].
    """
    L = len(sequence)
    homology = np.zeros(L, dtype=np.float32)

    # Self-alignment score of the protein (for normalization)
    self_score = sum(
        int(_BLOSUM62_MATRIX[_AA_INDEX[aa], _AA_INDEX[aa]])
        if aa in _AA_INDEX else 0
        for aa in protein_sequence
    )
    if self_score <= 0:
        return homology

    window_nt = 60  # 20 aa

    for frame in range(3):
        pos = frame
        while pos + window_nt <= L:
            peptide = translate(sequence[pos:pos + window_nt])
            if "*" not in peptide:
                raw = _ungapped_slide_score(peptide, protein_sequence)
                # Normalize by the peptide self-score (local alignment scale)
                pep_self = sum(
                    int(_BLOSUM62_MATRIX[_AA_INDEX[aa], _AA_INDEX[aa]])
                    if aa in _AA_INDEX else 0
                    for aa in peptide
                )
                norm = float(raw) / max(pep_self, 1)
                norm = float(np.clip(norm, 0.0, 1.0))
                homology[pos:pos + window_nt] = np.maximum(
                    homology[pos:pos + window_nt], norm
                )
            pos += 3

    return homology


def score_exon_homology(
    sequence: str,
    scores: SignalScores,
    protein_embedding: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """
    Per-nucleotide homology score using cosine similarity between BLOSUM62
    peptide embeddings and the protein embedding.

    For each 20-aa window in each of 3 frames, computes a mean BLOSUM62
    row vector for the peptide and compares it against a summary of the
    protein embedding via cosine similarity.

    Works with any protein embedding dimensionality (ESM-2 at 320/640/1280
    or a 20-dim BLOSUM62 embedding). The protein embedding is projected to
    20 dims by taking the first 20 principal columns (or all if dim < 20).
    """
    _ = scores  # reserved for future signal-gated scoring
    L = len(sequence)
    homology = np.zeros(L, dtype=np.float32)

    # Summarize protein embedding: mean over residues, project to 20 dims
    prot_mean = protein_embedding.mean(axis=0)  # (embed_dim,)
    dim = prot_mean.shape[0]
    if dim >= 20:
        prot_vec = prot_mean[:20].astype(np.float32)
    else:
        prot_vec = np.pad(prot_mean, (0, 20 - dim)).astype(np.float32)

    prot_norm = np.linalg.norm(prot_vec)
    if prot_norm < 1e-8:
        return homology
    prot_vec = prot_vec / prot_norm

    window_nt = 60  # 20 aa

    for frame in range(3):
        pos = frame
        while pos + window_nt <= L:
            peptide = translate(sequence[pos:pos + window_nt])
            if "*" not in peptide:
                pep_vec = _peptide_embedding(peptide)
                pep_norm = np.linalg.norm(pep_vec)
                if pep_norm > 1e-8:
                    sim = float(np.dot(pep_vec / pep_norm, prot_vec))
                    sim = float(np.clip(sim, 0.0, 1.0))
                    homology[pos:pos + window_nt] = np.maximum(
                        homology[pos:pos + window_nt], sim
                    )
            pos += 3

    return homology
