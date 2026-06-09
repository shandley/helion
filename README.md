# Helion

Helion predicts protein-coding genes in eukaryotic genomes. It combines a deep learning signal model with optional protein homology guidance to identify splice sites, exon boundaries, and reading frames, then assembles those signals into complete gene structures using dynamic programming.

If you have a protein sequence that is related to a gene you expect to find, Helion can use that to improve prediction accuracy. Without a protein, it runs in ab initio mode.

## Requirements

- Python 3.11+
- Rust (for building from source)
- PyTorch 2.2+

## Installation

```bash
git clone https://github.com/shandley/helion
cd helion
uv venv
uv pip install maturin
maturin develop
uv pip install -e ".[dev]"
```

For protein homology support (ESM-2 embeddings), install the optional dependencies:

```bash
uv pip install -e ".[homology]"
```

## Usage

### Predict genes

```bash
helion predict genome.fa output.gff3 --model models/vertebrate
```

With a homologous protein:

```bash
helion predict genome.fa output.gff3 --model models/vertebrate --protein myprotein.fa
```

Options:

```
--model      Path to trained model weights directory (required)
--protein    Protein sequence to guide prediction (optional)
--organism   One of: vertebrate, insect, plant, fungus (default: vertebrate)
--device     Compute device: cpu, cuda, or mps (default: cpu)
--threshold  Signal score cutoff for candidate exons (default: 0.1)
```

The genome should be repeat-masked before running. Hard masking (replacing repeats with N) is recommended.

### Train a model

```bash
helion train annotations.gff3 genome.fa models/myorganism --organism vertebrate --epochs 50
```

Training data should be a GFF3 annotation file and a matching genome FASTA. GENCODE and Ensembl annotations work directly.

## Output

Helion writes a GFF3 file with gene, mRNA, and CDS features. Coordinates are 1-based inclusive, matching the GFF3 standard.

## How it works

Prediction runs in three stages:

1. A dilated residual CNN scores each nucleotide position for splice donor sites, splice acceptors, start codons, stop codons, and coding potential in all three frames.

2. Those scores are used to build a graph of candidate exons. A dynamic programming pass finds the highest-scoring chains of exons connected by valid introns.

3. If a homologous protein is provided, translated candidate exons are aligned against it using BLOSUM62 similarity. High-scoring alignments add weight to the corresponding exon candidates before the DP step.

## Project structure

```
python/helion/     Python package
  signals/         CNN signal model and training
  homology/        Protein alignment and ESM-2 embedding
  io/              FASTA reader and GFF3 writer
src/               Rust extension (DAG construction, Viterbi decoding)
tests/             Test suite
models/            Trained model weights (not included in repo)
data/              Training data (not included in repo)
```

## License

MIT
