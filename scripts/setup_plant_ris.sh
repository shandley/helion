#!/bin/bash
# Download Arabidopsis thaliana TAIR10 genome + Ensembl Plants annotation.
# Runs on the RIS login node (download only, no compute).
# Chr4 is the held-out val chromosome (matches DEFAULT_VAL_CHROMS in train.py).

set -euo pipefail

HELION_ROOT="/storage3/fs1/shandley/Active/helion"
DATA_DIR="${HELION_ROOT}/data/plant"
ENSEMBL_PLANTS_RELEASE="60"
BASE="https://ftp.ebi.ac.uk/pub/databases/ensemblgenomes/pub/plants/release-${ENSEMBL_PLANTS_RELEASE}"

mkdir -p "${DATA_DIR}"

echo "=== Plant (Arabidopsis thaliana TAIR10) ==="

GENOME="${DATA_DIR}/arabidopsis.fa"
if [ ! -f "${GENOME}" ]; then
    echo "Downloading Arabidopsis genome..."
    wget -q --show-progress -O "${GENOME}.gz" \
        "${BASE}/fasta/arabidopsis_thaliana/dna/Arabidopsis_thaliana.TAIR10.dna.toplevel.fa.gz"
    gunzip "${GENOME}.gz"
    echo "  done: $(du -sh ${GENOME} | cut -f1)"
else
    echo "Genome already present."
fi

GFF="${DATA_DIR}/arabidopsis.gff3"
if [ ! -f "${GFF}" ]; then
    echo "Downloading Arabidopsis annotation..."
    wget -q --show-progress -O "${GFF}.gz" \
        "${BASE}/gff3/arabidopsis_thaliana/Arabidopsis_thaliana.TAIR10.${ENSEMBL_PLANTS_RELEASE}.gff3.gz"
    gunzip "${GFF}.gz"
    echo "  done: $(wc -l < ${GFF}) lines"
else
    echo "Annotation already present."
fi

# Index for pyfaidx
if [ ! -f "${GENOME}.fai" ]; then
    echo "Indexing genome..."
    source /storage3/fs1/shandley/Active/echobase/miniforge/etc/profile.d/conda.sh
    conda activate echobase-ml
    python -c "from pyfaidx import Fasta; Fasta('${GENOME}')"
    echo "  done"
fi

echo ""
echo "=== Plant setup complete ==="
echo "Genome:     ${GENOME}"
echo "Annotation: ${GFF}"
echo "Val chrom:  4 (held out)"
