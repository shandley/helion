#!/bin/bash
# One-time setup on RIS for Helion training.
# Run interactively: bash setup_ris.sh
# Expects an active RIS session (VPN + DUO already done).

set -euo pipefail

HELION_ROOT="/storage3/fs1/shandley/Active/helion"
DATA_DIR="${HELION_ROOT}/data/drosophila"
MODEL_DIR="${HELION_ROOT}/models"
LOG_DIR="${HELION_ROOT}/logs"

CONDA_SH="/storage3/fs1/shandley/Active/echobase/miniforge/etc/profile.d/conda.sh"
ENSEMBL_RELEASE="113"
BDGP="BDGP6.46"

echo "=== Setting up Helion on RIS ==="

# Directories
mkdir -p "${DATA_DIR}" "${MODEL_DIR}" "${LOG_DIR}"

# Clone or update repo
if [ -d "${HELION_ROOT}/.git" ]; then
    echo "Updating existing clone..."
    git -C "${HELION_ROOT}" pull --rebase
else
    echo "Cloning helion..."
    git clone https://github.com/shandley/helion.git "${HELION_ROOT}"
fi

# Add pyfaidx to echobase-ml (idempotent)
echo "Installing pyfaidx into echobase-ml..."
source "${CONDA_SH}"
conda activate echobase-ml
pip install --quiet pyfaidx

# Download Drosophila genome from Ensembl
GENOME_GZ="${DATA_DIR}/drosophila.fa.gz"
GENOME_FA="${DATA_DIR}/drosophila.fa"
GFF_GZ="${DATA_DIR}/drosophila.gff3.gz"
GFF="${DATA_DIR}/drosophila.gff3"

BASE="https://ftp.ensembl.org/pub/release-${ENSEMBL_RELEASE}"

if [ ! -f "${GENOME_FA}" ]; then
    echo "Downloading Drosophila genome..."
    wget -q -O "${GENOME_GZ}" \
        "${BASE}/fasta/drosophila_melanogaster/dna/Drosophila_melanogaster.${BDGP}.dna.toplevel.fa.gz"
    gunzip "${GENOME_GZ}"
    echo "  done: ${GENOME_FA}"
else
    echo "Genome already present."
fi

if [ ! -f "${GFF}" ]; then
    echo "Downloading Drosophila annotation..."
    wget -q -O "${GFF_GZ}" \
        "${BASE}/gff3/drosophila_melanogaster/Drosophila_melanogaster.${BDGP}.${ENSEMBL_RELEASE}.gff3.gz"
    gunzip "${GFF_GZ}"
    echo "  done: ${GFF}"
else
    echo "Annotation already present."
fi

# Index the genome for pyfaidx
if [ ! -f "${GENOME_FA}.fai" ]; then
    echo "Indexing genome..."
    python -c "from pyfaidx import Fasta; Fasta('${GENOME_FA}')"
    echo "  done"
fi

echo ""
echo "=== Setup complete ==="
echo "Genome:     ${GENOME_FA}"
echo "Annotation: ${GFF}"
echo "Models out: ${MODEL_DIR}/drosophila/"
echo ""
echo "Next: sbatch ${HELION_ROOT}/scripts/train_drosophila.sh"
