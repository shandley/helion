#!/bin/bash
# Download Arabidopsis thaliana TAIR10 genome + annotation from NCBI.
# Post-processes RefSeq chromosome IDs to simple numbers (1-5).
# Chr4 is the held-out val chromosome (matches DEFAULT_VAL_CHROMS in train.py).

set -euo pipefail

HELION_ROOT="/storage3/fs1/shandley/Active/helion"
DATA_DIR="${HELION_ROOT}/data/plant"
NCBI_BASE="https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/735/GCF_000001735.4_TAIR10.1"

mkdir -p "${DATA_DIR}"

echo "=== Plant (Arabidopsis thaliana TAIR10 via NCBI) ==="

GENOME="${DATA_DIR}/arabidopsis.fa"
GFF="${DATA_DIR}/arabidopsis.gff3"

if [ ! -f "${GENOME}" ]; then
    echo "Downloading genome..."
    wget -q --show-progress -O "${DATA_DIR}/raw.fa.gz" \
        "${NCBI_BASE}/GCF_000001735.4_TAIR10.1_genomic.fna.gz"
    echo "Extracting nuclear chromosomes and renaming..."
    zcat "${DATA_DIR}/raw.fa.gz" | awk '
        /^>NC_003070/ {print ">1"; keep=1; next}
        /^>NC_003071/ {print ">2"; keep=1; next}
        /^>NC_003074/ {print ">3"; keep=1; next}
        /^>NC_003075/ {print ">4"; keep=1; next}
        /^>NC_003076/ {print ">5"; keep=1; next}
        /^>/          {keep=0; next}
        keep          {print}
    ' > "${GENOME}"
    rm "${DATA_DIR}/raw.fa.gz"
    echo "  done: $(du -sh ${GENOME} | cut -f1)"
else
    echo "Genome already present."
fi

if [ ! -f "${GFF}" ]; then
    echo "Downloading annotation..."
    wget -q --show-progress -O "${DATA_DIR}/raw.gff.gz" \
        "${NCBI_BASE}/GCF_000001735.4_TAIR10.1_genomic.gff.gz"
    echo "Filtering to nuclear chromosomes and renaming..."
    zcat "${DATA_DIR}/raw.gff.gz" \
        | awk '$1 ~ /^#/ || $1=="NC_003070.9" || $1=="NC_003071.7" || $1=="NC_003074.8" || $1=="NC_003075.7" || $1=="NC_003076.8"' \
        | sed 's/^NC_003070\.[0-9]*/1/; s/^NC_003071\.[0-9]*/2/; s/^NC_003074\.[0-9]*/3/; s/^NC_003075\.[0-9]*/4/; s/^NC_003076\.[0-9]*/5/' \
        > "${GFF}"
    rm "${DATA_DIR}/raw.gff.gz"
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
