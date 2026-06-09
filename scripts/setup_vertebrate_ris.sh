#!/bin/bash
# Download human chr1, chr19, chr22 + Ensembl annotation for vertebrate training.
# Runs on the RIS login node (download only, no compute).
# chr22 is the held-out val chromosome (matches DEFAULT_VAL_CHROMS in train.py).

set -euo pipefail

HELION_ROOT="/storage3/fs1/shandley/Active/helion"
DATA_DIR="${HELION_ROOT}/data/vertebrate"
ENSEMBL_RELEASE="113"
BASE="https://ftp.ensembl.org/pub/release-${ENSEMBL_RELEASE}"

mkdir -p "${DATA_DIR}"

echo "=== Vertebrate (human chr1, chr19, chr22) ==="

# Individual chromosome FASTAs
for CHR in 1 19 22; do
    FA="${DATA_DIR}/chr${CHR}.fa.gz"
    if [ ! -f "${DATA_DIR}/chr${CHR}.fa" ]; then
        echo "Downloading chr${CHR}..."
        wget -q --show-progress -O "${FA}" \
            "${BASE}/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.chromosome.${CHR}.fa.gz"
        gunzip "${FA}"
        echo "  done: $(du -sh ${DATA_DIR}/chr${CHR}.fa | cut -f1)"
    else
        echo "chr${CHR} already present."
    fi
done

# Combine into one FASTA for pyfaidx
GENOME="${DATA_DIR}/vertebrate.fa"
if [ ! -f "${GENOME}" ]; then
    echo "Combining chromosomes..."
    cat "${DATA_DIR}/chr1.fa" "${DATA_DIR}/chr19.fa" "${DATA_DIR}/chr22.fa" > "${GENOME}"
    rm "${DATA_DIR}/chr1.fa" "${DATA_DIR}/chr19.fa" "${DATA_DIR}/chr22.fa"
    echo "  done: $(du -sh ${GENOME} | cut -f1)"
fi

# Annotation: download full GFF3 then filter to our three chromosomes
GFF="${DATA_DIR}/vertebrate.gff3"
if [ ! -f "${GFF}" ]; then
    echo "Downloading annotation (full GFF3, will filter)..."
    ANN_GZ="${DATA_DIR}/annotation.gff3.gz"
    wget -q --show-progress -O "${ANN_GZ}" \
        "${BASE}/gff3/homo_sapiens/Homo_sapiens.GRCh38.${ENSEMBL_RELEASE}.gff3.gz"
    echo "Filtering to chr1, chr19, chr22..."
    zcat "${ANN_GZ}" | awk '$1 ~ /^#/ || $1=="1" || $1=="19" || $1=="22"' > "${GFF}"
    rm "${ANN_GZ}"
    echo "  done: $(wc -l < ${GFF}) lines"
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
echo "=== Vertebrate setup complete ==="
echo "Genome:     ${GENOME}"
echo "Annotation: ${GFF}"
echo "Val chrom:  22 (held out)"
