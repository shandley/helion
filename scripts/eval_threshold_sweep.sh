#!/bin/bash
#SBATCH --job-name=helion-thresh-sweep
#SBATCH --account=compute2-shandley
#SBATCH --partition=general-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/storage3/fs1/shandley/Active/helion/logs/%j_thresh_sweep.out
#SBATCH --error=/storage3/fs1/shandley/Active/helion/logs/%j_thresh_sweep.err

# Required: --export=THRESHOLD=<value>
# Optional: --export=THRESHOLD=<value>,ORGANISM=insect,MODEL=drosophila_v2,CHROM=4

set -euo pipefail

H="/storage3/fs1/shandley/Active/helion"
CONDA_SH="/storage3/fs1/shandley/Active/echobase/miniforge/etc/profile.d/conda.sh"

ORGANISM="${ORGANISM:-insect}"
MODEL="${MODEL:-drosophila_v2}"
CHROM="${CHROM:-4}"

echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      ${SLURMD_NODENAME}"
echo "Threshold: ${THRESHOLD}"
echo "Model:     ${MODEL}  Organism: ${ORGANISM}  Chrom: ${CHROM}"
echo "Started:   $(date)"
echo ""

source "${CONDA_SH}"
conda activate echobase-ml
export PYTHONPATH="${H}/python"

source "$HOME/.cargo/env"
cd "${H}"
git fetch -q origin && git reset -q --hard origin/main
maturin develop 2>&1

FA="${H}/results/${MODEL%_v*}_chr${CHROM}.fa"
REF="${H}/results/ref_${MODEL%_v*}_chr${CHROM}.gff3"
PRED="${H}/results/pred_${MODEL}_chr${CHROM}_t${THRESHOLD}.gff3"

echo "=== helion predict (threshold=${THRESHOLD}) ==="
helion predict \
    "${FA}" \
    "${PRED}" \
    --model    "${H}/models/${MODEL}" \
    --organism "${ORGANISM}" \
    --threshold "${THRESHOLD}" \
    --device   cpu

echo "Predicted: $(grep -c CDS ${PRED} 2>/dev/null || echo 0) CDS lines"

echo ""
echo "=== helion evaluate ==="
helion evaluate "${REF}" "${PRED}" "${FA}"

echo ""
echo "Finished: $(date)"
