#!/bin/bash
#SBATCH --job-name=helion-train-drosophila
#SBATCH --partition=general-gpu
#SBATCH --gres=gpu:H100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/storage3/fs1/shandley/Active/helion/logs/%j_train_drosophila.out
#SBATCH --error=/storage3/fs1/shandley/Active/helion/logs/%j_train_drosophila.err

set -euo pipefail

HELION_ROOT="/storage3/fs1/shandley/Active/helion"
DATA_DIR="${HELION_ROOT}/data/drosophila"
MODEL_OUT="${HELION_ROOT}/models/drosophila"
CONDA_SH="/storage3/fs1/shandley/Active/echobase/miniforge/etc/profile.d/conda.sh"

echo "Job ID:   ${SLURM_JOB_ID}"
echo "Node:     ${SLURMD_NODENAME}"
echo "Started:  $(date)"
echo ""

source "${CONDA_SH}"
conda activate echobase-ml

# Helion lives in the repo checkout; no Rust build needed for training
export PYTHONPATH="${HELION_ROOT}/python"

mkdir -p "${MODEL_OUT}" "${HELION_ROOT}/logs"

python "${HELION_ROOT}/scripts/train_helion.py" \
    --annotations "${DATA_DIR}/drosophila.gff3" \
    --genome      "${DATA_DIR}/drosophila.fa" \
    --output      "${MODEL_OUT}" \
    --organism    insect \
    --epochs      50 \
    --batch-size  128 \
    --lr          1e-4 \
    --val-fraction 0.1 \
    --window-size 5000 \
    --channels    256 \
    --workers     8 \
    --device      cuda

echo ""
echo "Finished: $(date)"
