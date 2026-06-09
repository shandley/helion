#!/bin/bash
#SBATCH --job-name=helion-train-vertebrate
#SBATCH --partition=general-gpu
#SBATCH --gres=gpu:H100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=/storage3/fs1/shandley/Active/helion/logs/%j_train_vertebrate.out
#SBATCH --error=/storage3/fs1/shandley/Active/helion/logs/%j_train_vertebrate.err

set -euo pipefail

HELION_ROOT="/storage3/fs1/shandley/Active/helion"
DATA_DIR="${HELION_ROOT}/data/vertebrate"
MODEL_OUT="${HELION_ROOT}/models/vertebrate"
CONDA_SH="/storage3/fs1/shandley/Active/echobase/miniforge/etc/profile.d/conda.sh"

echo "Job ID:   ${SLURM_JOB_ID}"
echo "Node:     ${SLURMD_NODENAME}"
echo "Started:  $(date)"
echo ""

source "${CONDA_SH}"
conda activate echobase-ml

export PYTHONPATH="${HELION_ROOT}/python"

mkdir -p "${MODEL_OUT}" "${HELION_ROOT}/logs"

# Pull latest code in case fixes landed since data download
git -C "${HELION_ROOT}" pull --rebase -q

python "${HELION_ROOT}/scripts/train_helion.py" \
    --annotations "${DATA_DIR}/vertebrate.gff3" \
    --genome      "${DATA_DIR}/vertebrate.fa" \
    --output      "${MODEL_OUT}" \
    --organism    vertebrate \
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
