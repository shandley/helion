#!/bin/bash
#SBATCH --job-name=helion-train-plant-v3
#SBATCH --account=compute2-shandley
#SBATCH --partition=general-gpu
#SBATCH --gres=gpu:H100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/storage3/fs1/shandley/Active/helion/logs/%j_train_plant_v3.out
#SBATCH --error=/storage3/fs1/shandley/Active/helion/logs/%j_train_plant_v3.err

# v3 change: dataset normalizes all training examples to sense orientation
# (RC sequence + RC-mapped labels for minus-strand genes).
# Window size stays at 2kb -- Arabidopsis introns are short (p90 ~300nt).

set -euo pipefail

HELION_ROOT="/storage3/fs1/shandley/Active/helion"
DATA_DIR="${HELION_ROOT}/data/plant"
MODEL_OUT="${HELION_ROOT}/models/plant_v3"
CONDA_SH="/storage3/fs1/shandley/Active/echobase/miniforge/etc/profile.d/conda.sh"

echo "Job ID:   ${SLURM_JOB_ID}"
echo "Node:     ${SLURMD_NODENAME}"
echo "Started:  $(date)"
echo ""

source "${CONDA_SH}"
conda activate echobase-ml
export PYTHONPATH="${HELION_ROOT}/python"

source "$HOME/.cargo/env"
cd "${HELION_ROOT}"
git fetch -q origin && git reset -q --hard origin/main
maturin develop -q 2>&1 | tail -2

mkdir -p "${MODEL_OUT}" "${HELION_ROOT}/logs"

python "${HELION_ROOT}/scripts/train_helion.py" \
    --annotations "${DATA_DIR}/arabidopsis.gff3" \
    --genome      "${DATA_DIR}/arabidopsis.fa" \
    --output      "${MODEL_OUT}" \
    --organism    plant \
    --epochs      50 \
    --batch-size  128 \
    --lr          1e-4 \
    --window-size 2000 \
    --channels    256 \
    --workers     8 \
    --device      cuda

echo ""
echo "Finished: $(date)"
