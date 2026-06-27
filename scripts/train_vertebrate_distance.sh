#!/bin/bash
#SBATCH --job-name=helion-train-vert-dist
#SBATCH --account=compute2-shandley
#SBATCH --partition=general-gpu
#SBATCH --gres=gpu:H100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/storage3/fs1/shandley/Active/helion/logs/%j_train_vert_dist.out
#SBATCH --error=/storage3/fs1/shandley/Active/helion/logs/%j_train_vert_dist.err

# Phase A proof: train a vertebrate model with the signed-distance regression
# head (--distance-head) alongside the usual classification loss. Plain CE
# (no boundary emphasis -- that approach failed); v4 hard negatives kept.
# 15-epoch proxy for the boundary-localization A/B, then distance_head_proof.py
# checks whether the distance field's zero-crossings localise splice sites more
# sharply than the classification peaks. No Rust touched.

set -euo pipefail

HELION_ROOT="/storage3/fs1/shandley/Active/helion"
DATA_DIR="${HELION_ROOT}/data/vertebrate"
CONDA_SH="/storage3/fs1/shandley/Active/echobase/miniforge/etc/profile.d/conda.sh"

EPOCHS="${EPOCHS:-15}"
DIST_WEIGHT="${DIST_WEIGHT:-1.0}"
MODEL_OUT="${HELION_ROOT}/models/vertebrate_distance${OUT_SUFFIX:-}"

echo "Job ID:      ${SLURM_JOB_ID}"
echo "Node:        ${SLURMD_NODENAME}"
echo "Started:     $(date)"
echo "EPOCHS:      ${EPOCHS}"
echo "DIST_WEIGHT: ${DIST_WEIGHT}"
echo "OUTPUT:      ${MODEL_OUT}"
echo ""

source "${CONDA_SH}"
conda activate echobase-ml
export PYTHONPATH="${HELION_ROOT}/python"

source "$HOME/.cargo/env"
cd "${HELION_ROOT}"
flock -x -w 300 "${HELION_ROOT}/.build.lock" bash -c "
    git fetch -q origin
    git reset -q --hard origin/main
    maturin develop -q 2>&1 | tail -2
"

mkdir -p "${MODEL_OUT}" "${HELION_ROOT}/logs"

python "${HELION_ROOT}/scripts/train_helion.py" \
    --annotations    "${DATA_DIR}/vertebrate.gff3" \
    --genome         "${DATA_DIR}/vertebrate.fa" \
    --output         "${MODEL_OUT}" \
    --organism       vertebrate \
    --epochs         "${EPOCHS}" \
    --batch-size     64 \
    --lr             1e-4 \
    --window-size    5000 \
    --channels       256 \
    --workers        8 \
    --neg-fraction   0.5 \
    --distance-head \
    --distance-weight "${DIST_WEIGHT}" \
    --device         cuda

echo ""
echo "Finished: $(date)"
