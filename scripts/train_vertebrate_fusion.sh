#!/bin/bash
#SBATCH --job-name=helion-train-fusion
#SBATCH --account=compute2-shandley
#SBATCH --partition=general-gpu
#SBATCH --gres=gpu:H100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=/storage3/fs1/shandley/Active/helion/logs/%j_train_fusion.out
#SBATCH --error=/storage3/fs1/shandley/Active/helion/logs/%j_train_fusion.err

# DNA-embedding fusion A/B (Phase 1). Trains on a chromosome subset.
#   Control (4ch):    sbatch scripts/train_vertebrate_fusion.sh
#   Treatment (10ch): FEATURE_DIR=.../features/vertebrate_fusion OUT_SUFFIX="" \
#                     sbatch scripts/train_vertebrate_fusion.sh   (with FEATURE_DIR set)
# Same windows, same hyperparams, plain CE -- the only difference is the 6 fused
# HyenaDNA channels. Training reads precomputed features (no HyenaDNA needed here).

set -euo pipefail

H="/storage3/fs1/shandley/Active/helion"
CONDA_SH="/storage3/fs1/shandley/Active/echobase/miniforge/etc/profile.d/conda.sh"
DATA_DIR="${H}/data/vertebrate"
TRAIN_CHROMS="${TRAIN_CHROMS:-19 20 21}"
EPOCHS="${EPOCHS:-15}"
MODEL_OUT="${H}/models/vertebrate_fusion${OUT_SUFFIX:-_ctrl}"

EXTRA=""
[ -n "${FEATURE_DIR:-}" ] && EXTRA="--feature-dir ${FEATURE_DIR}"

echo "Job ID: ${SLURM_JOB_ID}  Node: ${SLURMD_NODENAME}  Started: $(date)"
echo "Train chromosomes: ${TRAIN_CHROMS}   Epochs: ${EPOCHS}"
echo "Feature dir: ${FEATURE_DIR:-<none, 4-channel control>}"
echo "Output: ${MODEL_OUT}"

source "${CONDA_SH}"
conda activate echobase-ml
export PYTHONPATH="${H}/python"
cd "${H}"
flock -x -w 300 "${H}/.build.lock" bash -c "
    git fetch -q origin && git reset -q --hard origin/main
    maturin develop -q 2>&1 | tail -2
"

mkdir -p "${MODEL_OUT}"

python scripts/train_helion.py \
    --annotations "${DATA_DIR}/vertebrate.gff3" \
    --genome      "${DATA_DIR}/vertebrate.fa" \
    --output      "${MODEL_OUT}" \
    --organism    vertebrate \
    --epochs      "${EPOCHS}" \
    --batch-size  64 \
    --lr          1e-4 \
    --window-size 5000 \
    --channels    256 \
    --workers     8 \
    --neg-fraction 0.5 \
    --train-chromosomes ${TRAIN_CHROMS} \
    ${EXTRA} \
    --device      cuda

echo "Finished: $(date)"
