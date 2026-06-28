#!/bin/bash
#SBATCH --job-name=helion-precompute-feat
#SBATCH --account=compute2-shandley
#SBATCH --partition=general-gpu
#SBATCH --gres=gpu:H100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=/storage3/fs1/shandley/Active/helion/logs/%j_precompute_feat.out
#SBATCH --error=/storage3/fs1/shandley/Active/helion/logs/%j_precompute_feat.err

# Precompute HyenaDNA offset features for the fusion A/B train + val windows.
# One-time; the cached .npy files are reused by both control (ignored) and the
# 10-channel treatment.  Env: TRAIN_CHROMS, FEATURE_DIR.

set -euo pipefail

H="/storage3/fs1/shandley/Active/helion"
CONDA_SH="/storage3/fs1/shandley/Active/echobase/miniforge/etc/profile.d/conda.sh"
DATA_DIR="${H}/data/vertebrate"
TRAIN_CHROMS="${TRAIN_CHROMS:-19 20 21}"
FEATURE_DIR="${FEATURE_DIR:-${H}/features/vertebrate_fusion}"

export HF_HOME="${H}/hf_cache"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "Job ID: ${SLURM_JOB_ID}  Node: ${SLURMD_NODENAME}  Started: $(date)"
echo "Train chromosomes: ${TRAIN_CHROMS}"
echo "Feature dir: ${FEATURE_DIR}"

source "${CONDA_SH}"
conda activate echobase-ml
export PYTHONPATH="${H}/python"
cd "${H}"
flock -x -w 180 "${H}/.build.lock" bash -c '
    git fetch -q origin && git reset -q --hard origin/main
'

python scripts/precompute_window_features.py \
    --genome         "${DATA_DIR}/vertebrate.fa" \
    --annotations    "${DATA_DIR}/vertebrate.gff3" \
    --feature-dir    "${FEATURE_DIR}" \
    --train-chromosomes ${TRAIN_CHROMS} \
    --neg-fraction   0.5 \
    --device         cuda

echo "Finished: $(date)"
