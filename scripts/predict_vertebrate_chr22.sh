#!/bin/bash
#SBATCH --job-name=helion-predict-vert
#SBATCH --partition=general-gpu
#SBATCH --gres=gpu:H100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/storage3/fs1/shandley/Active/helion/logs/%j_predict_vertebrate.out
#SBATCH --error=/storage3/fs1/shandley/Active/helion/logs/%j_predict_vertebrate.err

set -euo pipefail

HELION_ROOT="/storage3/fs1/shandley/Active/helion"
CONDA_SH="/storage3/fs1/shandley/Active/echobase/miniforge/etc/profile.d/conda.sh"

echo "Job ID:   ${SLURM_JOB_ID}"
echo "Node:     ${SLURMD_NODENAME}"
echo "Started:  $(date)"

source "${CONDA_SH}"
conda activate echobase-ml

source "$HOME/.cargo/env"
cd "${HELION_ROOT}"
maturin develop -q 2>&1 | tail -2

helion predict \
    "${HELION_ROOT}/results/vertebrate_chr22.fa" \
    "${HELION_ROOT}/results/pred_vertebrate_chr22.gff3" \
    --model   "${HELION_ROOT}/models/vertebrate" \
    --organism vertebrate \
    --device  cuda

echo "Finished: $(date)"
echo "Lines: $(wc -l < ${HELION_ROOT}/results/pred_vertebrate_chr22.gff3)"
