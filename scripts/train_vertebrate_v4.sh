#!/bin/bash
#SBATCH --job-name=helion-train-vert-v4
#SBATCH --account=compute2-shandley
#SBATCH --partition=general-gpu
#SBATCH --gres=gpu:H100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --output=/storage3/fs1/shandley/Active/helion/logs/%j_train_vert_v4.out
#SBATCH --error=/storage3/fs1/shandley/Active/helion/logs/%j_train_vert_v4.err

# v4 changes vs v3:
# Hard negative sampling (--neg-fraction 0.5): adds intergenic windows sampled from
# genome regions with no annotated gene within 2kb on either side. These windows are
# labeled all-class-7 (intergenic), teaching the CNN to output high intergenic
# confidence in non-genic sequence. This should cut false positives in the Viterbi
# decoder by reducing intragenic node scores for truly intergenic regions.
# N-rich windows (centromeric/telomeric gaps) are excluded.

set -euo pipefail

HELION_ROOT="/storage3/fs1/shandley/Active/helion"
DATA_DIR="${HELION_ROOT}/data/vertebrate"
MODEL_OUT="${HELION_ROOT}/models/vertebrate_v4"
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
flock "${HELION_ROOT}/.git.lock" bash -c "
    git fetch -q origin
    git reset -q --hard origin/main
    maturin develop -q 2>&1 | tail -2
"

mkdir -p "${MODEL_OUT}" "${HELION_ROOT}/logs"

python "${HELION_ROOT}/scripts/train_helion.py" \
    --annotations "${DATA_DIR}/vertebrate.gff3" \
    --genome      "${DATA_DIR}/vertebrate.fa" \
    --output      "${MODEL_OUT}" \
    --organism    vertebrate \
    --epochs      50 \
    --batch-size  64 \
    --lr          1e-4 \
    --window-size 5000 \
    --channels    256 \
    --workers     8 \
    --neg-fraction 0.5 \
    --device      cuda

echo ""
echo "Finished: $(date)"
