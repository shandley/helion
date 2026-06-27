#!/bin/bash
#SBATCH --job-name=helion-train-vert-v5
#SBATCH --account=compute2-shandley
#SBATCH --partition=general-gpu
#SBATCH --gres=gpu:H100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --output=/storage3/fs1/shandley/Active/helion/logs/%j_train_vert_v5.out
#SBATCH --error=/storage3/fs1/shandley/Active/helion/logs/%j_train_vert_v5.err

# v5 changes vs v4:
# Boundary-emphasis weighted loss (--boundary-emphasis / --boundary-radius):
# upweights per-position cross-entropy on positions within +/-RADIUS nt of every
# boundary label (donor/acceptor/start/stop). This forces the CNN to localize
# splice/start/stop boundaries sharply and cut the coding signal off crisply,
# rather than letting it bleed past the true exon end. v4's hard negatives
# (--neg-fraction 0.5) are KEPT.
#
# A/B is env-overridable so a paired control vs emphasis run can be launched:
#   Emphasis (real):  sbatch scripts/train_vertebrate_v5.sh
#   Control (off):    EMPHASIS=0 OUT_SUFFIX=_ctrl sbatch scripts/train_vertebrate_v5.sh

set -euo pipefail

HELION_ROOT="/storage3/fs1/shandley/Active/helion"
DATA_DIR="${HELION_ROOT}/data/vertebrate"

EMPHASIS="${EMPHASIS:-5}"
RADIUS="${RADIUS:-3}"
EPOCHS="${EPOCHS:-50}"
MODEL_OUT="${HELION_ROOT}/models/vertebrate_v5${OUT_SUFFIX:-}"
CONDA_SH="/storage3/fs1/shandley/Active/echobase/miniforge/etc/profile.d/conda.sh"

echo "Job ID:   ${SLURM_JOB_ID}"
echo "Node:     ${SLURMD_NODENAME}"
echo "Started:  $(date)"
echo "EMPHASIS: ${EMPHASIS}"
echo "RADIUS:   ${RADIUS}"
echo "EPOCHS:   ${EPOCHS}"
echo "OUTPUT:   ${MODEL_OUT}"
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
    --boundary-emphasis "${EMPHASIS}" \
    --boundary-radius   "${RADIUS}" \
    --device      cuda

echo ""
echo "Finished: $(date)"
