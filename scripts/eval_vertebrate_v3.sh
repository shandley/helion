#!/bin/bash
#SBATCH --job-name=helion-eval-vert-v3
#SBATCH --account=compute2-shandley
#SBATCH --partition=general-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=/storage3/fs1/shandley/Active/helion/logs/%j_eval_vert_v3.out
#SBATCH --error=/storage3/fs1/shandley/Active/helion/logs/%j_eval_vert_v3.err

# Evaluates vertebrate_v3 (sense-oriented training + RC inference) on chr22.
# Uses strand-aware evaluation since predictions now include both strands.

set -euo pipefail

H="/storage3/fs1/shandley/Active/helion"
CONDA_SH="/storage3/fs1/shandley/Active/echobase/miniforge/etc/profile.d/conda.sh"
THRESHOLD="${THRESHOLD:-0.3}"

echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      ${SLURMD_NODENAME}"
echo "Threshold: ${THRESHOLD}"
echo "Started:   $(date)"
echo ""

source "${CONDA_SH}"
conda activate echobase-ml
export PYTHONPATH="${H}/python"

source "$HOME/.cargo/env"
cd "${H}"
git fetch -q origin && git reset -q --hard origin/main
flock -x -w 120 "${H}/.maturin_build.lock" maturin develop 2>&1

mkdir -p "${H}/results"

FA="${H}/results/vertebrate_chr22.fa"
REF="${H}/results/ref_vertebrate_chr22.gff3"
PRED="${H}/results/pred_vertebrate_v3_chr22_t${THRESHOLD}.gff3"

echo "=== helion predict (vertebrate_v3, chr22, t=${THRESHOLD}) ==="
helion predict \
    "${FA}" \
    "${PRED}" \
    --model    "${H}/models/vertebrate_v3" \
    --organism vertebrate \
    --threshold "${THRESHOLD}" \
    --device   cpu

echo "Predicted: $(grep -c CDS ${PRED} 2>/dev/null || echo 0) CDS lines"

echo ""
echo "=== helion evaluate (strand-agnostic) ==="
helion evaluate "${REF}" "${PRED}" "${FA}"

echo ""
echo "=== helion evaluate (strand-aware) ==="
helion evaluate "${REF}" "${PRED}" "${FA}" --strand-aware

echo ""
echo "Finished: $(date)"
