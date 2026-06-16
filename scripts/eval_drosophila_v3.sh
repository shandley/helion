#!/bin/bash
#SBATCH --job-name=helion-eval-droso-v3
#SBATCH --account=compute2-shandley
#SBATCH --partition=general-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/storage3/fs1/shandley/Active/helion/logs/%j_eval_droso_v3.out
#SBATCH --error=/storage3/fs1/shandley/Active/helion/logs/%j_eval_droso_v3.err

# Evaluates drosophila_v3 (sense-oriented training + RC inference) on chr4.
# Uses strand-aware evaluation since predictions now include both strands.

set -euo pipefail

H="/storage3/fs1/shandley/Active/helion"
CONDA_SH="/storage3/fs1/shandley/Active/echobase/miniforge/etc/profile.d/conda.sh"
THRESHOLD="${THRESHOLD:-0.2}"

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
# Serialize git reset + maturin install across concurrent jobs on shared conda env.
flock -x -w 180 "${H}/.build.lock" bash -c '
    git fetch -q origin && git reset -q --hard origin/main
    maturin develop 2>&1
' 

mkdir -p "${H}/results"

FA="${H}/results/drosophila_chr4.fa"
REF="${H}/results/ref_drosophila_chr4.gff3"
PRED="${H}/results/pred_drosophila_v3_chr4_t${THRESHOLD}.gff3"

echo "=== helion predict (drosophila_v3, chr4, t=${THRESHOLD}) ==="
helion predict \
    "${FA}" \
    "${PRED}" \
    --model    "${H}/models/drosophila_v3" \
    --organism insect \
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
