#!/bin/bash
#SBATCH --job-name=helion-consensus-expt
#SBATCH --account=compute2-shandley
#SBATCH --partition=general-gpu
#SBATCH --gres=gpu:H100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/storage3/fs1/shandley/Active/helion/logs/%j_consensus_expt.out
#SBATCH --error=/storage3/fs1/shandley/Active/helion/logs/%j_consensus_expt.err

# Decode-only experiment (no retraining): re-predict vertebrate_v3 on chr22 with
# the GT-AG/codon consensus filter + donor-score fix wired into the DAG, then
# evaluate and re-run the recall diagnostic. Same model weights as v3, so any
# change is purely from the decoder fixes (Findings 1 + 2).
#
# Compare against the original v3 numbers:
#   recall 47.6% (2,800 TP), donor-canonical 64.0%, BOUNDARY_OFF 2,381.

set -euo pipefail

H="/storage3/fs1/shandley/Active/helion"
CONDA_SH="/storage3/fs1/shandley/Active/echobase/miniforge/etc/profile.d/conda.sh"
THRESHOLD="${THRESHOLD:-0.3}"
DEVICE="${DEVICE:-cuda}"  # H100 prediction; CPU is ~1.5-2h per chr22, GPU is minutes
MODEL="${MODEL:-vertebrate_v3}"  # vertebrate_v4 stacks hard-neg FP suppression with the decoder fixes

echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      ${SLURMD_NODENAME}"
echo "Model:     ${MODEL}"
echo "Threshold: ${THRESHOLD}"
echo "Device:    ${DEVICE}"
echo "Started:   $(date)"
echo ""

source "${CONDA_SH}"
conda activate echobase-ml
export PYTHONPATH="${H}/python"

source "$HOME/.cargo/env"
cd "${H}"
flock -x -w 300 "${H}/.build.lock" bash -c '
    git fetch -q origin && git reset -q --hard origin/main
    maturin develop --release 2>&1 | tail -5
'

echo ""
echo "=== assembly tests (gate: consensus filter must behave) ==="
python -m pytest tests/test_assembly.py -q

FA="${H}/results/vertebrate_chr22.fa"
REF="${H}/results/ref_vertebrate_chr22.gff3"
BOUNDARY_CONTRAST="${BOUNDARY_CONTRAST:-0.0}"
BC_TAG=""
[ "${BOUNDARY_CONTRAST}" != "0.0" ] && BC_TAG="_bc${BOUNDARY_CONTRAST}"
PRED="${H}/results/pred_${MODEL}_chr22_t${THRESHOLD}_gs0.0_consensus${BC_TAG}.gff3"

echo ""
echo "=== helion predict (${MODEL} + consensus decoder, chr22, t=${THRESHOLD}) ==="
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available(), '| device:', '${DEVICE}')"
helion predict \
    "${FA}" \
    "${PRED}" \
    --model             "${H}/models/${MODEL}" \
    --organism          vertebrate \
    --threshold         "${THRESHOLD}" \
    --boundary-contrast "${BOUNDARY_CONTRAST:-0.0}" \
    --device            "${DEVICE}"
echo "Predicted: $(grep -c CDS "${PRED}" 2>/dev/null || echo 0) CDS lines"

echo ""
echo "=== helion evaluate (strand-aware) ==="
helion evaluate "${REF}" "${PRED}" "${FA}" --strand-aware

echo ""
echo "=== recall diagnostic (new consensus prediction) ==="
python3 "${H}/scripts/recall_diagnostic.py" \
    --ref    "${REF}" \
    --pred   "${PRED}" \
    --genome "${FA}"

echo ""
echo "Finished: $(date)"
