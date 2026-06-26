#!/bin/bash
#SBATCH --job-name=helion-recall-diag
#SBATCH --account=compute2-shandley
#SBATCH --partition=general-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=/storage3/fs1/shandley/Active/helion/logs/%j_recall_diag.out
#SBATCH --error=/storage3/fs1/shandley/Active/helion/logs/%j_recall_diag.err

# Classify every MISSED chr22 reference exon (boundary-off / undetected / isoform-only)
# against the existing UNFILTERED vertebrate_v3 prediction already on disk.
# Pure-python analysis: no prediction, no maturin, no GPU.
#
# Optional overrides via --export: MODEL, CHROM, TOLERANCE

set -euo pipefail

H="/storage3/fs1/shandley/Active/helion"
CONDA_SH="/storage3/fs1/shandley/Active/echobase/miniforge/etc/profile.d/conda.sh"

MODEL="${MODEL:-vertebrate_v3}"
CHROM="${CHROM:-22}"
TOLERANCE="${TOLERANCE:-0}"

ORG="${MODEL%_v*}"
REF="${H}/results/ref_${ORG}_chr${CHROM}.gff3"
PRED="${H}/results/pred_${MODEL}_chr${CHROM}_t0.3_gs0.0.gff3"
FA="${H}/results/${ORG}_chr${CHROM}.fa"

echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      ${SLURMD_NODENAME}"
echo "Model:     ${MODEL}  Chrom: ${CHROM}  Tolerance: ${TOLERANCE}"
echo "Ref:       ${REF}"
echo "Pred:      ${PRED}"
echo "Started:   $(date)"
echo ""

source "${CONDA_SH}"
conda activate echobase-ml
export PYTHONPATH="${H}/python"

cd "${H}"
# Pull the latest script. No maturin: the diagnostic imports only helion.evaluate
# (pure python) + numpy. Still serialize git reset against any concurrent job.
flock -x -w 180 "${H}/.build.lock" bash -c '
    git fetch -q origin && git reset -q --hard origin/main
'

python3 "${H}/scripts/recall_diagnostic.py" \
    --ref       "${REF}" \
    --pred      "${PRED}" \
    --genome    "${FA}" \
    --tolerance "${TOLERANCE}"

echo ""
echo "Finished: $(date)"
