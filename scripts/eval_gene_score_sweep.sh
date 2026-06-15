#!/bin/bash
#SBATCH --job-name=helion-gene-score-sweep
#SBATCH --account=compute2-shandley
#SBATCH --partition=general-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=/storage3/fs1/shandley/Active/helion/logs/%j_gene_score_sweep.out
#SBATCH --error=/storage3/fs1/shandley/Active/helion/logs/%j_gene_score_sweep.err

# Sweep --min-gene-score at fixed threshold to find optimal post-filter cutoff.
# Optional overrides via --export: THRESHOLD, ORGANISM, MODEL, CHROM
# Runs sequentially in one job to avoid concurrent git lock collisions.

set -euo pipefail

H="/storage3/fs1/shandley/Active/helion"
CONDA_SH="/storage3/fs1/shandley/Active/echobase/miniforge/etc/profile.d/conda.sh"

THRESHOLD="${THRESHOLD:-0.3}"
ORGANISM="${ORGANISM:-insect}"
MODEL="${MODEL:-drosophila_v3}"
CHROM="${CHROM:-4}"

SCORES="${SCORES:-0.0 0.4 0.6 0.8 0.9 1.0 1.1 1.2 1.4}"

echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      ${SLURMD_NODENAME}"
echo "Model:     ${MODEL}  Organism: ${ORGANISM}  Chrom: ${CHROM}"
echo "Threshold: ${THRESHOLD}"
echo "Scores:    ${SCORES}"
echo "Started:   $(date)"
echo ""

source "${CONDA_SH}"
conda activate echobase-ml
export PYTHONPATH="${H}/python"

source "$HOME/.cargo/env"
cd "${H}"
git fetch -q origin && git reset -q --hard origin/main
maturin develop 2>&1

FA="${H}/results/${MODEL%_v*}_chr${CHROM}.fa"
REF="${H}/results/ref_${MODEL%_v*}_chr${CHROM}.gff3"

for SCORE in ${SCORES}; do
    PRED="${H}/results/pred_${MODEL}_chr${CHROM}_t${THRESHOLD}_gs${SCORE}.gff3"

    echo "=== min-gene-score=${SCORE} ==="
    helion predict \
        "${FA}" \
        "${PRED}" \
        --model         "${H}/models/${MODEL}" \
        --organism      "${ORGANISM}" \
        --threshold     "${THRESHOLD}" \
        --min-gene-score "${SCORE}" \
        --device        cpu

    N_CDS=$(grep -c CDS "${PRED}" 2>/dev/null || echo 0)
    echo "Predicted: ${N_CDS} CDS lines"
    echo ""
    helion evaluate "${REF}" "${PRED}" "${FA}" --strand-aware
    echo ""
done

echo "Finished: $(date)"
