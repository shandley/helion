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
# Predict ONCE with no filter, write mean_exon_score into gene GFF3 attributes,
# then filter+evaluate at each score cutoff using filter_gff3_by_score.py.
# This avoids re-running the expensive prediction step for each cutoff.
#
# Optional overrides via --export: THRESHOLD, ORGANISM, MODEL, CHROM, SCORES

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
# Serialize git reset + maturin install across concurrent jobs on shared conda env.
flock -x -w 180 "${H}/.build.lock" bash -c '
    git fetch -q origin && git reset -q --hard origin/main
    maturin develop 2>&1
' 

FA="${H}/results/${MODEL%_v*}_chr${CHROM}.fa"
REF="${H}/results/ref_${MODEL%_v*}_chr${CHROM}.gff3"
RAW="${H}/results/pred_${MODEL}_chr${CHROM}_t${THRESHOLD}_gs_raw.gff3"

echo "=== helion predict (no score filter) ==="
helion predict \
    "${FA}" \
    "${RAW}" \
    --model     "${H}/models/${MODEL}" \
    --organism  "${ORGANISM}" \
    --threshold "${THRESHOLD}" \
    --device    cpu

N_CDS=$(grep -c CDS "${RAW}" 2>/dev/null || echo 0)
echo "Raw predicted: ${N_CDS} CDS lines"
echo ""

for SCORE in ${SCORES}; do
    PRED="${H}/results/pred_${MODEL}_chr${CHROM}_t${THRESHOLD}_gs${SCORE}.gff3"

    echo "=== min-gene-score=${SCORE} ==="
    python3 "${H}/scripts/filter_gff3_by_score.py" "${RAW}" "${PRED}" "${SCORE}"

    N_CDS=$(grep -c CDS "${PRED}" 2>/dev/null || echo 0)
    echo "Filtered: ${N_CDS} CDS lines"
    echo ""
    helion evaluate "${REF}" "${PRED}" "${FA}" --strand-aware
    echo ""
done

echo "Finished: $(date)"
