#!/bin/bash
#SBATCH --job-name=helion-filter-sweep
#SBATCH --account=compute2-shandley
#SBATCH --partition=general-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=/storage3/fs1/shandley/Active/helion/logs/%j_filter_sweep.out
#SBATCH --error=/storage3/fs1/shandley/Active/helion/logs/%j_filter_sweep.err

# Sweep --min-gene-score over an EXISTING prediction GFF3 (no re-prediction).
# Filter many: find the best operating point for a prediction already on disk.
# Pure python (filter + helion evaluate); no maturin, no GPU.
#
# Required via --export: RAW (path to the gs0.0 prediction GFF3)
# Optional: REF, FA, SCORES, STRAND_FLAG

set -euo pipefail

H="/storage3/fs1/shandley/Active/helion"
CONDA_SH="/storage3/fs1/shandley/Active/echobase/miniforge/etc/profile.d/conda.sh"

RAW="${RAW:?must pass RAW=<gs0.0 prediction gff3>}"
REF="${REF:-${H}/results/ref_vertebrate_chr22.gff3}"
FA="${FA:-${H}/results/vertebrate_chr22.fa}"
SCORES="${SCORES:-0.0 0.4 0.6 0.8 0.9 1.0 1.1 1.2 1.4}"
STRAND_FLAG="${STRAND_FLAG:---strand-aware}"

echo "Job ID:  ${SLURM_JOB_ID}   Node: ${SLURMD_NODENAME}"
echo "Raw:     ${RAW}"
echo "Scores:  ${SCORES}"
echo "Started: $(date)"
echo ""

source "${CONDA_SH}"
conda activate echobase-ml
export PYTHONPATH="${H}/python"
cd "${H}"
flock -x -w 180 "${H}/.build.lock" bash -c '
    git fetch -q origin && git reset -q --hard origin/main
'

for SCORE in ${SCORES}; do
    OUT="${RAW%.gff3}_gs${SCORE}.gff3"
    echo "=== min-gene-score=${SCORE} ==="
    python3 "${H}/scripts/filter_gff3_by_score.py" "${RAW}" "${OUT}" "${SCORE}"
    helion evaluate "${REF}" "${OUT}" "${FA}" ${STRAND_FLAG}
    echo ""
done

echo "Finished: $(date)"
