#!/bin/bash
#SBATCH --job-name=helion-bench-augustus
#SBATCH --account=compute2-shandley
#SBATCH --partition=general-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --output=/storage3/fs1/shandley/Active/helion/logs/%j_bench_augustus.out
#SBATCH --error=/storage3/fs1/shandley/Active/helion/logs/%j_bench_augustus.err

# Head-to-head baseline: run AUGUSTUS (--species=human, ab initio) on chr22 and
# evaluate it against the SAME reference with `helion evaluate`, so the numbers
# are directly comparable to Helion's. Calibrates "how close are we".

set -euo pipefail

H="/storage3/fs1/shandley/Active/helion"
CONDA_SH="/storage3/fs1/shandley/Active/echobase/miniforge/etc/profile.d/conda.sh"
FA="${H}/results/vertebrate_chr22.fa"
REF="${H}/results/ref_vertebrate_chr22.gff3"
OUT="${H}/results/augustus_chr22.gff3"

echo "Job ID: ${SLURM_JOB_ID}   Node: ${SLURMD_NODENAME}"
echo "Started: $(date)"

source "${CONDA_SH}"

echo "=== AUGUSTUS (human, ab initio) on chr22 ==="
conda activate augustus
augustus --species=human --gff3=on --softmasking=off "${FA}" > "${OUT}"
echo "AUGUSTUS CDS lines: $(grep -cP '\tCDS\t' "${OUT}" || echo 0)"

echo ""
echo "=== helion evaluate: AUGUSTUS vs reference (strand-aware) ==="
conda activate echobase-ml
export PYTHONPATH="${H}/python"
helion evaluate "${REF}" "${OUT}" "${FA}" --strand-aware --label "AUGUSTUS-human"

echo ""
echo "=== recall diagnostic on AUGUSTUS prediction ==="
python3 "${H}/scripts/recall_diagnostic.py" --ref "${REF}" --pred "${OUT}" --genome "${FA}" || true

echo ""
echo "Finished: $(date)"
