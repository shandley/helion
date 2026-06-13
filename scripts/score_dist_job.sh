#!/bin/bash
#SBATCH --job-name=helion-score-dist
#SBATCH --account=compute2-shandley
#SBATCH --partition=general-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/storage3/fs1/shandley/Active/helion/logs/%j_score_dist.out
#SBATCH --error=/storage3/fs1/shandley/Active/helion/logs/%j_score_dist.err

set -euo pipefail

HELION_ROOT="/storage3/fs1/shandley/Active/helion"
CONDA_SH="/storage3/fs1/shandley/Active/echobase/miniforge/etc/profile.d/conda.sh"

echo "Job ID:  ${SLURM_JOB_ID}"
echo "Node:    ${SLURMD_NODENAME}"
echo "Started: $(date)"
echo ""

source "${CONDA_SH}"
conda activate echobase-ml
export PYTHONPATH="${HELION_ROOT}/python"

python "${HELION_ROOT}/scripts/score_dist.py"

echo ""
echo "Finished: $(date)"
echo "SENTINEL_DONE"
