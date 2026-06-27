#!/bin/bash
#SBATCH --job-name=helion-hyena-phase0
#SBATCH --account=compute2-shandley
#SBATCH --partition=general-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/storage3/fs1/shandley/Active/helion/logs/%j_hyena_phase0.out
#SBATCH --error=/storage3/fs1/shandley/Active/helion/logs/%j_hyena_phase0.err

# Phase 0 of the DNA-embedding fusion: verify the vendored HyenaDNA offset-cosine
# features align 1:1 with Helion coordinates and that the offset-3 inversion
# separates coding from non-coding on a real chr22 gene. Pure python (no maturin,
# no GPU): HyenaDNA-small is tiny, runs on CPU. Uses the pre-downloaded HF cache.

set -euo pipefail

H="/storage3/fs1/shandley/Active/helion"
CONDA_SH="/storage3/fs1/shandley/Active/echobase/miniforge/etc/profile.d/conda.sh"

export HF_HOME="${H}/hf_cache"
export HF_HUB_OFFLINE=1          # rely on the login-node pre-download; no compute-node internet
export TOKENIZERS_PARALLELISM=false

echo "Job ID: ${SLURM_JOB_ID}  Node: ${SLURMD_NODENAME}  Started: $(date)"

source "${CONDA_SH}"
conda activate echobase-ml
export PYTHONPATH="${H}/python"
cd "${H}"
flock -x -w 180 "${H}/.build.lock" bash -c '
    git fetch -q origin && git reset -q --hard origin/main
'

python scripts/hyenadna_phase0_check.py --device cpu

echo "Finished: $(date)"
