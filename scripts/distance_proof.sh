#!/bin/bash
#SBATCH --job-name=helion-dist-proof
#SBATCH --account=compute2-shandley
#SBATCH --partition=general-gpu
#SBATCH --gres=gpu:H100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/storage3/fs1/shandley/Active/helion/logs/%j_dist_proof.out
#SBATCH --error=/storage3/fs1/shandley/Active/helion/logs/%j_dist_proof.err

# Phase A proof: does the signed-distance head's zero-crossings localize splice
# sites more sharply than the classification peaks? (vertebrate_distance, chr22)

set -euo pipefail
H="/storage3/fs1/shandley/Active/helion"
source "/storage3/fs1/shandley/Active/echobase/miniforge/etc/profile.d/conda.sh"
conda activate echobase-ml
export PYTHONPATH="${H}/python"
cd "${H}"
flock -x -w 300 "${H}/.build.lock" bash -c '
    git fetch -q origin && git reset -q --hard origin/main
    source "$HOME/.cargo/env"; maturin develop -q 2>&1 | tail -2
'
python scripts/distance_head_proof.py \
    --model "${H}/models/vertebrate_distance" \
    --genome "${H}/results/vertebrate_chr22.fa" \
    --ref "${H}/results/ref_vertebrate_chr22.gff3" \
    --device cuda
echo "Finished: $(date)"
