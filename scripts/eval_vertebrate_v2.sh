#!/bin/bash
#SBATCH --job-name=helion-eval-vert-v2
#SBATCH --account=compute2-shandley
#SBATCH --partition=general-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/storage3/fs1/shandley/Active/helion/logs/%j_eval_vert_v2.out
#SBATCH --error=/storage3/fs1/shandley/Active/helion/logs/%j_eval_vert_v2.err

set -euo pipefail

H="/storage3/fs1/shandley/Active/helion"
CONDA_SH="/storage3/fs1/shandley/Active/echobase/miniforge/etc/profile.d/conda.sh"

echo "Job ID:  ${SLURM_JOB_ID}"
echo "Node:    ${SLURMD_NODENAME}"
echo "Started: $(date)"
echo ""

source "${CONDA_SH}"
conda activate echobase-ml
export PYTHONPATH="${H}/python"

source "$HOME/.cargo/env"
cd "${H}"
git fetch -q origin && git reset -q --hard origin/main
maturin develop -q 2>&1 | tail -2

mkdir -p "${H}/results"

# ── Extract chr22 FASTA if not already done ────────────────────────────────────
python - <<'PYEOF'
from pyfaidx import Fasta
from pathlib import Path

out = Path("/storage3/fs1/shandley/Active/helion/results")
p = out / "vertebrate_chr22.fa"
if p.exists():
    print(f"  vertebrate_chr22.fa already exists")
else:
    fa = Fasta("/storage3/fs1/shandley/Active/helion/data/vertebrate/vertebrate.fa")
    seq = str(fa["22"])
    p.write_text(f">22\n{seq}\n")
    print(f"  wrote vertebrate_chr22.fa: {len(seq):,} nt")
PYEOF

# ── Extract chr22 reference GFF3 if not already done ──────────────────────────
REF="${H}/results/ref_vertebrate_chr22.gff3"
if [ ! -f "${REF}" ]; then
    awk '$1=="22"' "${H}/data/vertebrate/vertebrate.gff3" > "${REF}"
    echo "  ref vertebrate chr22: $(grep -c CDS ${REF}) CDS lines"
else
    echo "  ref vertebrate chr22 already exists"
fi

echo ""
echo "=== helion predict (vertebrate_v2, chr22) ==="

PRED="${H}/results/pred_vertebrate_v2_chr22.gff3"
helion predict \
    "${H}/results/vertebrate_chr22.fa" \
    "${PRED}" \
    --model "${H}/models/vertebrate_v2" \
    --organism vertebrate \
    --device cpu

echo "Predicted: $(grep -c CDS ${PRED} 2>/dev/null || echo 0) CDS lines"

echo ""
echo "=== helion evaluate ==="
helion evaluate "${REF}" "${PRED}" "${H}/results/vertebrate_chr22.fa"

echo ""
echo "Finished: $(date)"
