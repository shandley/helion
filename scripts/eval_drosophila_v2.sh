#!/bin/bash
#SBATCH --job-name=helion-eval-droso-v2
#SBATCH --account=compute2-shandley
#SBATCH --partition=general-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/storage3/fs1/shandley/Active/helion/logs/%j_eval_droso_v2.out
#SBATCH --error=/storage3/fs1/shandley/Active/helion/logs/%j_eval_droso_v2.err

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

# ── Extract chr4 FASTA if not already done ─────────────────────────────────────
python - <<'PYEOF'
from pyfaidx import Fasta
from pathlib import Path

out = Path("/storage3/fs1/shandley/Active/helion/results")
p = out / "drosophila_chr4.fa"
if p.exists():
    print(f"  drosophila_chr4.fa already exists")
else:
    fa = Fasta("/storage3/fs1/shandley/Active/helion/data/drosophila/drosophila.fa")
    seq = str(fa["4"])
    p.write_text(f">4\n{seq}\n")
    print(f"  wrote drosophila_chr4.fa: {len(seq):,} nt")
PYEOF

# ── Extract chr4 reference GFF3 if not already done ───────────────────────────
REF="${H}/results/ref_drosophila_chr4.gff3"
if [ ! -f "${REF}" ]; then
    awk '$1=="4"' "${H}/data/drosophila/drosophila.gff3" > "${REF}"
    echo "  ref drosophila chr4: $(grep -c CDS ${REF}) CDS lines"
else
    echo "  ref drosophila chr4 already exists"
fi

echo ""
echo "=== helion predict (drosophila_v2, chr4) ==="

PRED="${H}/results/pred_drosophila_v2_chr4.gff3"
helion predict \
    "${H}/results/drosophila_chr4.fa" \
    "${PRED}" \
    --model "${H}/models/drosophila_v2" \
    --organism insect \
    --device cpu

echo "Predicted: $(grep -c CDS ${PRED} 2>/dev/null || echo 0) CDS lines"

echo ""
echo "=== helion evaluate ==="
helion evaluate "${REF}" "${PRED}" "${H}/results/drosophila_chr4.fa"

echo ""
echo "Finished: $(date)"
