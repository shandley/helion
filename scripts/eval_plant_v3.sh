#!/bin/bash
#SBATCH --job-name=helion-eval-plant-v3
#SBATCH --account=compute2-shandley
#SBATCH --partition=general-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/storage3/fs1/shandley/Active/helion/logs/%j_eval_plant_v3.out
#SBATCH --error=/storage3/fs1/shandley/Active/helion/logs/%j_eval_plant_v3.err

# Evaluates plant_v3 (sense-oriented training + RC inference) on chr4.
# Reports both strand-agnostic and strand-aware metrics.

set -euo pipefail

H="/storage3/fs1/shandley/Active/helion"
CONDA_SH="/storage3/fs1/shandley/Active/echobase/miniforge/etc/profile.d/conda.sh"
THRESHOLD="${THRESHOLD:-0.2}"

echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      ${SLURMD_NODENAME}"
echo "Threshold: ${THRESHOLD}"
echo "Started:   $(date)"
echo ""

source "${CONDA_SH}"
conda activate echobase-ml
export PYTHONPATH="${H}/python"

source "$HOME/.cargo/env"
cd "${H}"
git fetch -q origin && git reset -q --hard origin/main
flock -x -w 120 "${H}/.maturin_build.lock" maturin develop 2>&1

mkdir -p "${H}/results"

FA="${H}/results/plant_chr4.fa"
REF="${H}/results/ref_plant_chr4.gff3"
PRED="${H}/results/pred_plant_v3_chr4_t${THRESHOLD}.gff3"

# Extract chr4 FASTA and reference if not already done
python - <<'PYEOF'
from pyfaidx import Fasta
from pathlib import Path
import subprocess

out = Path("/storage3/fs1/shandley/Active/helion/results")
fa_path = out / "plant_chr4.fa"
ref_path = out / "ref_plant_chr4.gff3"

if not fa_path.exists():
    fa = Fasta("/storage3/fs1/shandley/Active/helion/data/plant/arabidopsis.fa")
    seq = str(fa["4"])
    fa_path.write_text(f">4\n{seq}\n")
    print(f"  wrote plant_chr4.fa: {len(seq):,} nt")
else:
    print("  plant_chr4.fa already exists")

if not ref_path.exists():
    result = subprocess.run(
        ["awk", '$1=="4"', str(Path("/storage3/fs1/shandley/Active/helion/data/plant/arabidopsis.gff3"))],
        capture_output=True, text=True
    )
    ref_path.write_text(result.stdout)
    cds = result.stdout.count("\tCDS\t")
    print(f"  wrote ref_plant_chr4.gff3: {cds} CDS lines")
else:
    print("  ref_plant_chr4.gff3 already exists")
PYEOF

echo ""
echo "=== helion predict (plant_v3, chr4, t=${THRESHOLD}) ==="
helion predict \
    "${FA}" \
    "${PRED}" \
    --model    "${H}/models/plant_v3" \
    --organism plant \
    --threshold "${THRESHOLD}" \
    --device   cpu

echo "Predicted: $(grep -c CDS ${PRED} 2>/dev/null || echo 0) CDS lines"

echo ""
echo "=== helion evaluate (strand-agnostic) ==="
helion evaluate "${REF}" "${PRED}" "${FA}"

echo ""
echo "=== helion evaluate (strand-aware) ==="
helion evaluate "${REF}" "${PRED}" "${FA}" --strand-aware

echo ""
echo "Finished: $(date)"
