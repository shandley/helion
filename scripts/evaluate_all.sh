#!/bin/bash
#SBATCH --job-name=helion-evaluate
#SBATCH --partition=general-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/storage3/fs1/shandley/Active/helion/logs/%j_evaluate.out
#SBATCH --error=/storage3/fs1/shandley/Active/helion/logs/%j_evaluate.err

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

# ── Extract val chromosome FASTAs if not already done ─────────────────────────
python - <<'PYEOF'
from pyfaidx import Fasta
from pathlib import Path

out = Path("/storage3/fs1/shandley/Active/helion/results")
for genome, chrom, name in [
    ("/storage3/fs1/shandley/Active/helion/data/drosophila/drosophila.fa", "4",  "drosophila_chr4.fa"),
    ("/storage3/fs1/shandley/Active/helion/data/plant/arabidopsis.fa",     "4",  "plant_chr4.fa"),
    ("/storage3/fs1/shandley/Active/helion/data/vertebrate/vertebrate.fa", "22", "vertebrate_chr22.fa"),
]:
    p = out / name
    if p.exists():
        print(f"  {name} already exists")
        continue
    fa = Fasta(genome)
    seq = str(fa[chrom])
    p.write_text(f">{chrom}\n{seq}\n")
    print(f"  wrote {name}: {len(seq):,} nt")
PYEOF

# ── Extract val chromosome reference GFF3s if not already done ────────────────
for spec in "drosophila:4:drosophila.gff3" "plant:4:arabidopsis.gff3" "vertebrate:22:vertebrate.gff3"; do
    name=$(echo $spec | cut -d: -f1)
    chrom=$(echo $spec | cut -d: -f2)
    src=$(echo $spec | cut -d: -f3)
    ref="${H}/results/ref_${name}_chr${chrom}.gff3"
    if [ ! -f "${ref}" ]; then
        awk "\$1==\"${chrom}\"" "${H}/data/${name}/${src}" > "${ref}"
        echo "  ref ${name} chr${chrom}: $(grep -c CDS ${ref}) CDS lines"
    else
        echo "  ref ${name} chr${chrom} already exists"
    fi
done

echo ""
echo "=== Running helion predict ==="

# Run predictions (skip if GFF3 already exists)
for spec in "drosophila:insect:drosophila_chr4" "plant:plant:plant_chr4"; do
    organism=$(echo $spec | cut -d: -f1)
    clade=$(echo $spec | cut -d: -f2)
    base=$(echo $spec | cut -d: -f3)

    pred="${H}/results/pred_${base}.gff3"
    if [ -f "${pred}" ]; then
        echo "  ${organism} prediction already exists ($(grep -c CDS ${pred} 2>/dev/null || echo 0) CDS lines)"
        continue
    fi

    model="${H}/models"
    [ "${organism}" = "plant" ] && model="${model}/plant_w2000" || model="${model}/${organism}"

    echo "  predicting ${organism} chr${base##*chr}..."
    helion predict \
        "${H}/results/${base}.fa" \
        "${pred}" \
        --model "${model}" \
        --organism "${clade}" \
        --device cpu
    echo "  done: $(grep -c CDS ${pred} 2>/dev/null || echo 0) CDS lines predicted"
done

echo ""
echo "=== helion evaluate ==="

for spec in "drosophila:4" "plant:4"; do
    name=$(echo $spec | cut -d: -f1)
    chrom=$(echo $spec | cut -d: -f2)
    ref="${H}/results/ref_${name}_chr${chrom}.gff3"
    pred="${H}/results/pred_${name}_chr${chrom}.gff3"
    fa="${H}/results/${name}_chr${chrom}.fa"

    echo ""
    echo "--- ${name} chr${chrom} ---"
    if [ -f "${pred}" ]; then
        helion evaluate "${ref}" "${pred}" "${fa}"
    else
        echo "  prediction file missing, skipping"
    fi
done

echo ""
echo "Finished: $(date)"
