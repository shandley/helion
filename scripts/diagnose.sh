#!/bin/bash
#SBATCH --job-name=helion-diagnose
#SBATCH --account=compute2-shandley
#SBATCH --partition=general-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/storage3/fs1/shandley/Active/helion/logs/%j_diagnose.out
#SBATCH --error=/storage3/fs1/shandley/Active/helion/logs/%j_diagnose.err

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

# ── Extract training-set chromosomes ──────────────────────────────────────────
python - <<'PYEOF'
from pyfaidx import Fasta
from pathlib import Path

out = Path("/storage3/fs1/shandley/Active/helion/results")
extractions = [
    # genome                                                  chrom  outfile
    ("/storage3/fs1/shandley/Active/helion/data/drosophila/drosophila.fa", "2L", "drosophila_chr2L.fa"),
    ("/storage3/fs1/shandley/Active/helion/data/plant/arabidopsis.fa",     "1",  "plant_chr1.fa"),
]
for genome_path, chrom, outname in extractions:
    p = out / outname
    if p.exists():
        print(f"  {outname} already exists")
        continue
    fa = Fasta(genome_path)
    seq = str(fa[chrom])
    p.write_text(f">{chrom}\n{seq}\n")
    print(f"  wrote {outname}: {len(seq):,} nt")
PYEOF

# ── Extract reference GFF3s for training chromosomes ─────────────────────────
if [ ! -f "${H}/results/ref_drosophila_chr2L.gff3" ]; then
    awk '$1=="2L"' "${H}/data/drosophila/drosophila.gff3" \
        > "${H}/results/ref_drosophila_chr2L.gff3"
    echo "  ref droso chr2L: $(grep -c CDS ${H}/results/ref_drosophila_chr2L.gff3) CDS lines"
fi
if [ ! -f "${H}/results/ref_plant_chr1.gff3" ]; then
    awk '$1=="1"' "${H}/data/plant/arabidopsis.gff3" \
        > "${H}/results/ref_plant_chr1.gff3"
    echo "  ref plant chr1: $(grep -c CDS ${H}/results/ref_plant_chr1.gff3) CDS lines"
fi

# ── Index new FASTAs ──────────────────────────────────────────────────────────
python - <<'PYEOF'
from pyfaidx import Fasta
from pathlib import Path
for f in [
    "/storage3/fs1/shandley/Active/helion/results/drosophila_chr2L.fa",
    "/storage3/fs1/shandley/Active/helion/results/plant_chr1.fa",
]:
    if not Path(f + ".fai").exists():
        Fasta(f)
        print(f"  indexed {Path(f).name}")
PYEOF

echo ""
echo "=== Running helion predict ==="

predict_if_missing() {
    local fa="$1" gff3="$2" model="$3" organism="$4" threshold="${5:-0.1}"
    if [ -f "${gff3}" ]; then
        echo "  $(basename ${gff3}) already exists ($(grep -c CDS ${gff3} 2>/dev/null || echo 0) CDS)"
        return
    fi
    echo "  predicting $(basename ${fa}) threshold=${threshold}..."
    helion predict "${fa}" "${gff3}" \
        --model "${model}" --organism "${organism}" \
        --device cpu --threshold "${threshold}"
    echo "  done: $(grep -c CDS ${gff3} 2>/dev/null || echo 0) CDS lines"
}

# Drosophila: training chr (chr2L, threshold 0.1)
predict_if_missing \
    "${H}/results/drosophila_chr2L.fa" \
    "${H}/results/pred_drosophila_chr2L_t01.gff3" \
    "${H}/models/drosophila" "insect" "0.1"

# Drosophila: held-out chr4 with lower threshold (0.01)
predict_if_missing \
    "${H}/results/drosophila_chr4.fa" \
    "${H}/results/pred_drosophila_chr4_t001.gff3" \
    "${H}/models/drosophila" "insect" "0.01"

# Plant: training chr (chr1, threshold 0.1)
predict_if_missing \
    "${H}/results/plant_chr1.fa" \
    "${H}/results/pred_plant_chr1_t01.gff3" \
    "${H}/models/plant_w2000" "plant" "0.1"

# Plant: held-out chr4 with lower threshold (0.01)
predict_if_missing \
    "${H}/results/plant_chr4.fa" \
    "${H}/results/pred_plant_chr4_t001.gff3" \
    "${H}/models/plant_w2000" "plant" "0.01"

echo ""
echo "=== helion evaluate ==="

run_eval() {
    local label="$1" ref="$2" pred="$3" fa="$4"
    shift 4
    echo ""
    echo "--- ${label} ---"
    if [ ! -f "${pred}" ]; then
        echo "  prediction file missing"
        return
    fi
    helion evaluate "${ref}" "${pred}" "${fa}" --label "${label}" "$@"
}

# Drosophila: training chromosome (upper bound)
run_eval "Drosophila chr2L (TRAINING, t=0.1)" \
    "${H}/results/ref_drosophila_chr2L.gff3" \
    "${H}/results/pred_drosophila_chr2L_t01.gff3" \
    "${H}/results/drosophila_chr2L.fa"

# Drosophila: held-out chr4, lower threshold
run_eval "Drosophila chr4 (HELD-OUT, t=0.01)" \
    "${H}/results/ref_drosophila_chr4.gff3" \
    "${H}/results/pred_drosophila_chr4_t001.gff3" \
    "${H}/results/drosophila_chr4.fa"

# Plant: training chromosome (upper bound)
run_eval "Plant chr1 (TRAINING, t=0.1)" \
    "${H}/results/ref_plant_chr1.gff3" \
    "${H}/results/pred_plant_chr1_t01.gff3" \
    "${H}/results/plant_chr1.fa" \
    --overlap-stats

# Plant: held-out chr4 -- exact + tolerance + overlap
run_eval "Plant chr4 (HELD-OUT, t=0.1, exact)" \
    "${H}/results/ref_plant_chr4.gff3" \
    "${H}/results/pred_plant_chr4.gff3" \
    "${H}/results/plant_chr4.fa" \
    --overlap-stats

run_eval "Plant chr4 (HELD-OUT, t=0.1, ±10 nt tolerance)" \
    "${H}/results/ref_plant_chr4.gff3" \
    "${H}/results/pred_plant_chr4.gff3" \
    "${H}/results/plant_chr4.fa" \
    --tolerance 10

run_eval "Plant chr4 (HELD-OUT, t=0.01, exact)" \
    "${H}/results/ref_plant_chr4.gff3" \
    "${H}/results/pred_plant_chr4_t001.gff3" \
    "${H}/results/plant_chr4.fa" \
    --overlap-stats

echo ""
echo "Finished: $(date)"
