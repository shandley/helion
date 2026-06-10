# Helion Training Results -- Round 1

## Setup

Four signal models trained on RIS (H100 80GB) using chromosome-level validation splits.
All models: 256-channel dilated residual CNN, AdamW lr=1e-4, cosine LR decay, 50 epochs.

Training data sourced from Ensembl 113 (Drosophila, vertebrate) and NCBI TAIR10.1 (Arabidopsis).

| Model | Organism | Genome | Training chroms | Val chrom | Train windows | Val windows |
|---|---|---|---|---|---|---|
| drosophila | Insect | BDGP6.46 | all except chr4 | chr4 | ~30k | ~3k |
| vertebrate | Vertebrate | GRCh38 chr1+19+22 | chr1, chr19 | chr22 | 258,366 | 34,469 |
| plant | Plant | TAIR10 | chr1-3, chr5 | chr4 | 4,368 | 879 |
| plant_w2000 | Plant | TAIR10 | chr1-3, chr5 | chr4 | 51,393 | 9,637 |

---

## Final training losses (epoch 50)

| Model | Window | Train loss | Val loss | Gap | Train time |
|---|---|---|---|---|---|
| drosophila | 5kb | 1.2892 | 1.2936 | 0.004 | 6.6h |
| vertebrate | 5kb | 1.2980 | 1.2972 | -0.001 | 18.9h |
| plant | 5kb | 1.2898 | 1.3941 | 0.104 | ~1h |
| plant_w2000 | 2kb | 1.3030 | 1.3567 | 0.054 | 1.6h |

Random baseline (8 classes, uniform): ln(8) = 2.079. All models learned well above baseline.

---

## Observations

**Drosophila** converged cleanly with a near-zero train/val gap (0.004). The validation chromosome
(chr4) is representative of the training data. Loss plateaued around epoch 40 at 1.289/1.294.

**Vertebrate** shows the strongest generalization -- val loss (1.2972) is fractionally below train
loss (1.2980), indicating no overfitting. This is likely because chr22 is well-represented by the
sequence diversity in chr1 and chr19. The 258k training windows gave the model enough signal.

**Plant 5kb** has a persistent gap of 0.104 that opened by epoch 5 and never closed. This is a
data starvation pattern: 4,368 training windows is too few for the model to learn features that
generalize to chr4. Arabidopsis chr4 also has unusually high transposable element density relative
to other chromosomes, which may contribute additional distribution shift.

**Plant 2kb (window size comparison):** Using 2kb windows instead of 5kb generates 12x more
training windows from the same genome (51,393 vs 4,368) because Arabidopsis genes average ~2kb.
The train/val gap narrowed from 0.104 to 0.054 -- a meaningful improvement. The plant_w2000 model
is the better plant model. The 5kb model should not be used for plant prediction.

---

## Window size finding

For organisms with short genes (Arabidopsis avg ~2kb, fungal avg ~1.5kb), the default 5kb window
size generates very few samples per gene. The window size should be matched to gene size:

- Vertebrate / large genomes: 5kb window appropriate
- Insect: 5kb window fine (Drosophila avg gene ~5.5kb)
- Plant / fungus: 2kb window recommended

A follow-up run with 1kb windows may narrow the plant gap further.

---

## Next steps

1. Run `helion evaluate` on held-out chromosomes to get Sn/Sp/CC at nucleotide and exon level
2. Determine if plant_w2000 gap (0.054) reflects chr4 distribution shift or insufficient data
3. Expand vertebrate training to more chromosomes (current model trained on only 3)
4. Train fungus model (no data collected yet)
5. Evaluate whether loss plateau signals model capacity limit or data limit -- try channels=512

---

## Model weights

All weights at `/storage3/fs1/shandley/Active/helion/models/` on RIS, 78MB each (fp32).
Best checkpoint saved based on val loss.
