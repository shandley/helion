# FGENESH+

## What it is

FGENESH+ is a gene prediction program developed by Softberry. It extends the base FGENESH ab initio HMM predictor by incorporating protein homology evidence, making it a combined (homology-assisted) gene finder rather than a pure ab initio tool.

The core idea: you supply a genomic DNA sequence plus a known protein that is homologous to a gene you expect to be in that sequence. FGENESH+ uses HMM gene models together with that protein similarity signal to produce significantly more accurate gene structures than ab initio alone.

---

## FGENESH vs. FGENESH+ vs. FGENESH++

These are three related but distinct tools:

| Tool | Category | How it works |
|---|---|---|
| FGENESH | Ab initio | HMM-based gene prediction, no external evidence |
| FGENESH+ | Homology-assisted | HMM + a single user-supplied homologous protein sequence |
| FGENESH++ | Full pipeline | Ab initio + BLAST against protein DB + second-pass HMM with homology scores |

---

## How FGENESH+ Works

1. Run an ab initio predictor (e.g., FGENESH) to get initial gene predictions.
2. Run BLASTP with predicted exon amino acid sequences against a protein database.
3. Identify a homologous protein.
4. Run FGENESH+ with the genomic sequence + that homologous protein; it re-scores exons with elevated weight for regions homologous to the known protein.

Accuracy scales with sequence similarity: higher similarity to the reference protein yields better predictions, potentially up to 100% correct gene structure.

---

## FGENESH++ Pipeline (the full annotation system)

FGENESH++ is a complete automated eukaryotic genome annotation pipeline. It:

- Takes a repeat-masked genome FASTA as input
- Runs ab initio prediction
- BLASTs predicted exons against NR, OrthoDB, or a custom protein DB
- Runs a second prediction pass with homology-weighted scores
- Can incorporate cDNA/mRNA evidence
- Outputs GFF3 annotations plus FASTA files of mRNAs, cDNAs, and proteins

It was developed to produce annotation quality comparable to manual curation at high speed.

---

## Inputs Required

- Hard repeat-masked genome FASTA (required)
- Organism-specific HMM parameter matrix (species-specific, must choose closest available)
- Homologous protein sequence (FGENESH+) or protein database (FGENESH++)
- Optional: cDNA/mRNA sequences for transcript evidence
- Pipeline selection: "mammal" vs. "non-mammal" database mode

---

## Accuracy and Benchmarks

- Predicts ~93% of coding nucleotides with ~90% specificity
- In human: ~80% of exons correctly predicted
- 50-100x faster than GENSCAN
- Outperforms GeneMark by ~11% on correct gene model identification (1,353-gene test set)
- Best-in-class for bread wheat gene prediction at both nucleotide and exon levels
- In mouse: highest correlation coefficient (0.87 at nucleotide level) among tested tools
- In maize: most accurate among 5 tools tested (FGENESH, GeneMark.hmm, GENSCAN, GlimmerR, GRAIL)
- FGENESH+ particularly strong with mouse proteins and for Drosophila when similarity > 60%

---

## Availability

FGENESH+ is commercial software from Softberry. It is available:
- As a web server at softberry.com (limited free use)
- Via licensed installation
- Through the Australian BioCommons FGENESH++ workflow (Galaxy-based, available to Australian researchers)
- Integrated in JGI's MycoCosm fungal genome annotation pipeline

---

## Typical Use Case

In a genome annotation workflow, FGENESH+ sits between ab initio prediction and full pipeline annotation. The typical approach is: run FGENESH for initial predictions, BLAST exons against a protein DB, then use FGENESH+ or FGENESH++ for a homology-guided second pass before combining with RNA-seq evidence in a consensus tool like EVidenceModeler or MAKER.

---

## Sources

- FGENESH+ - HMM plus similar protein-based gene prediction (Softberry): http://www.softberry.com/berry.phtml?topic=fgenes_plus&group=programs&subgroup=gfs
- Fgenesh+ description (MolQuest): http://www.molquest.com/help/2.3/programs/Fgenesh+/description.html
- FGENESH+ HELP (Softberry): https://www.softberry.com/berry.phtml?topic=fgenes_plus&group=help&subgroup=gfs
- Fgenesh++ pipeline help (Softberry): http://www.softberry.com/berry.phtml?topic=fgenesh_plus_plus&group=help&subgroup=pipelines
- Genome Annotation with FgenesH++ (Australian BioCommons): https://australianbiocommons.github.io/how-to-guides/genome_annotation/Fgenesh
- Automatic annotation of eukaryotic genes, pseudogenes and promoters (PMC): https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1810547/
- A benchmark study of ab initio gene prediction methods in diverse eukaryotic organisms (BMC Genomics): https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7147072/
- Evaluation of five ab initio gene prediction programs for maize (Springer): https://link.springer.com/article/10.1007/s11103-005-0271-1
- Gene identification programs in bread wheat (PubMed): https://pubmed.ncbi.nlm.nih.gov/24124688/
