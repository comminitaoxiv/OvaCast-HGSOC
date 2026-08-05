# OvaCast

OvaCast integrates pathway-level genomic profiles, structured CT annotations, pathology reports, and clinical covariates with BioMistral-7B. A shared causal-language representation supports Cox survival estimation and structured explanation generation for high-grade serous ovarian carcinoma.

## Environment

The reported configuration uses Python 3.10, PyTorch 2.1, Transformers 4.36, PEFT 0.7, CUDA 12.1, and one NVIDIA A100 with 80 GB memory.

```bash
conda env create -f environment.yml
conda activate ovacast
pip install -e .
```

The container uses the matching PyTorch and CUDA runtime.

```bash
docker build -t ovacast .
```

## Data

Verified public entry points are collected in `datasets.txt`. TCGA-OV supplies RNA-seq, copy-number, mutation, clinical, survival, and de-identified pathology-report inputs. TCIA OV-Radiogenomics supplies seven consensus CT feature categories for the overlapping imaging subset. GSE26712 and GSE9891 supply external microarray cohorts. PTRC-HGSOC supplies proteogenomic profiles and platinum-response labels. KEGG and Reactome pathway definitions must be downloaded under their respective terms.

Prepare a non-overlapping pathway registry after converting the pathway resources to GMT:

```bash
ovacast-prepare --kegg data/raw/kegg.gmt --reactome data/raw/reactome.gmt --output data/derived/pathways.json
```

The cohort metadata JSON and expression TSV schemas are enforced by `ovacast.cohort.io`. Expression TSV files begin with `gene`, followed by patient identifiers. Clinical JSON stores survival time, event status, FIGO stage, grade, debulking, pathology text, alterations, and optional seven-category radiology annotations. Training-only means and standard deviations are frozen before validation or external transfer. GPL96 and GPL570 duplicate probes are resolved by maximum interquartile range. Proteins are mapped to source genes before pathway aggregation.

## Training

The main configuration retains the reported three-phase curriculum: five genomic epochs, ten multimodal epochs, and three explanation epochs. AdamW uses a learning rate of 3e-4, 10% warmup, cosine decay to 10%, mini-batch 4, gradient accumulation 8, effective batch 32, BF16, modality dropout 0.3, LoRA rank 16 on query and value projections, and explanation weight 0.1.

```bash
ovacast-train --config settings/main.yaml --cohort data/derived/tcga_ov.json --pathways data/derived/pathways.json --output runs/tcga_ov
```

The primary TCGA-OV protocol uses a patient-level 70/30 split stratified by event and FIGO stage, with 15% of the training portion reserved for early stopping. The five seeds are 42, 123, 256, 512, and 1024. The TCIA subset uses five folds repeated three times. GEO cohorts are evaluated without refitting. PTRC-HGSOC uses five-fold evaluation for platinum sensitivity.

## Evaluation

```bash
ovacast-evaluate --predictions runs/tcga_ov/predictions.jsonl --output runs/tcga_ov/metrics.json
```

The primary survival outputs are concordance index, five-year time-dependent AUROC, integrated Brier score, bootstrap 95% confidence intervals with 1,000 resamples, and median-risk log-rank comparison. CLOVAR evaluation uses macro AUROC and weighted F1. Platinum response uses AUROC and balanced accuracy. Family-wise baseline comparisons use Benjamini-Hochberg adjustment at q=0.05; the three primary baseline comparisons use Holm-Bonferroni adjustment.

The reported TCGA-OV median C-index is 0.681 with a 95% interval of 0.642–0.720. External GEO performance is 0.633 for GSE26712 and 0.645 for GSE9891. PTRC-HGSOC platinum-response AUROC is 0.851. These are reference results, not guarantees for altered preprocessing, dataset revisions, or hardware kernels.

## Compute

The full curriculum requires approximately 18 A100 GPU-hours and 80 GB device memory. Input sequences are approximately 17,300 tokens within the 32,768-token context. Typical reported inference latency is 2–5 seconds per patient. Raw and derived storage depends on selected GDC and imaging files; inspect manifests before download and retain their SHA-256 digests with each run.

## Layout

`code/ovacast/genomics` contains frozen normalization and deterministic pathway tokenization. `code/ovacast/language` contains modality templates. `code/ovacast/model` contains the BioMistral adapter and survival projection. `code/ovacast/objectives` contains the Cox and joint language objective. `code/ovacast/measures`, `code/ovacast/metrics`, and `code/ovacast/evaluation` contain endpoints, resampling, multiplicity correction, calibration, attribution, and explanation-quality calculations. `code/ovacast/runtime` and `code/ovacast/training` contain seeded curriculum training, scheduling, modality dropout, and atomic state persistence.
