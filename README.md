# Osteophyte Severity Estimation

TU Delft CSE3000 research project on osteophyte detection and OARSI severity
estimation in hip X-ray images.

The repository contains a modular training and evaluation framework for:

- a location-specific binary osteophyte baseline;
- mixed weak/graded supervision experiments;
- ordinal severity heads, including threshold-based and CORAL-style variants;
- annotation-budget summaries and prediction diagnostics.

It does **not** contain copied confidential image data, copied annotations,
model checkpoints, or generated run outputs.

## Research Goal

The main question is whether abundant weak/binary supervision can be combined
with a controlled budget of complete graded OARSI annotations to improve
location-specific osteophyte severity prediction.

The model predicts four hip osteophyte locations:

```text
osteo_acet_inf
osteo_acet_sup
osteo_fem_inf
osteo_fem_sup
```

OARSI grades are:

```text
0 = absent
1 = small
2 = medium
3 = large
```

Binary labels collapse grades `1`, `2`, and `3` into present. The main expected
result is an annotation-budget curve where the x-axis is the amount of complete
graded supervision and the y-axis is severity prediction quality.

## Data Policy

The project is intended to run on DelftBlue. Image data and sensitive derived
artifacts must stay outside this repository.

Default runtime paths:

```text
dataset index:
/scratch/dgogoana/osteophytes_project/audits/dataset_index.csv

H5 images:
/scratch/dgogoana/osteophytes_project/data/all-for-hip-prediction-20260420-0.4mm-224x224.h5

pretrained ResNet18:
/scratch/dgogoana/osteophytes_project/pretrained/resnet18-f37072fd.pth

runs:
/scratch/dgogoana/osteophytes_project/runs/
```

Historical source-data paths are documented in `configs/delftblue.yaml` and
`docs/delftblue_environment.txt`.

## Repository Structure

```text
src/osteophytes/
  labels.py          constants, grade helpers, deterministic sample IDs
  dataset.py         H5-backed PyTorch dataset
  supervision.py     weak/mixed/ordinal supervision splits
  models.py          local ResNet18 and binary/ordinal/dual heads
  ordinal.py         threshold encoding and ordinal probability utilities
  losses.py          binary, ordinal, noisy-OR, and mixed losses
  metrics.py         severity and binary metrics
  training.py        shared training/checkpoint utilities
  evaluation.py      validation and prediction export
  plotting.py        summary plotting helpers

scripts/
  05_train_binary_baseline.py
  07_train_mixed_supervision.py
  08_summarize_mixed_supervision_runs.py
  09_diagnose_mixed_predictions.py

experiments/
  01_binary_baseline/
  02_mixed_supervision/

jobs/
  DelftBlue SLURM job scripts

tests/
  synthetic unit tests for labels, splits, losses, metrics, models, and scripts
```

The scripts `00` to `04` are retained for data discovery, audit, dataset-index
construction, visual audit, and dataloader smoke testing.

## Binary Baseline

The binary baseline trains one logit per osteophyte location using masked
binary BCE. It evaluates binary AUROC/AUPRC and the severity signal in binary
confidence scores via Spearman correlation against true OARSI grades.

Smoke run:

```bash
python scripts/05_train_binary_baseline.py --smoke
```

Normal run:

```bash
python scripts/05_train_binary_baseline.py \
  --epochs 5 \
  --batch-size 32 \
  --num-workers 2 \
  --selection-metric mean_auroc
```

DelftBlue:

```bash
sbatch jobs/01_binary_baseline_smoke.sh
sbatch jobs/01_binary_baseline_train.sh
```

## Mixed Supervision

The mixed trainer uses all weak/binary training rows and a controlled subset of
complete graded rows. Strong samples train full ordinal severity thresholds;
weak samples train presence only.

Default behavior:

```text
supervision_mode       mixed
strong_fraction        0.05
weak_label_mode        location_binary
sampling_strategy      random
model_head             threshold_independent
loss_balance_mode      proportional
selection_metric       mean_spearman
```

Smoke run:

```bash
python scripts/07_train_mixed_supervision.py --smoke
```

Recommended first real mixed run:

```bash
python scripts/07_train_mixed_supervision.py \
  --supervision-mode mixed \
  --strong-fraction 0.25 \
  --weak-label-mode location_binary \
  --strong-sampling-strategy random \
  --model-head threshold_independent \
  --selection-metric mean_spearman \
  --epochs 10 \
  --batch-size 32
```

Severity-aware 25% run:

```bash
python scripts/07_train_mixed_supervision.py \
  --supervision-mode mixed \
  --strong-fraction 0.25 \
  --strong-sampling-strategy severity_aware \
  --selection-metric mean_spearman \
  --epochs 10 \
  --batch-size 32
```

DelftBlue examples:

```bash
sbatch jobs/02_mixed_supervision_005_smoke.sh
sbatch jobs/02_mixed_supervision_025_random.sh
sbatch jobs/02_mixed_supervision_025_severity_aware.sh
sbatch jobs/02_mixed_supervision_025_coral.sh
```

For binary-initialized mixed training:

```bash
BINARY_CHECKPOINT=/scratch/dgogoana/osteophytes_project/runs/01_binary_baseline/<timestamp>/best_model.pt \
  sbatch jobs/02_mixed_supervision_025_binary_init.sh
```

## Outputs

Each run writes a timestamped directory under `/scratch/.../runs/`.

Training outputs:

```text
config.json
metrics_history.json
best_model.pt
last_model.pt
val_predictions.csv
```

Mixed-supervision split outputs:

```text
strong_sample_ids.csv
supervision_split.csv
strong_grade_distribution_by_location.csv
strong_high_grade_coverage_summary.csv
```

Summarize budget curves:

```bash
python scripts/08_summarize_mixed_supervision_runs.py \
  --runs-root /scratch/dgogoana/osteophytes_project/runs/02_mixed_supervision \
  --output-dir /scratch/dgogoana/osteophytes_project/runs/02_mixed_supervision/summary
```

Diagnose one run:

```bash
python scripts/09_diagnose_mixed_predictions.py \
  --run-dir /scratch/dgogoana/osteophytes_project/runs/02_mixed_supervision/<experiment>/<timestamp>
```

## Tests

Local tests are synthetic and skip torch/pandas/h5py/matplotlib-specific cases
when those packages are unavailable:

```bash
pytest
```

At minimum, syntax can be checked with:

```bash
python3 -m compileall src scripts tests
```
