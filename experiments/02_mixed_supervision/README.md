# 02 Mixed Supervision

This experiment tests whether abundant weak/binary labels plus a controlled
budget of complete graded OARSI annotations improves location-specific severity
prediction over the binary baseline.

The main expected output is an annotation-budget curve:

```text
x-axis: number or fraction of complete graded training annotations
y-axis: severity prediction quality
```

## Scientific Comparison

The binary baseline trains only location-specific presence. Mixed supervision
keeps that weak signal and adds a strong subset with complete OARSI grades for:

```text
osteo_acet_inf
osteo_acet_sup
osteo_fem_inf
osteo_fem_sup
```

Strong samples train all ordinal thresholds. Weak samples train presence only.
The default weak mode is `location_binary`, which uses one binary target per
location. The optional `image_binary` mode derives one image-level label and
uses noisy-OR over location probabilities, so a positive image label is not
forced onto a specific location.

## Model Heads

`threshold_independent` outputs three logits per location for:

```text
grade > 0
grade > 1
grade > 2
```

This is threshold-based ordinal regression. It does not enforce ordered
threshold parameters.

`coral` uses one severity score per location and learned positive threshold
increments. This guarantees monotonic threshold probabilities:

```text
P(grade > 0) >= P(grade > 1) >= P(grade > 2)
```

`dual_head` shares the ResNet18 backbone and uses both a binary head and an
ordinal head. The binary head stabilizes detection with weak labels while the
ordinal head calibrates severity on strong labels.

Mixed runs can initialize the backbone from a trained binary baseline:

```bash
--init-from-binary-checkpoint /path/to/best_model.pt
```

Only matching backbone tensors are loaded; final heads are skipped.

## Annotation Budgets

Default random budget experiments sample complete graded training rows
uniformly:

```text
0.05
0.10
0.25
0.50
0.90
```

Severity-aware and per-location-balanced strategies are opt-in. They are useful
because plateau behavior may reflect underrepresentation of grades 2 and 3,
especially at specific locations.

Saved split artifacts:

```text
strong_sample_ids.csv
supervision_split.csv
strong_grade_distribution_by_location.csv
strong_high_grade_coverage_summary.csv
```

## Metrics

Primary severity metrics:

```text
mean Spearman between expected grade and true grade
mean MAE between expected grade and true grade
mean quadratic weighted kappa
```

Secondary binary metrics:

```text
mean AUROC using P(grade > 0)
mean AUPRC using P(grade > 0)
```

The default checkpoint selection metric is severity-sensitive:

```text
--selection-metric mean_spearman
```

Binary AUROC is available as `--selection-metric mean_auroc`, but it is not the
default for mixed-supervision experiments.

## Commands

Smoke run:

```bash
python scripts/07_train_mixed_supervision.py \
  --smoke \
  --supervision-mode mixed \
  --strong-fraction 0.05 \
  --weak-label-mode location_binary \
  --strong-sampling-strategy random \
  --model-head threshold_independent \
  --selection-metric mean_spearman
```

Random 25% budget:

```bash
python scripts/07_train_mixed_supervision.py \
  --supervision-mode mixed \
  --strong-fraction 0.25 \
  --weak-label-mode location_binary \
  --strong-sampling-strategy random \
  --model-head threshold_independent \
  --loss-balance-mode proportional \
  --selection-metric mean_spearman \
  --epochs 10 \
  --batch-size 32
```

Severity-aware 25% budget:

```bash
python scripts/07_train_mixed_supervision.py \
  --supervision-mode mixed \
  --strong-fraction 0.25 \
  --strong-sampling-strategy severity_aware \
  --selection-metric mean_spearman \
  --epochs 10 \
  --batch-size 32
```

Summarize runs:

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

DelftBlue jobs:

```bash
sbatch jobs/02_mixed_supervision_005_smoke.sh
sbatch jobs/02_mixed_supervision_025_random.sh
sbatch jobs/02_mixed_supervision_025_severity_aware.sh
sbatch jobs/02_mixed_supervision_025_coral.sh
```

For binary-initialized mixed training:

```bash
sbatch jobs/02_mixed_supervision_025_binary_init.sh
```

The job automatically uses the newest binary baseline `best_model.pt` under
`/scratch/dgogoana/osteophytes_project/runs/01_binary_baseline/`. To force a
specific binary checkpoint:

```bash
BINARY_CHECKPOINT=/scratch/dgogoana/osteophytes_project/runs/01_binary_baseline/<timestamp>/best_model.pt \
  sbatch jobs/02_mixed_supervision_025_binary_init.sh
```
