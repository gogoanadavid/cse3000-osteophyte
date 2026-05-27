# 01 Binary Baseline

This experiment estimates the severity signal available from binary
location-specific supervision alone. It trains one binary logit per hip
osteophyte location:

```text
osteo_acet_inf
osteo_acet_sup
osteo_fem_inf
osteo_fem_sup
```

OARSI grades are collapsed only for training:

```text
0 -> absent
1, 2, 3 -> present
```

Validation still uses available graded OARSI labels to measure whether the
binary confidence score increases with true severity. This answers the baseline
research question: how much location-specific severity information is present in
a binary-only detector?

## Model

The baseline uses the project-local ResNet18 implementation, not torchvision.
The ImageNet ResNet18 checkpoint is loaded from:

```text
/scratch/dgogoana/osteophytes_project/pretrained/resnet18-f37072fd.pth
```

RGB `conv1` weights are averaged to initialize the single-channel X-ray input
stem. The head outputs four binary logits.

## Outputs

Each run writes a timestamped directory under:

```text
/scratch/dgogoana/osteophytes_project/runs/01_binary_baseline/
```

Saved artifacts:

```text
config.json
metrics_history.json
best_model.pt
last_model.pt
val_predictions.csv
```

Primary binary metrics are AUROC and AUPRC. The main severity diagnostic is
Spearman correlation between `p_present` and the true OARSI grade, plus
probability-by-grade summaries.

## Commands

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

DelftBlue jobs:

```bash
sbatch jobs/01_binary_baseline_smoke.sh
sbatch jobs/01_binary_baseline_train.sh
```
