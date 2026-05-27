# Osteophyte Mixed-Supervision Pipeline

This repository implements a fresh experimental pipeline for OARSI osteophyte severity estimation in 224x224 grayscale hip X-ray crops. The main experiment measures how severity estimation improves as expensive complete OARSI grades are added on top of cheap binary osteophyte-presence labels.

## Data Format

The expected dataset is one HDF5 file with image crops, subject metadata, split metadata, side, four binary labels, and four grade labels. The exact HDF5 key names are configured in `configs/data.json`; start from:

```bash
cp configs/data_template.json configs/data.json
python -m src.h5_inspect --h5 /path/to/data.h5 --out outputs/logs/h5_structure.txt
```

Edit `configs/data.json` so `h5_path`, `image_key`, metadata keys, and label keys match the inspection output. DelftBlue users should verify the HDF5 path and keep the existing module/PYTHONPATH/TORCH_HOME conventions in `slurm/`.

## Build And Audit

```bash
python -m src.build_index --data-config configs/data.json --out outputs/index.csv
python -m src.audit_data --index outputs/index.csv --out-dir outputs/audit
```

The index stores one row per crop, converts missing grades to `-1`, validates label ranges, checks grade/binary consistency, and writes train pixel mean/std to `outputs/train_mean_std.json`. Subject overlap across splits is reported as a prominent warning.

## Synthetic Smoke Test

```bash
bash scripts/run_all_local_smoke.sh
```

This creates a tiny synthetic HDF5, builds an index, audits it, trains one binary epoch, predicts train scores, creates a score-stratified budget, trains one ordinal epoch, evaluates, and runs the unit tests.

## Binary Pretraining

```bash
python -m src.train_binary \
  --data-config configs/data.json \
  --train-config configs/binary_pretrain_template.json \
  --seed 0 \
  --out-dir outputs/checkpoints/binary_seed0

python -m src.predict \
  --data-config configs/data.json \
  --checkpoint outputs/checkpoints/binary_seed0/best.pt \
  --split train \
  --out outputs/predictions/binary_train_scores_seed0.csv
```

Binary pretraining uses all training samples but only supervises threshold `y>=1`.

## Mixed-Supervision Experiments After Binary Pretraining

Current binary baselines on DelftBlue are strong enough to use as the budget-0 baseline and ordinal initialization:

```text
seed 0 validation AUROC ~= 0.8255
seed 1 validation AUROC ~= 0.8421
seed 2 validation AUROC ~= 0.8355
```

Generate binary validation predictions:

```bash
for seed in 0 1 2; do
  python -m src.predict \
    --data-config configs/data.json \
    --checkpoint outputs/checkpoints/binary_seed${seed}/best.pt \
    --split val \
    --out outputs/predictions/binary_val_scores_seed${seed}.csv \
    --batch-size 32 \
    --num-workers 2
done
```

Evaluate whether the binary-only `p_ge1` score contains a severity signal. This analysis intentionally ignores `p_ge2` and `p_ge3` from the binary checkpoint:

```bash
for seed in 0 1 2; do
  python -m src.evaluate_binary_baseline_severity \
    --data-config configs/data.json \
    --index outputs/index.csv \
    --predictions outputs/predictions/binary_val_scores_seed${seed}.csv \
    --split val \
    --seed ${seed} \
    --out outputs/metrics/binary_baseline_severity_val_seed${seed}.csv \
    --plot
done
```

Create and verify the main score-stratified graded budgets:

```bash
bash scripts/make_score_stratified_budgets.sh
python -m src.verify_budgets --index outputs/index.csv --budget-root budgets/score_stratified_seed0
```

Run one validation-only mixed sanity experiment at budget 1024:

```bash
bash scripts/run_one_mixed_sanity.sh
```

Equivalent manual commands:

```bash
python -m src.train_ordinal \
  --data-config configs/data.json \
  --train-config configs/ordinal_template.json \
  --seed 0 \
  --budget-file budgets/score_stratified_seed0/budget_1024.csv \
  --budget-name 1024 \
  --strategy score_stratified \
  --binary-checkpoint outputs/checkpoints/binary_seed0/best.pt \
  --out-dir outputs/checkpoints/ordinal/score_stratified_seed0_budget1024 \
  --batch-size-all 32 \
  --batch-size-graded 16 \
  --num-workers 2

python -m src.evaluate \
  --data-config configs/data.json \
  --checkpoint outputs/checkpoints/ordinal/score_stratified_seed0_budget1024/best.pt \
  --split val \
  --out-dir outputs/metrics/score_stratified_seed0_budget1024_val \
  --bootstrap 0 \
  --batch-size 32 \
  --num-workers 2
```

Generate the main mixed-curve job list. Budget 0 is the binary-only baseline and is excluded from ordinal training jobs:

```bash
python -m src.make_job_list \
  --experiment-grid configs/experiment_grid.json \
  --out outputs/job_lists/ordinal_jobs.csv \
  --eval-out outputs/job_lists/eval_jobs.csv
```

Submit the main mixed curve only after the one-run sanity result is acceptable:

```bash
N=$(($(wc -l < outputs/job_lists/ordinal_jobs.csv)-2))
sbatch --array=0-${N}%1 slurm/train_ordinal_array.sbatch
```

Later, run graded-only ablations and sampling comparisons from the same generated job list. Do not touch the test set until validation analysis and model-selection decisions are frozen.

## Budgets

```bash
python -m src.sampling \
  --index outputs/index.csv \
  --strategy score_stratified \
  --scores outputs/predictions/binary_train_scores_seed0.csv \
  --score-column severity_proxy \
  --seed 0 \
  --out-dir budgets/score_stratified_seed0
```

Supported strategies are `random`, `binary_positive_enriched`, `score_stratified`, and `oracle_grade_stratified`. Oracle budgets are labeled exactly as oracle outputs and should not be used for the main deployable curve.

## Ordinal Training

```bash
python -m src.train_ordinal \
  --data-config configs/data.json \
  --train-config configs/ordinal_template.json \
  --seed 0 \
  --budget-file budgets/score_stratified_seed0/budget_1024.csv \
  --budget-name 1024 \
  --strategy score_stratified \
  --binary-checkpoint outputs/checkpoints/binary_seed0/best.pt \
  --out-dir outputs/checkpoints/ordinal/score_stratified_seed0_budget1024
```

For graded-only ablations, pass `--mode graded_only` or set `"mode": "graded_only"` in a copied ordinal config.

## Evaluation And Plots

```bash
python -m src.evaluate \
  --data-config configs/data.json \
  --checkpoint outputs/checkpoints/ordinal/score_stratified_seed0_budget1024/best.pt \
  --split test \
  --out-dir outputs/metrics/score_stratified_seed0_budget1024_test \
  --bootstrap 1000

python -m src.collect_results --metrics-root outputs/metrics --checkpoints-root outputs/checkpoints --binary-baseline-root outputs/metrics --out outputs/results_all.csv
python -m src.plot_curves --results outputs/results_all.csv --out-dir outputs/figures
python -m src.plateau --results outputs/results_all.csv --full-budget-name full --primary-metric quality_mean --higher-is-better true --out outputs/plateau_analysis.json
```

Primary reporting uses balanced ordinal MAE and `quality = 1 - BMAE/3`, not ordinary accuracy. Plateau means a budget is within 0.01 quality of full-budget performance and doubling the budget improves quality by less than 0.005 where such a doubled budget exists.

## Slurm

The Slurm scripts preserve the DelftBlue module stack, `/home/dgogoana/osteophytes_project`, scratch `PYTHONPATH`, and `TORCH_HOME` conventions from the previous jobs.

```bash
sbatch slurm/binary_pretrain_array.sbatch
sbatch slurm/make_budgets.sbatch
python -m src.make_job_list --experiment-grid configs/experiment_grid.json --out outputs/job_lists/ordinal_jobs.csv --eval-out outputs/job_lists/eval_jobs.csv
sbatch --array=0-$(($(wc -l < outputs/job_lists/ordinal_jobs.csv)-2)) slurm/train_ordinal_array.sbatch
sbatch --array=0-$(($(wc -l < outputs/job_lists/eval_jobs.csv)-2)) slurm/evaluate_array.sbatch
sbatch slurm/plot_results.sbatch
```

## Outputs

Checkpoints go to `outputs/checkpoints/`, metrics to `outputs/metrics/`, predictions to `outputs/predictions/`, figures to `outputs/figures/`, logs to `outputs/logs/`, and annotation budgets to `budgets/`.

## Safeguards

Mixed training masks hidden training grades to `-1` for all samples outside the budget file. Pos weights are computed only from visible labels. Validation and test metrics use their own visible grades only. Bootstrap confidence intervals resample subject clusters. Missing labels are explicit and never silently converted to grade 0.

## Troubleshooting

If HDF5 keys fail, rerun `src.h5_inspect` and edit `configs/data.json`. If subject leakage is reported, rebuild splits before final experiments. If validation/test has no grade-3 samples, grade-3 recall, severe miss rate, and `y>=3` metrics may be `NaN`. For CUDA OOM, reduce batch sizes in the JSON configs. For slow HDF5 loading, increase workers up to the allocated CPU count and keep data on scratch. If matplotlib is missing, metric collection still works but plotting and attention overlays will fail with a clear message. If pytest is missing, run the four test files directly with `python tests/test_*.py`.
