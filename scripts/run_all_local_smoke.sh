#!/bin/bash
set -euo pipefail
PYTHON_BIN="${PYTHON:-python3}"

rm -rf outputs/synthetic budgets/smoke_score_seed0 outputs/checkpoints/smoke_binary outputs/checkpoints/smoke_ordinal outputs/metrics/smoke_test
mkdir -p outputs/synthetic outputs/predictions

"$PYTHON_BIN" -m src.make_synthetic_h5 --out outputs/synthetic/synthetic.h5 --n 96 --seed 0
"$PYTHON_BIN" -m src.h5_inspect --h5 outputs/synthetic/synthetic.h5 --out outputs/synthetic/h5_structure.txt
"$PYTHON_BIN" -m src.build_index --data-config configs/data_synthetic.json --out outputs/synthetic/index.csv
"$PYTHON_BIN" -m src.audit_data --index outputs/synthetic/index.csv --out-dir outputs/synthetic/audit
"$PYTHON_BIN" -m src.train_binary \
  --data-config configs/data_synthetic.json \
  --train-config configs/binary_pretrain_template.json \
  --seed 0 \
  --out-dir outputs/checkpoints/smoke_binary \
  --epochs 1 \
  --batch-size 12 \
  --num-workers 0
"$PYTHON_BIN" -m src.predict \
  --data-config configs/data_synthetic.json \
  --checkpoint outputs/checkpoints/smoke_binary/best.pt \
  --split train \
  --out outputs/predictions/binary_train_scores_seed0.csv \
  --batch-size 12 \
  --num-workers 0
"$PYTHON_BIN" -m src.sampling \
  --index outputs/synthetic/index.csv \
  --strategy score_stratified \
  --scores outputs/predictions/binary_train_scores_seed0.csv \
  --score-column severity_proxy \
  --seed 0 \
  --budgets 0,8,16,full \
  --out-dir budgets/smoke_score_seed0
"$PYTHON_BIN" -m src.train_ordinal \
  --data-config configs/data_synthetic.json \
  --train-config configs/ordinal_template.json \
  --seed 0 \
  --budget-file budgets/smoke_score_seed0/budget_full.csv \
  --budget-name full \
  --strategy score_stratified \
  --binary-checkpoint outputs/checkpoints/smoke_binary/best.pt \
  --out-dir outputs/checkpoints/smoke_ordinal \
  --epochs 1 \
  --batch-size-all 8 \
  --batch-size-graded 4 \
  --num-workers 0
"$PYTHON_BIN" -m src.evaluate \
  --data-config configs/data_synthetic.json \
  --checkpoint outputs/checkpoints/smoke_ordinal/best.pt \
  --split test \
  --out-dir outputs/metrics/smoke_test \
  --batch-size 12 \
  --num-workers 0 \
  --bootstrap 0

if "$PYTHON_BIN" -m pytest tests; then
  exit 0
fi

"$PYTHON_BIN" tests/test_model_shapes.py
"$PYTHON_BIN" tests/test_losses.py
"$PYTHON_BIN" tests/test_metrics.py
"$PYTHON_BIN" tests/test_sampling.py
