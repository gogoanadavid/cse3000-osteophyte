#!/bin/bash
set -euo pipefail

INDEX_CSV="${INDEX_CSV:-outputs/index.csv}"
PYTHON_BIN="${PYTHON:-python3}"
BUDGETS="${BUDGETS:-0,64,128,256,512,1024,2048,4096,8192,full}"

for SEED in 0 1 2; do
  SCORES="outputs/predictions/binary_train_scores_seed${SEED}.csv"
  OUT_DIR="budgets/score_stratified_seed${SEED}"
  if [[ ! -f "$SCORES" ]]; then
    echo "Missing $SCORES" >&2
    exit 1
  fi
  "$PYTHON_BIN" -m src.sampling \
    --index "$INDEX_CSV" \
    --strategy score_stratified \
    --scores "$SCORES" \
    --score-column severity_proxy \
    --seed "$SEED" \
    --budgets "$BUDGETS" \
    --out-dir "$OUT_DIR"
  "$PYTHON_BIN" -m src.verify_budgets --index "$INDEX_CSV" --budget-root "$OUT_DIR"
done
