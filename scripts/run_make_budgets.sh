#!/bin/bash
set -euo pipefail

INDEX_CSV="${INDEX_CSV:-outputs/index.csv}"
SEED="${SEED:-0}"
STRATEGY="${STRATEGY:-score_stratified}"
SCORES="${SCORES:-outputs/predictions/binary_train_scores_seed${SEED}.csv}"
OUT_DIR="${OUT_DIR:-budgets/${STRATEGY}_seed${SEED}}"
PYTHON_BIN="${PYTHON:-python3}"

if [[ "$STRATEGY" == "score_stratified" ]]; then
  "$PYTHON_BIN" -m src.sampling --index "$INDEX_CSV" --strategy "$STRATEGY" --scores "$SCORES" --score-column severity_proxy --seed "$SEED" --out-dir "$OUT_DIR"
else
  "$PYTHON_BIN" -m src.sampling --index "$INDEX_CSV" --strategy "$STRATEGY" --seed "$SEED" --out-dir "$OUT_DIR"
fi
