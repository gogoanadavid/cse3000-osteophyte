#!/bin/bash
set -euo pipefail

DATA_CONFIG="${DATA_CONFIG:-configs/data.json}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/ordinal_template.json}"
SEED="${SEED:-0}"
STRATEGY="${STRATEGY:-score_stratified}"
BUDGET="${BUDGET:-1024}"
MODE="${MODE:-mixed}"
BUDGET_FILE="${BUDGET_FILE:-budgets/${STRATEGY}_seed${SEED}/budget_${BUDGET}.csv}"
BINARY_CHECKPOINT="${BINARY_CHECKPOINT:-outputs/checkpoints/binary_seed${SEED}/best.pt}"
OUT_DIR="${OUT_DIR:-outputs/checkpoints/ordinal/${STRATEGY}_seed${SEED}_budget${BUDGET}}"
PYTHON_BIN="${PYTHON:-python3}"

"$PYTHON_BIN" -m src.train_ordinal \
  --data-config "$DATA_CONFIG" \
  --train-config "$TRAIN_CONFIG" \
  --seed "$SEED" \
  --budget-file "$BUDGET_FILE" \
  --budget-name "$BUDGET" \
  --strategy "$STRATEGY" \
  --mode "$MODE" \
  --binary-checkpoint "$BINARY_CHECKPOINT" \
  --out-dir "$OUT_DIR"
