#!/bin/bash
set -euo pipefail

DATA_CONFIG="${DATA_CONFIG:-configs/data.json}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/binary_pretrain_template.json}"
SEED="${SEED:-0}"
OUT_DIR="${OUT_DIR:-outputs/checkpoints/binary_seed${SEED}}"
PYTHON_BIN="${PYTHON:-python3}"

"$PYTHON_BIN" -m src.train_binary \
  --data-config "$DATA_CONFIG" \
  --train-config "$TRAIN_CONFIG" \
  --seed "$SEED" \
  --out-dir "$OUT_DIR"
