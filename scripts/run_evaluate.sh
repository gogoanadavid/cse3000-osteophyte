#!/bin/bash
set -euo pipefail

DATA_CONFIG="${DATA_CONFIG:-configs/data.json}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to a best.pt file}"
SPLIT="${SPLIT:-test}"
OUT_DIR="${OUT_DIR:-outputs/metrics/eval_${SPLIT}}"
BOOTSTRAP="${BOOTSTRAP:-1000}"
PYTHON_BIN="${PYTHON:-python3}"

"$PYTHON_BIN" -m src.evaluate \
  --data-config "$DATA_CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --split "$SPLIT" \
  --out-dir "$OUT_DIR" \
  --bootstrap "$BOOTSTRAP"
