#!/bin/bash
set -euo pipefail

# This script trains one mixed ordinal sanity model. On DelftBlue login nodes,
# submit it through Slurm with:
#   sbatch slurm/one_mixed_sanity.sbatch
if [[ -z "${SLURM_JOB_ID:-}" && "$(hostname 2>/dev/null || true)" == login* && "${ALLOW_LOGIN_TRAINING:-0}" != "1" ]]; then
  echo "Refusing to train on a DelftBlue login node." >&2
  echo "Submit the GPU sanity job instead:" >&2
  echo "  sbatch slurm/one_mixed_sanity.sbatch" >&2
  echo "If you are intentionally inside an interactive compute allocation, SLURM_JOB_ID should be set." >&2
  exit 2
fi

PYTHON_BIN="${PYTHON:-python3}"
DATA_CONFIG="${DATA_CONFIG:-configs/data.json}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/ordinal_template.json}"
SEED="${SEED:-0}"
STRATEGY="${STRATEGY:-score_stratified}"
BUDGET="${BUDGET:-1024}"
BATCH_ALL="${BATCH_ALL:-32}"
BATCH_GRADED="${BATCH_GRADED:-16}"
NUM_WORKERS="${NUM_WORKERS:-2}"

TRAIN_SCORES="outputs/predictions/binary_train_scores_seed${SEED}.csv"
BINARY_CKPT="outputs/checkpoints/binary_seed${SEED}/best.pt"
BUDGET_DIR="budgets/${STRATEGY}_seed${SEED}"
BUDGET_FILE="${BUDGET_DIR}/budget_${BUDGET}.csv"
OUT_DIR="outputs/checkpoints/ordinal/${STRATEGY}_seed${SEED}_budget${BUDGET}"
METRICS_DIR="outputs/metrics/${STRATEGY}_seed${SEED}_budget${BUDGET}_val"

if [[ ! -f "$BINARY_CKPT" ]]; then
  echo "Missing binary checkpoint: $BINARY_CKPT" >&2
  exit 1
fi

if [[ ! -f "$TRAIN_SCORES" ]]; then
  "$PYTHON_BIN" -m src.predict \
    --data-config "$DATA_CONFIG" \
    --checkpoint "$BINARY_CKPT" \
    --split train \
    --out "$TRAIN_SCORES" \
    --batch-size "$BATCH_ALL" \
    --num-workers "$NUM_WORKERS"
fi

if [[ ! -f "$BUDGET_FILE" ]]; then
  "$PYTHON_BIN" -m src.sampling \
    --index outputs/index.csv \
    --strategy "$STRATEGY" \
    --scores "$TRAIN_SCORES" \
    --score-column severity_proxy \
    --seed "$SEED" \
    --out-dir "$BUDGET_DIR"
fi

"$PYTHON_BIN" -m src.verify_budgets --index outputs/index.csv --budget-root "$BUDGET_DIR"

"$PYTHON_BIN" -m src.train_ordinal \
  --data-config "$DATA_CONFIG" \
  --train-config "$TRAIN_CONFIG" \
  --seed "$SEED" \
  --budget-file "$BUDGET_FILE" \
  --budget-name "$BUDGET" \
  --strategy "$STRATEGY" \
  --binary-checkpoint "$BINARY_CKPT" \
  --out-dir "$OUT_DIR" \
  --batch-size-all "$BATCH_ALL" \
  --batch-size-graded "$BATCH_GRADED" \
  --num-workers "$NUM_WORKERS"

"$PYTHON_BIN" -m src.evaluate \
  --data-config "$DATA_CONFIG" \
  --checkpoint "${OUT_DIR}/best.pt" \
  --split val \
  --out-dir "$METRICS_DIR" \
  --batch-size "$BATCH_ALL" \
  --num-workers "$NUM_WORKERS" \
  --bootstrap 0

"$PYTHON_BIN" - "$METRICS_DIR/metrics_summary.json" <<'PY'
import json
p=__import__("sys").argv[1]
with open(p, "r", encoding="utf-8") as f:
    m=json.load(f)
for k in ["bmae_mean", "quality_mean", "qwk_mean", "spearman_mean", "macro_f1_mean", "auc_ge1_mean", "auc_ge2_mean", "auc_ge3_mean"]:
    print(f"{k}: {m.get(k)}")
PY
