#!/bin/bash
set -euo pipefail

DATA_CONFIG="${DATA_CONFIG:-configs/data.json}"
INDEX_CSV="${INDEX_CSV:-outputs/index.csv}"
PYTHON_BIN="${PYTHON:-python3}"

H5_PATH="$(DATA_CONFIG="$DATA_CONFIG" "$PYTHON_BIN" -c 'import json, os; print(json.load(open(os.environ["DATA_CONFIG"], encoding="utf-8"))["h5_path"])')"
"$PYTHON_BIN" -m src.h5_inspect --h5 "$H5_PATH" --out outputs/logs/h5_structure.txt
"$PYTHON_BIN" -m src.build_index --data-config "$DATA_CONFIG" --out "$INDEX_CSV"
"$PYTHON_BIN" -m src.audit_data --index "$INDEX_CSV" --out-dir outputs/audit
