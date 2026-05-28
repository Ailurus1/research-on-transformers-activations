#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_EVAL_DIR="$ROOT_DIR/experiments/base_evals"

echo "Running base eval scripts from: $BASE_EVAL_DIR"

for script in "$BASE_EVAL_DIR"/eval_*.py; do
  name="$(basename "$script")"
  if [[ "$name" == "eval_many.py" ]]; then
    continue
  fi
  echo
  echo ">>> Running $name"
  python "$script"
done

echo
echo "All base eval scripts completed."
