#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$ROOT_DIR/experiments/explore_train_params_effect.py"
OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/outputs/train_params_options}"

mkdir -p "$OUT_ROOT"

run_case() {
  local name="$1"
  shift
  echo
  echo "=== Running option: $name ==="
  python "$SCRIPT" \
    --output-dir "$OUT_ROOT/$name" \
    "$@"
}

run_case baseline

run_case optimizer_soap --optimizer soap
run_case attention_linear_bias --attention-linear-bias
run_case context_aware_scaling --context-aware-scaling
run_case op_blocks --op-blocks
run_case qat --qat
run_case label_smoothing_01 --label-smoothing 0.1

echo
echo "Completed all train-option runs."
echo "Results are in: $OUT_ROOT"
