#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAX_SAMPLES="${MAX_SAMPLES-}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

DOMAINS=(
  "masked-language-modeling"
  "text-generation"
  "image-classification"
  "automatic-speech-recognition"
)

for domain in "${DOMAINS[@]}"; do
  echo "Running size-effect domain: ${domain}"
  EXTRA_ARGS=()
  if [[ -n "${MAX_SAMPLES}" ]]; then
    EXTRA_ARGS+=( --max-samples "${MAX_SAMPLES}" )
  fi
  python "${ROOT_DIR}/experiments/explore_size_effect.py" \
    "${EXTRA_ARGS[@]}" \
    --batch-size "${BATCH_SIZE}" \
    --log-level "${LOG_LEVEL}" \
    --tasks "${domain}"
done

python "${ROOT_DIR}/experiments/aggregate_size_effect_results.py" \
  --output-dir "${ROOT_DIR}/outputs/size_effect"

echo "DONE"
