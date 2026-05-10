#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAX_SAMPLES="${MAX_SAMPLES:-32}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

DOMAINS=(
  "masked-language-modeling"
  "text-generation"
  "image-classification"
  "automatic-speech-recognition"
)

for domain in "${DOMAINS[@]}"; do
  echo "=== Running size-effect domain: ${domain} ==="
  python "${ROOT_DIR}/experiments/explore_size_effect.py" \
    --max-samples "${MAX_SAMPLES}" \
    --batch-size "${BATCH_SIZE}" \
    --log-level "${LOG_LEVEL}" \
    --tasks "${domain}"
done

echo "DONE"
