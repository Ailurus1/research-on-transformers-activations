#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${ROOT_DIR}/outputs/domain_effect"
SUMMARY_CSV="${ROOT_DIR}/experiment_domain_effect.csv"
BATCH_SIZE="${BATCH_SIZE:-1}"

MAX_SAMPLES="${MAX_SAMPLES-}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

DOMAINS=(
  "masked-language-modeling"
#   "machine-translation"
  "text-generation"
  "image-classification"
  "automatic-speech-recognition"
)

for domain in "${DOMAINS[@]}"; do
  echo "Domain: ${domain}"
  EXTRA_ARGS=()
  if [[ -n "${MAX_SAMPLES}" ]]; then
    EXTRA_ARGS+=( --max-samples "${MAX_SAMPLES}" )
  fi
  python "${ROOT_DIR}/experiments/explore_domain_effect.py" \
    "${EXTRA_ARGS[@]}" \
    --batch-size "${BATCH_SIZE}" \
    --log-level "${LOG_LEVEL}" \
    --tasks "${domain}"
done

python3 experiments/aggregate_domain_effect_results.py

echo "DONE"
