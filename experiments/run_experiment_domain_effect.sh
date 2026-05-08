#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${ROOT_DIR}/outputs/domain_effect"
SUMMARY_CSV="${ROOT_DIR}/experiment_domain_effect.csv"

MAX_SAMPLES="${MAX_SAMPLES:-32}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

DOMAINS=(
  "text-classification"
  "machine-translation"
  "text-generation"
  "image-classification"
  "automatic-speech-recognition"
)

for domain in "${DOMAINS[@]}"; do
  echo "Domain: ${domain}"
  python "${ROOT_DIR}/experiments/explore_domain_effect.py" \
    --max-samples "${MAX_SAMPLES}" \
    --batch-size 1 \
    --log-level "${LOG_LEVEL}" \
    --tasks "${domain}"
done

python3 aggregate_domain_effect_results.py

echo "DONE"
