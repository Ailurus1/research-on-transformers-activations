#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAX_SAMPLES="${MAX_SAMPLES-}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
DOMAIN="${DOMAIN-}"
SKIP_AGGREGATION="${SKIP_AGGREGATION:-0}"
HF_TMP_CACHE="$(mktemp -d -t hf-size-effect-cache-XXXXXX)"
export HF_HOME="${HF_TMP_CACHE}"
export HF_HUB_CACHE="${HF_TMP_CACHE}/hub"
export HF_DATASETS_CACHE="${HF_TMP_CACHE}/datasets"
export TRANSFORMERS_CACHE="${HF_TMP_CACHE}/transformers"
trap 'rm -rf "${HF_TMP_CACHE}"' EXIT

DOMAINS=(
  "masked-language-modeling"
  "text-generation"
  "image-classification"
  "automatic-speech-recognition"
)

if [[ -n "${DOMAIN}" ]]; then
  DOMAINS=( "${DOMAIN}" )
fi

for domain in "${DOMAINS[@]}"; do
  case "${domain}" in
    "masked-language-modeling"|"text-generation"|"image-classification"|"automatic-speech-recognition")
      ;;
    *)
      echo "Unsupported DOMAIN: ${domain}" >&2
      exit 1
      ;;
  esac
  echo "Running size-effect domain: ${domain}"
  EXTRA_ARGS=()
  if [[ -n "${MAX_SAMPLES}" ]]; then
    EXTRA_ARGS+=( --max-samples "${MAX_SAMPLES}" )
  fi
  if [[ "${SKIP_AGGREGATION}" == "1" ]]; then
    EXTRA_ARGS+=( --skip-aggregation )
  fi
  python "${ROOT_DIR}/experiments/explore_size_effect.py" \
    "${EXTRA_ARGS[@]}" \
    --batch-size "${BATCH_SIZE}" \
    --log-level "${LOG_LEVEL}" \
    --tasks "${domain}"
done

echo "DONE"
