#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ -z "${CHRONOSPHERE_API_KEY:-}" ]]; then
  echo "CHRONOSPHERE_API_KEY is not set. Export it before running analysis." >&2
fi

export CHRONO_PROM_BASE_URL="${CHRONO_PROM_BASE_URL:-https://openai.chronosphere.io/data/metrics}"

exec node server.mjs

