#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

VERSIONS="${VERSIONS:-3.5.1 3.6.0}"
SHAPES="${SHAPES:-32x128}"
DTYPES="${DTYPES:-bf16}"
WARPS="${WARPS:-4}"
M="${M:-8192}"
N="${N:-8192}"
REPEAT="${REPEAT:-5}"
WARMUP="${WARMUP:-5}"
GPU_LABEL="${GPU_LABEL:-gpu}"
PYTHON_BIN="${PYTHON_BIN:-}"
RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results}"

if [ -z "${PYTHON_BIN}" ]; then
  if [ -x /opt/venv/bin/python ]; then
    PYTHON_BIN=/opt/venv/bin/python
  else
    PYTHON_BIN=python3
  fi
fi

mkdir -p "${RESULTS_DIR}"

echo "python=${PYTHON_BIN}"
"${PYTHON_BIN}" - <<'PY'
import torch
print("torch", torch.__version__)
PY

for version in ${VERSIONS}; do
  "${PYTHON_BIN}" -m pip install -q --disable-pip-version-check --no-cache-dir --no-deps --force-reinstall \
    "triton==${version}"

  out="${RESULTS_DIR}/${GPU_LABEL}_triton_${version}_$(date -u +%Y%m%dT%H%M%SZ).jsonl"
  echo "running Triton ${version}; output=${out}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/bench_desc_load_reduce.py" \
    --m "${M}" \
    --n "${N}" \
    --shapes "${SHAPES}" \
    --dtypes "${DTYPES}" \
    --warps "${WARPS}" \
    --warmup "${WARMUP}" \
    --repeat "${REPEAT}" \
    --check | tee "${out}"
done
