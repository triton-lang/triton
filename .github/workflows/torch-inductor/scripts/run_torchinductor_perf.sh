#!/bin/bash

# remember where we started
ROOT="$(pwd)"
INDUCTOR="$ROOT"/.github/workflows/torch-inductor
MODEL_SPEC=$1

# shellcheck source=/dev/null
source /tmp/torchinductor_venv/bin/activate
# shellcheck source=/dev/null
source "$INDUCTOR"/scripts/common.sh

# lock GPU clocks to 1350 MHz
sudo nvidia-smi -i 0 -pm 1
sudo nvidia-smi -i 0 --lock-gpu-clocks=1350,1350

pushd "$PYTORCH_DIR" || exit
TRITON_TEST_REPORTS_DIR=$TEST_REPORTS_DIR/perf
BASE_TEST_REPORTS_DIR=$TEST_REPORTS_DIR/acc
mkdir -p "$TRITON_TEST_REPORTS_DIR"
mkdir -p "$BASE_TEST_REPORTS_DIR"

# Dependency of 'pytorch/benchmarks/dynamo/common.py'.
pip3 install pandas scipy

echo "Running with Triton Nightly"
for model in "${MODELS[@]}"; do
  if [ "$model" != "$MODEL_SPEC" ] && [ "$MODEL_SPEC" != "all" ]; then
    continue
  fi
  echo "Running performance test for $model"
  python3 benchmarks/dynamo/"$model".py --ci --float32 --training --inductor --performance --device cuda \
    --output "$TRITON_TEST_REPORTS_DIR"/"$model".csv
done

# install pytorch-triton
pip3 uninstall triton -y
pip3 install --pre pytorch-triton --extra-index-url https://download.pytorch.org/whl/nightly/cu121

echo "Running with pytorch-triton"
for model in "${MODELS[@]}"; do
  if [ "$model" != "$MODEL_SPEC" ] && [ "$MODEL_SPEC" != "all" ]; then
    continue
  fi
  echo "Running performance test for $model"
  python3 benchmarks/dynamo/"$model".py --ci --float32 --training --inductor --performance --device cuda \
    --output "$BASE_TEST_REPORTS_DIR"/"$model".csv
done

# uninstall pytorch-triton
pip3 uninstall pytorch-triton -y

popd || exit
for model in "${MODELS[@]}"; do
  if [ "$model" != "$MODEL_SPEC" ] && [ "$MODEL_SPEC" != "all" ]; then
    continue
  fi
  echo "Checking performance test for $model"
  python3 "$INDUCTOR"/scripts/check_perf.py --new "$TRITON_TEST_REPORTS_DIR"/"$model".csv --baseline "$BASE_TEST_REPORTS_DIR"/"$model".csv
  EXIT_STATUS=$?
  if [ "$EXIT_STATUS" -ne 0 ]; then
    echo "Performance test for $model failed"
    exit "$EXIT_STATUS"
  fi
done

# unlock GPU clocks
sudo nvidia-smi -i 0 -rgc

# go back to where we started
popd || exit
