#!/bin/bash

# remember where we started
ROOT="$(pwd)"
INDUCTOR="$ROOT"/.github/workflows/torchinductor

# shellcheck source=/dev/null
source /opt/torchinductor_venv/bin/activate
# shellcheck source=/dev/null
source "$INDUCTOR"/scripts/common.sh

# lock GPU clocks to 1350 MHz
sudo nvidia-smi -i 0 -pm 1
sudo nvidia-smi -i 0 --lock-gpu-clocks=1350,1350

cd "$PYTORCH_DIR" || exit
TEST_REPORTS_DIR=$TEST_REPORTS_DIR/perf
mkdir -p "$TEST_REPORTS_DIR"

for model in "${MODELS[@]}"; do
  echo "Running performance test for $model"
  python3 benchmarks/dynamo/"$model".py --ci --training --performance --disable-cudagraphs\
    --device cuda --inductor --amp --output "$TEST_REPORTS_DIR"/"$model".csv
done

cd "$ROOT" || exit
for model in "${MODELS[@]}"; do
  echo "Checking performance test for $model"
  python3 "$INDUCTOR"/scripts/check_perf.py --new "$TEST_REPORTS_DIR"/"$model".csv --baseline "$INDUCTOR"/data/"$model".csv
  EXIT_STATUS=$?
  if [ "$EXIT_STATUS" -ne 0 ]; then
    echo "Performance test for $model failed"
    exit "$EXIT_STATUS"
  fi
done

# unlock GPU clocks
sudo nvidia-smi -i 0 -rgc

# go back to where we started
cd "$ROOT" || exit
