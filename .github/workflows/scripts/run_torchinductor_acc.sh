#!/bin/bash

# remember where we started
ROOT="$(pwd)"

# shellcheck source=/dev/null
source ./common.sh

cd "$PYTORCH_DIR" || exit
TEST_REPORTS_DIR=$TEST_REPORTS_DIR/acc
mkdir -p "$TEST_REPORTS_DIR"

for model in "${MODELS[@]}"; do
  echo "Running performance test for $model"
  python benchmarks/dynamo/"$model".py --ci --training --performance --disable-cudagraphs\
    --device cuda --inductor --amp --output "$TEST_REPORTS_DIR"/"$model".csv
done

cd "$ROOT" || exit
for model in "${MODELS[@]}"; do
  echo "Checking performance test for $model"
  python .github/workflows/scripts/check_perf.py "$TEST_REPORTS_DIR"/"$model".csv
done

# go back to where we started
cd "$ROOT" || exit
