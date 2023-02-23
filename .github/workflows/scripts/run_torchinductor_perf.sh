#!/bin/bash

# remember where we started
ROOT="$(pwd)"

# shellcheck source=/dev/null
source ./common.sh

cd "$PYTORCH_DIR" || exit
TEST_REPORTS_DIR=$TEST_REPORTS_DIR/perf
mkdir -p "$TEST_REPORTS_DIR"

for model in "${MODELS[@]}"; do
  echo "Running performance test for $model"
  python benchmarks/dynamo/"$model".py --ci --accuracy --timing --explain --inductor --device cuda \
    --output "$TEST_REPORTS_DIR"/inference_"$model".csv
  python benchmarks/dynamo/"$model".py --ci --accuracy --timing --explain --inductor --training --amp --device cuda \
    --output "$TEST_REPORTS_DIR"/training_"$model".csv
  python benchmarks/dynamo/"$model".py --ci --accuracy --timing --explain --inductor --dynamic-shapes --device cuda \
    --output "$TEST_REPORTS_DIR"/dynamic_shapes_"$model".csv
done

cd "$ROOT" || exit
for model in "${MODELS[@]}"; do
  echo "Checking performance test for $model"
  python .github/workflows/scripts/check_perf.py --new "$TEST_REPORTS_DIR"/inference_"$model".csv --baseline .github/workflows/scripts/baseline_inference_"$model".csv
  python .github/workflows/scripts/check_perf.py --new "$TEST_REPORTS_DIR"/training_"$model".csv --baseline .github/workflows/scripts/baseline_training_"$model".csv
  python .github/workflows/scripts/check_perf.py --new "$TEST_REPORTS_DIR"/dynamic_shapes_"$model".csv --baseline .github/workflows/scripts/baseline_dynamic_shapes_"$model".csv
done

# go back to where we started
cd "$ROOT" || exit
