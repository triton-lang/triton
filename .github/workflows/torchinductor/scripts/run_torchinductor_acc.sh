#!/bin/bash

# remember where we started
ROOT="$(pwd)"
INDUCTOR="$ROOT"/.github/workflows/torchinductor

# shellcheck source=/dev/null
source /opt/torchinductor_venv/bin/activate
# shellcheck source=/dev/null
source "$INDUCTOR"/scripts/common.sh

cd "$PYTORCH_DIR" || exit
TEST_REPORTS_DIR=$TEST_REPORTS_DIR/acc
mkdir -p "$TEST_REPORTS_DIR"

for model in "${MODELS[@]}"; do
  echo "Running accuracy test for $model"
  python3 benchmarks/dynamo/"$model".py --ci --accuracy --timing --explain --inductor --device cuda \
    --output "$TEST_REPORTS_DIR"/inference_"$model".csv
  python3 benchmarks/dynamo/"$model".py --ci --accuracy --timing --explain --inductor --training --amp --device cuda \
    --output "$TEST_REPORTS_DIR"/training_"$model".csv
  python3 benchmarks/dynamo/"$model".py --ci --accuracy --timing --explain --inductor --dynamic-shapes --device cuda \
    --output "$TEST_REPORTS_DIR"/dynamic_shapes_"$model".csv
done

cd "$ROOT" || exit
for model in "${MODELS[@]}"; do
  echo "Checking accuracy test for $model"
  python3 "$INDUCTOR"/scripts/check_acc.py "$TEST_REPORTS_DIR"/inference_"$model".csv
  python3 "$INDUCTOR"/scripts/check_acc.py "$TEST_REPORTS_DIR"/training_"$model".csv
  python3 "$INDUCTOR"/scripts/check_acc.py "$TEST_REPORTS_DIR"/dynamic_shapes_"$model".csv
done

# go back to where we started
cd "$ROOT" || exit
