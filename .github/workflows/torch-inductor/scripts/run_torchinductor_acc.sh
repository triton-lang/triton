#!/bin/bash

# remember where we started
ROOT="$(pwd)"
INDUCTOR="$ROOT"/.github/workflows/torch-inductor
MODEL_SPEC=$1

# shellcheck source=/dev/null
source /tmp/torchinductor_venv/bin/activate
# shellcheck source=/dev/null
source "$INDUCTOR"/scripts/common.sh

# Dependency of 'torch/fx/experimental/validator.py'.
pip3 install --upgrade z3-solver

# Install our own triton.
pip3 uninstall pytorch-triton -y
pushd python || exit
if [ -d "./dist" ]; then
  pip3 install dist/triton*.whl
else
  rm -rf build
  pip3 install -e .
fi

pushd "$PYTORCH_DIR" || exit
TEST_REPORTS_DIR=$TEST_REPORTS_DIR/acc
mkdir -p "$TEST_REPORTS_DIR"

for model in "${MODELS[@]}"; do
  if [ "$model" != "$MODEL_SPEC" ] && [ "$MODEL_SPEC" != "all" ]; then
    continue
  fi
  echo "Running accuracy test for $model"
  python3 benchmarks/dynamo/"$model".py --ci --accuracy --timing --explain --inductor --inference --device cuda \
    --output "$TEST_REPORTS_DIR"/inference_"$model".csv
  python3 benchmarks/dynamo/"$model".py --ci --accuracy --timing --explain --inductor --training --amp --device cuda \
    --output "$TEST_REPORTS_DIR"/training_"$model".csv
  python3 benchmarks/dynamo/"$model".py --ci --accuracy --timing --explain --inductor --training --dynamic-shapes --device cuda \
    --output "$TEST_REPORTS_DIR"/dynamic_shapes_"$model".csv
done

popd || exit
for model in "${MODELS[@]}"; do
  if [ "$model" != "$MODEL_SPEC" ] && [ "$MODEL_SPEC" != "all" ]; then
    continue
  fi
  echo "Checking accuracy test for $model"
  python3 "$INDUCTOR"/scripts/check_acc.py "$TEST_REPORTS_DIR"/inference_"$model".csv
  python3 "$INDUCTOR"/scripts/check_acc.py "$TEST_REPORTS_DIR"/training_"$model".csv
  python3 "$INDUCTOR"/scripts/check_acc.py "$TEST_REPORTS_DIR"/dynamic_shapes_"$model".csv
done

# go back to where we started
popd || exit
