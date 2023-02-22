#!/bin/bash

TEST_REPORTS_DIR=/tmp/torchinductor_root/acc
PYTORCH_DIR=/tmp/pytorch
TRITON_DIR=/tmp/triton

cd "$PYTORCH_DIR" || exit
mkdir -p "$TEST_REPORTS_DIR"

MODELS=(timm huggingface torchbench)

for model in "${MODELS[@]}"; do
  echo "Running accuracy test for $model"
  python benchmarks/dynamo/"$model".py --ci --accuracy --timing --explain --inductor --device cuda \
    --output "$TEST_REPORTS_DIR"/inference_"$model".csv
  python benchmarks/dynamo/"$model".py --ci --accuracy --timing --explain --inductor --training --amp --device cuda \
    --output "$TEST_REPORTS_DIR"/training_"$model".csv
  python benchmarks/dynamo/"$model".py --ci --accuracy --timing --explain --inductor --dynamic-shapes --device cuda \
    --output "$TEST_REPORTS_DIR"/dynamic_shapes_"$model".csv
done

cd "$TRITON_DIR" || exit
for model in "${MODELS[@]}"; do
  echo "Checking accuracy test for $model"
  python .github/workflows/scripts/check_acc.py "$TEST_REPORTS_DIR"/inference_"$model".csv
  python .github/workflows/scripts/check_acc.py "$TEST_REPORTS_DIR"/training_"$model".csv
  python .github/workflows/scripts/check_acc.py "$TEST_REPORTS_DIR"/dynamic_shapes_"$model".csv
done
