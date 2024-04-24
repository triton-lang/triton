#!/bin/bash

TEST_REPORTS_DIR=/tmp/torchinductor_reports
PYTORCH_DIR=/tmp/pytorch
MODELS=(timm_models huggingface torchbench)

echo "$TEST_REPORTS_DIR"
echo "$PYTORCH_DIR"
echo "${MODELS[@]}"
