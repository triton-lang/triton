#!/bin/bash

TEST_REPORTS_DIR=/opt/torchinductor_reports
PYTORCH_DIR=/opt/pytorch
MODELS=(timm_models huggingface torchbench)

echo "$TEST_REPORTS_DIR"
echo "$PYTORCH_DIR"
echo "${MODELS[@]}"
