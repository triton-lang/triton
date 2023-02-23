#!/bin/bash

# remember where we started
ROOT="$(pwd)"

# torchinductor venv 
python3 -m venv /opt/torchinductor_venv
# shellcheck source=/dev/null
source /opt/torchinductor_venv/bin/activate
# shellcheck source=/dev/null
source ./.github/workflows/scripts/common.sh

# pytorch nightly
cd "$PYTORCH_DIR"/.. || exit
pip3 install --force-reinstall --pre torch torchtext torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu118
# pytorch source to get torchbench for dynamo
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch || exit
# if you are updating an existing checkout
git submodule sync
git submodule update --init --recursive
cd ..

# expecttest
pip3 install expecttest

# torchbench
pip3 install pyyaml
git clone https://github.com/pytorch/benchmark.git
cd benchmark || exit
python install.py
cd ..

# timm
git clone https://github.com/huggingface/pytorch-image-models.git
cd pytorch-image-models || exit
pip install -e .
cd ..

# build our own triton
cd "$ROOT" || exit
cd python || exit
rm -rf build
pip3 install -e .
pip uninstall pytorch-triton -y

# clean up cache
rm -rf /tmp/torchinductor_root/
rm -rf ~/.triton/cache
rm -rf "$TEST_REPORTS_DIR"

# go back to where we started
cd "$ROOT" || exit
