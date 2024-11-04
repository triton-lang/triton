#!/bin/bash

MODEL_SPEC=$1

# torchinductor venv
whoami

sudo apt-get update && sudo apt-get install -y python3-venv libgl1

# clean up old venv
rm -rf /tmp/torchinductor_venv
python3 -m venv /tmp/torchinductor_venv
# shellcheck source=/dev/null
source /tmp/torchinductor_venv/bin/activate
# shellcheck source=/dev/null
source ./.github/workflows/torch-inductor/scripts/common.sh

pip3 install --upgrade pip wheel setuptools

# Install torchtext stable first. Bundling it in the same install as torch
# nightly forces torch stable release to be installed instead.
# From https://github.com/pytorch/text?tab=readme-ov-file#torchtext,
# "WARNING: TorchText development is stopped and the 0.18 release (April 2024)
# will be the last stable release of the library."
pip3 install --force-reinstall torchtext

# pytorch nightly
pip3 install --force-reinstall --pre torch torchvision torchaudio torchrec --extra-index-url https://download.pytorch.org/whl/nightly/cu121
# pytorch source to get torchbench for dynamo
pushd /tmp || exit
# cleanup old pytorch
rm -rf pytorch
git clone --recursive https://github.com/pytorch/pytorch
pushd pytorch || exit
# if you are updating an existing checkout
git submodule sync
git submodule update --init --recursive
popd || exit

# required packages
# https://github.com/pytorch/benchmark/blob/main/docker/gcp-a100-runner-dind.dockerfile#L17
sudo apt-get install --yes libpango-1.0-0 libpangoft2-1.0-0
pip3 install expecttest psutil lightning-utilities pyre_extensions

# torchbench
if [ "$MODEL_SPEC" == "torchbench" ] || [ "$MODEL_SPEC" != "all" ]; then
	# clean up old torchbench
	rm -rf benchmark
	pip3 install pyyaml
	git clone https://github.com/pytorch/benchmark.git
	pushd benchmark || exit
	python3 install.py
	popd || exit
fi

# timm
if [ "$MODEL_SPEC" == "timm_models" ] || [ "$MODEL_SPEC" != "all" ]; then
	# clean up old timm
	rm -rf pytorch-image-models
	git clone https://github.com/huggingface/pytorch-image-models.git
	pushd pytorch-image-models || exit
	pip3 install -e .
	popd || exit
fi

# clean up cache
rm -rf /tmp/torchinductor_"$(whoami)"/
rm -rf ~/.triton/cache
rm -rf "$TEST_REPORTS_DIR"

# go back to where we started
popd || exit
