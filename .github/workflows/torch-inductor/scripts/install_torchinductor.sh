#!/bin/bash

# remember where we started
ROOT="$(pwd)"
MODEL_SPEC=$1

# torchinductor venv
whoami
# clean up old venv
rm -rf /tmp/torchinductor_venv
python3 -m venv /tmp/torchinductor_venv
# shellcheck source=/dev/null
source /tmp/torchinductor_venv/bin/activate
# shellcheck source=/dev/null
source ./.github/workflows/torch-inductor/scripts/common.sh

# pytorch nightly
pip3 install --force-reinstall --pre torch torchtext torchvision torchaudio torchrec --extra-index-url https://download.pytorch.org/whl/nightly/cu121
# pytorch source to get torchbench for dynamo
cd /tmp || exit
# cleanup old pytorch
rm -rf pytorch
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch || exit
# if you are updating an existing checkout
git submodule sync
git submodule update --init --recursive
cd ..

# required packages
# https://github.com/pytorch/benchmark/blob/main/docker/gcp-a100-runner-dind.dockerfile#L17
sudo apt-get install --yes libpango-1.0-0 libpangoft2-1.0-0
pip3 install --upgrade pip
pip3 install expecttest psutil lightning-utilities pyre_extensions

# torchbench
if [ "$MODEL_SPEC" == "torchbench" ] || [ "$MODEL_SPEC" != "all" ]; then
	# clean up old torchbench
	rm -rf benchmark
	pip3 install pyyaml
	git clone https://github.com/pytorch/benchmark.git
	cd benchmark || exit
	python3 install.py
	cd ..
fi

# timm
if [ "$MODEL_SPEC" == "timm_models" ] || [ "$MODEL_SPEC" != "all" ]; then
	# clean up old timm
	rm -rf pytorch-image-models
	git clone https://github.com/huggingface/pytorch-image-models.git
	cd pytorch-image-models || exit
	pip3 install -e .
	cd ..
fi

# build our own triton
cd "$ROOT" || exit
cd python || exit
rm -rf build
pip3 install -e .
pip3 uninstall pytorch-triton -y

# clean up cache
rm -rf /tmp/torchinductor_root/
rm -rf ~/.triton/cache
rm -rf "$TEST_REPORTS_DIR"

# go back to where we started
cd "$ROOT" || exit
