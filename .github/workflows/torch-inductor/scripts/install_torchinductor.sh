#!/bin/bash

# remember where we started
ROOT="$(pwd)"
MODEL_SPEC=$1

# torchinductor venv
whoami
# clean up old venv
rm -rf /opt/torchinductor_venv
python3 -m venv /opt/torchinductor_venv
# shellcheck source=/dev/null
source /opt/torchinductor_venv/bin/activate
# shellcheck source=/dev/null
source ./.github/workflows/torch-inductor/scripts/common.sh

# pytorch nightly
pip3 install --force-reinstall --pre torch torchtext torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu121
# pytorch source to get torchbench for dynamo
cd /opt || exit
# cleanup old pytorch
rm -rf pytorch
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch || exit
# if you are updating an existing checkout
git submodule sync
git submodule update --init --recursive
cd ..

# required packages
pip3 install expecttest psutil

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
