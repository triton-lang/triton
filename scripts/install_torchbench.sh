#!/bin/bash

# pytorch nightly
pip3 install --force-reinstall --pre torch torchtext torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu116

# torchbench
pip3 install pyyaml
git clone https://github.com/pytorch/benchmark.git
cd benchmark || exit
python install.py

# build our own triton
cd ../python || exit
rm -rf build
pip3 install -e .
rm -rf /tmp/torchinductor_root/
rm -rf ~/.triton/cache
pip uninstall pytorch-triton