#!/bin/bash

# remember where we started
ROOT="$(pwd)"

# shellcheck source=/dev/null
source /tmp/torchinductor_venv/bin/activate
# shellcheck source=/dev/null
source ./.github/workflows/torch-inductor/scripts/common.sh

# Triton build-time dependencies
pip3 install --upgrade cmake ninja lit

# build our own triton and preserve the wheel build for later re-use in this test run.
cd python || exit
pip3 uninstall pytorch-triton -y
rm -rf build dist
python3 setup.py bdist_wheel
pip3 install dist/triton*.whl

# clean up cache
rm -rf ~/.triton/cache

# go back to where we started
cd "$ROOT" || exit
