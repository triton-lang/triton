#!/bin/bash

# remember where we started
ROOT="$(pwd)"

# shellcheck source=/dev/null
source ./.github/workflows/scripts/common.sh

# build our own triton
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
