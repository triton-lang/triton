#!/bin/bash

# remember where we started
ROOT="$(pwd)"

# shellcheck source=/dev/null
source /opt/torchinductor_venv/bin/activate
# shellcheck source=/dev/null
source ./.github/workflows/torchinductor/scripts/common.sh

# build our own triton
cd python || exit
pip3 install --pre pytorch-triton --extra-index-url https://download.pytorch.org/whl/nightly/cu118
rm -rf build
pip3 install -e .
pip3 uninstall pytorch-triton -y

# clean up cache
rm -rf /tmp/torchinductor_root/
rm -rf ~/.triton/cache
rm -rf "$TEST_REPORTS_DIR"

# go back to where we started
cd "$ROOT" || exit
