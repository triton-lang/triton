#!/bin/bash

set -euxo pipefail

# export HIP_VISIBLE_DEVICES=2

unset AMD_INSERT_TTGIR

# export AMD_INSERT_TTGIR='matmul_kernel:./moe-cache-2-stages/matmul_kernel_modified.ttgir'
export AMD_INSERT_TTGIR='matmul_kernel:./moe-cache-1-stage/matmul_kernel_modified.ttgir'

rm -rf ~/.triton/cache/
python moe_kernel.py

