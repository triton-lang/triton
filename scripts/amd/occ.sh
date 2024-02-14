#! /bin/bash

## $1: input script that contains one kernel

rm -rf ~/.triton/cache/

export MLIR_ENABLE_DUMP=1
export LLVM_IR_ENABLE_DUMP=1
export AMDGCN_ENABLE_DUMP=1
## Assume CDNA arch
SIMD=4
LDS_SIZE=65536
TOTAL_VGPR=512

$1 > output.mlir 2>&1

LDS_line=$(sed -n '/triton_gpu\.shared\ /p' output.mlir | tail -n 1 | grep -o 'triton_gpu.shared = [0-9]*')
numWarps_line=$(sed -n '/triton_gpu\.num-warps/p' output.mlir | tail -n 1 | grep -o 'triton_gpu.num-warps. = [0-9]*')

LDS=${LDS_line##*=}
num_warps=${numWarps_line##*=}
echo "LDS: $LDS, num_warps: $num_warps"

VGPRs=$(sed -n '/vgpr_count/p' output.mlir | tail -n 1 | awk '{print $2}')
SPILLs=$(sed -n '/vgpr_spill/p' output.mlir | tail -n 1 | awk '{print $2}')

echo "VGPRS: $VGPRs (spill: $SPILLs)"

occ_LDS=$((LDS_SIZE/LDS*num_warps/SIMD))
occ_vgpr=$((TOTAL_VGPR/VGPRs))
occ=$occ_vgpr
if [ $occ_LDS -lt $occ_vgpr ];then
    occ=$occ_LDS
fi
echo "occ: $occ waves/SIMD (occ_LDS: $occ_LDS, occ_vgpr: $occ_vgpr)"

perf=$(tail -n 2 output.mlir)
echo "$perf"

## remove distracting info from the assembly
sed -i '/\.loc/d' output.mlir
sed -i '/\.Ltmp.*:/d' output.mlir
sed -i '/AMD clang version/d' output.mlir
