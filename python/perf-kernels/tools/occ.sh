#! /bin/bash

## $1: input script that contains one kernel

rm -rf ~/.triton/cache/

export MLIR_ENABLE_DUMP=1
export AMDGCN_ENABLE_DUMP=1
## Assume CDNA arch
SIMD=4
LDS_SIZE=65536
TOTAL_VGPR=512

get_occ_per_CU() {
    ## $1: vgpr count
    vgpr=$1
    occPerEU=$((TOTAL_VGPR/vgpr))
    if [[ $vgpr -gt 256 ]]; then
        occPerEU=1
    elif [[ $vgpr -gt 168 ]]; then
        occPerEU=2
    elif [[ $vgpr -gt 128 ]]; then
        occPerEU=3
    elif [[ $vgpr -gt 96 ]]; then
        occPerEU=4
    elif [[ $vgpr -gt 80 ]]; then
        occPerEU=5
    elif [[ $vgpr -gt 72 ]]; then
        occPerEU=6
    elif [[ $vgpr -gt 64 ]]; then
        occPerEU=7
    else
        occPerEU=8
    fi

    occPerCU=$((occPerEU*SIMD/num_warps))
    echo $occPerCU
}

$1 > output.mlir 2>&1

LDS_line=$(sed -n '/triton_gpu\.shared\ /p' output.mlir | tail -n 1 | grep -o 'triton_gpu.shared = [0-9]*')
numWarps_line=$(sed -n '/triton_gpu\.num-warps/p' output.mlir | tail -n 1 | grep -o 'triton_gpu.num-warps. = [0-9]*')

LDS=${LDS_line##*=}
num_warps=${numWarps_line##*=}
echo "LDS: $LDS, num_warps: $num_warps"

VGPRs=$(sed -n '/vgpr_count/p' output.mlir | tail -n 1 | awk '{print $2}')
SPILLs=$(sed -n '/vgpr_spill/p' output.mlir | tail -n 1 | awk '{print $2}')

echo "VGPRS: $VGPRs (spill: $SPILLs)"

occLDSPerCU=$((LDS_SIZE/LDS))
occVgprPerCU=$(get_occ_per_CU $VGPRs)
occPerCU=$occVgprPerCU
if [ $occLDSPerCU -lt $occVgprPerCU ];then
    occPerCU=$occLDSPerCU
fi
occPerEU=$((occPerCU*num_warps/SIMD))
echo "occupancy: $occPerEU waves/SIMD or $occPerCU workgroups/CU (occLDSPerCU: $occLDSPerCU, occVgprPerCU: $occVgprPerCU)"

perf=$(tail -n 2 output.mlir)
echo "$perf"

## remove distracting info from the assembly
sed -i '/local_/! {/\.loc/d}' output.mlir
sed -i '/\.Ltmp.*:/d' output.mlir
sed -i '/AMD clang version/d' output.mlir

sed -n '/AMDGCN/, $p' output.mlir > output.amdgcn
