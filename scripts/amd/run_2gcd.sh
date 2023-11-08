#! /bin/bash


## A simple script to run two flash attention kernels
## with batch2-nheads48-d64 on two GPUs in parallel
## $1: mode, fwd or bwd

if [[ $# -eq 0 ]];then
    echo "Must specify mode, fwd or bwd"
    exit
fi

TRITON_DIR=$(git rev-parse --show-toplevel)

BENCHMARK_DRIVER=${TRITON_DIR}/scripts/amd/benchmark_flash_attention.py

bs=2
nheads=48
mode=$1

declare -A repA

if [[ $mode == "fwd" ]];then
    repA[1024]=160000
    repA[2048]=80000
    repA[4096]=40000
    repA[8192]=20000
    repA[16384]=10000
else
    repA[1024]=10000
    repA[2048]=10000
    repA[4096]=2500
    repA[8192]=600
    repA[16384]=100
fi

for d in 128 64
do
    echo "Benchmarking FA $mode kernel with D = $d on 2 GCDs"
    for seqlen in 1024 2048  4096 8192 16384
    do
        rep=${repA[$seqlen]}
        args="-bs $bs -nheads $nheads -d $d -seqlen $seqlen -mode $mode"

        ## pre-compile the kernel
        python ${BENCHMARK_DRIVER} $args -rep 1

        start_time=$(date +%s.%3N)
        export ROCR_VISIBLE_DEVICES=0
        python ${BENCHMARK_DRIVER} $args -rep $rep &

        export ROCR_VISIBLE_DEVICES=1
        python ${BENCHMARK_DRIVER} $args -rep $rep

        wait
        end_time=$(date +%s.%3N)

        # elapsed time with millisecond resolution
        # keep three digits after floating point.
        elapsed=$(echo "scale=3; $end_time - $start_time" | bc)
        # Convert second to tflops
        if [[ $mode == "fwd" ]];then
            tflops=$(echo "scale=2; 8*$seqlen*$seqlen*$bs*$nheads*$d*$rep/$elapsed/1000000000000" | bc)
        else
            tflops=$(echo "scale=2; 7*4*0.5*$seqlen*$seqlen*$bs*$nheads*$d*$rep/$elapsed/1000000000000" | bc)
        fi
        echo "$seqlen  $tflops tflops $elapsed s"

    done
done
