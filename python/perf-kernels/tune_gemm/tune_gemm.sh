#! /bin/bash

## $1: driver program
## $2: M
## $3: N
## $4: K
## $5: 1: reduced tuning space

if [[ $# -lt 4 ]];then
    echo "Usage: ./tune_gemm.sh <driver program> M N K"
    exit
fi

DRIVER=$1
M=$2
N=$3
K=$4
reduceSpace=$5

DRIVER=$(echo $DRIVER | sed -e "s/matmul_grouped.py/matmul.py/g")

# $DRIVER is the actual tuning scripts, it is the file matmul.py
# -mnk are the size of input matrices, matrix (m, k) x (k, n)
# --specify_size means using -mnk to specify size of input matrices
# --rocprof means using rocprof to measure kernel time. If not set,
# kernel time is from do_bench()
python $DRIVER -m $M -n $N -k $K --specify_size --rocprof
