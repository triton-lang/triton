#! /bin/bash

## $1: driver program
## $2: M
## $3: N
## $4: K
## $5: 1: reduced tuning space

if [[ $# -lt 4 ]];then
    echo "Usage: ./tritonProfiler.sh <driver program> M N K"
    exit
fi

DRIVER=$1
M=$2
N=$3
K=$4
reduceSpace=$5

DRIVER=$(echo $DRIVER | sed -e "s/matmul_grouped.py/matmul.py/g")

python $DRIVER -m $M -n $N -k $K
