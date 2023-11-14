# AMD Perf Kernels

This directory contains customized/tuned/experimental kernels on AMD MI series GPUs.

## `06-fused-attention-transV.py`

This script is a copy of `tutorials/06-fused-attention.py` with the following
two changes:

- Tensor V is transposed in the way that seqlen/N_CTX dimension becomes the
fastest changing (a.k.a. leading or least strided) dimension.
This script produces better performance than `tutorials/06-fused-attention.py`
since it has better LDS access efficiency for tensor V.
Note that in the future, we'll improve the LDS access efficiency for
non-transposed tensor V, i.e. head dimension is the fastest changing dimension.
- Only fwd kernel is benchmarked.

## `06-fused-attention-fwd-transV.py`

This script is used to produce the best performance for fwd kernel.
It is a copy of `06-fused-attention-transV.py` with the following
changes:

- All bwd kernels are removed.
- Storing `m` at the end of the fwd kernel is removed.
- Autotuner is removed. All parameters for D=64 ad D=128 are pre-tuned
on MI250X and hard coded.

Note that this script is also used to benchmark FA performance with 2 GCDs.
Check the [2GCD benchmark script](https://github.com/ROCmSoftwarePlatform/triton/blob/triton-mlir/scripts/amd/benchmark_flash_attention.py) for more details.
