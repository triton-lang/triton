"""
Invalid Memory Access Analysis
===============

In this tutorial, you will learn utlities to detect invalid memory accesses, including:
1. `triton.enable_invalid_memory_access_analysis` that catches exceptions on-the-fly and print the problematic line number.
2. `compute-sanitizer` as an external tool to instrument binaries and exhaustively capture invalid memory accesses.

"""

# %%
# Runtime invalid Memory Access Analysis
# --------------

import os

import torch

import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # This line will cause an invalid memory access error
    y = tl.load(y_ptr + offsets - 2147483647, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


# %%
# Before running the kernel, we disable PyTorch's caching allocator.
# Otherwise, PyTorch could allocate additional memory for tensors.
# We register the exception handling hook to print the exception stack trace.
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"

triton.enable_invalid_memory_access_analysis('./core_file')
torch.manual_seed(0)
size = 127
x = torch.rand(size, device='cuda')
y = torch.rand(size, device='cuda')
output_triton = add(x, y)

# %%
# Compute-Sanitizer Invalid Memory Access Analysis
# --------------
# Alternatively, you can try NVIDIA "compute-sanitizer" to instrument binaries and
# capture invalide memory accesses.
# See https://docs.nvidia.com/cuda/compute-sanitizer/index.html
# for more details.
# For example, you can turn off Triton's invalid memory analysis (Line 57)
# and run the following command to instrument the binary:
# $ compute-sanitizer --tool memcheck python ./09-invalid-memory-access-analysis.py
# Then, you are expected to see the following output:
# ```
# ========= Compute Sanitizer Output =========
# ========= CUDA-MEMCHECK
# ========= Invalid __global__ read of size 4 bytes
# =========     at 0x270 in /root/code/triton/python/tutorials/09-invalid-memory-access-analysis.py:37:add_kernel_0d1d2d3
# =========     by thread (30,0,0) in block (0,0,0)
# =========     Address 0x7f78f32003ec is out of bounds
# =========     and is 8589933588 bytes before the nearest allocation at 0x7f7af3200000 of size 508 bytes
# =========     Saved host backtrace up to driver entry point at kernel launch time
# ```
