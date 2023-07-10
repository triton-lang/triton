"""
Exception Analysis
===============

In this tutorial, you will write a simple vector addition using Triton.

"""

# %%
# Compute Kernel
# --------------

import torch

import triton
import triton.language as tl
import os


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
    # This line will cause an out-of-bounds error
    y = tl.load(y_ptr + offsets - 3276800, mask=mask)
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

triton.enable_exception_analysis('./core_file')
torch.manual_seed(0)
size = 127
x = torch.rand(size, device='cuda')
y = torch.rand(size, device='cuda')
output_triton = add(x, y)
