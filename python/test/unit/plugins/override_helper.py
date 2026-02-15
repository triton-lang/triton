import torch

import triton
import triton.language as tl
from triton import knobs
import sys

from typing import NamedTuple
import custom_stages

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def metadata_fn(grid: tuple, metadata: NamedTuple, args: dict):
    BLOCK_SIZE = args["BLOCK_SIZE"]
    return {"name": f"add_{BLOCK_SIZE}"}


@triton.jit(launch_metadata=metadata_fn)
def add_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor, path):
    output = torch.empty_like(x)
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    knobs.runtime.add_stages_inspection_hook = custom_stages.inspect_stages_hook_dialect
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024, num_warps=1)
    return output


size = 98432
x = torch.rand(size, device=DEVICE)
y = torch.rand(size, device=DEVICE)
output_torch = x + y
output_triton = add(x, y, sys.argv[-1])
