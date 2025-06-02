import torch
import pathlib
import pytest
import json
import os

import triton
import triton.language as tl
import triton.profiler.language as pl
import triton.profiler as proton

from typing import NamedTuple

dir_path = os.path.dirname(os.path.realpath(__file__))

def metadata_fn(grid: tuple, metadata: NamedTuple, args: dict):
    BLOCK_SIZE = args["BLOCK_SIZE"]
    return {"name": f"add_{BLOCK_SIZE}"}
@triton.jit(launch_metadata=metadata_fn)
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
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
torch.manual_seed(0)
size = 256
x = torch.rand(size, device='cuda')
y = torch.rand(size, device='cuda')
tmp_path = pathlib.Path(dir_path + '/tmp')
temp_file = tmp_path / "test_tree.hatchet"
output = torch.empty_like(x)
n_elements = output.numel()
grid = (1, 1, 1)
proton.start(str(temp_file.with_suffix("")), backend="instrumentation", hook="hook")
add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024, num_warps=1)
proton.finalize()

# with open(temp_file, "rb") as f:
#     data = json.load(f)
#     if hook:
#         assert "add_1024" == data[0]["children"][0]["frame"]["name"]
#     kernel_frame = data[0]["children"][0]["children"][0]
#     load_ops = kernel_frame["children"][0]
#     assert "load_ops" in load_ops["frame"]["name"]
#     assert ("load_x" in load_ops["children"][0]["frame"]["name"]
#             or "load_x" in load_ops["children"][1]["frame"]["name"])
#     assert ("load_y" in load_ops["children"][0]["frame"]["name"]
#             or "load_y" in load_ops["children"][1]["frame"]["name"])
#     assert load_ops["children"][0]["metrics"]["cycles"] > 0
#     assert load_ops["children"][1]["metrics"]["cycles"] > 0
