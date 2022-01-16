import pytest
import torch
import triton.language as tl
from triton.code_gen import next_power_of_2

import triton


@triton.heuristics({"block_len": lambda args: next_power_of_2(args["N"])})
@triton.jit
def _normalized_op(inputs_ptr, normalized_out_ptr, N, eps, block_len: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, block_len)
    inputs = inputs_ptr + row * N + cols
    inputs = tl.load(inputs, mask=cols < N, other=0)
    denom = tl.sum(inputs.to(tl.float32), 0) + eps
    normalized_out = normalized_out_ptr + row * N + cols
    tl.store(normalized_out, inputs / denom, mask=cols < N)


def normalized(inputs, eps):
    n_rows, n_cols = inputs.shape
    grid = lambda opt: (n_rows,)
    normalized = torch.empty_like(inputs)
    _normalized_op[grid](inputs, normalized, n_cols, eps)
    return normalized


def torch_normalized(inputs, eps):
    return inputs / (torch.sum(inputs, axis=1) + eps)[:, None]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_normalized(dtype):
    eps = 1e-3
    inputs = torch.ones((4, 23), dtype=torch.bfloat16, device="cuda")
    triton_result = normalized(inputs, eps=eps)
    torch_result = torch_normalized(inputs, eps=eps)
    triton.testing.assert_almost_equal(triton_result, torch_result)
