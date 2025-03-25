# Copyright (c) 2025, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.
# Licensed under the MIT License

import torch

import triton
import triton.language as tl
from triton.runtime.build import is_corex

import pytest

if not is_corex():
    float_dtypes = [torch.float16, torch.float32, torch.float64]
else:
    float_dtypes = [torch.float16, torch.float32]
dtypes = float_dtypes
dtypes_with_bfloat16 = dtypes + [torch.bfloat16]


def patch_kernel(template, to_replace):
    kernel = triton.JITFunction(template.fn)
    for key, value in to_replace.items():
        kernel.src = kernel.src.replace(key, value)
    return kernel


# ---------------
# test binary ops
# ---------------


@pytest.mark.parametrize("dtype_x, dtype_y, op",
                         [(torch.bfloat16, torch.bfloat16, op) for op in ['+', '-', '*', '/', '%']])
def test_bin_op(dtype_x, dtype_y, op, device='cuda'):
    SIZE = 128
    # define the kernel / launch-grid

    @triton.jit
    def kernel(Z, X, Y, SIZE: tl.constexpr):
        off = tl.arange(0, SIZE)
        x = tl.load(X + off)
        y = tl.load(Y + off)
        z = GENERATE_TEST_HERE
        tl.store(Z + off, z)

    x = torch.rand(SIZE, dtype=dtype_x, device=device) + 1.0
    y = torch.rand(SIZE, dtype=dtype_y, device=device) + 1.0

    expr = f' x {op} y'

    # reference result
    z_torch = eval(expr)

    # triton result
    kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': expr})
    z_tri = torch.empty(SIZE, dtype=z_torch.dtype, device=device)
    kernel[(1, )](z_tri, x, y, SIZE=SIZE, num_warps=4)
    torch.testing.assert_close(z_torch, z_tri, rtol=0.0, atol=0.0)


# ---------------
# test cast
# ---------------
@pytest.mark.parametrize("dtype_x, dtype_z", [
    (torch.float32, torch.bfloat16),
    (torch.bfloat16, torch.float32),
    (torch.int32, torch.bfloat16),
    (torch.bfloat16, torch.int32),
])
def test_cast(dtype_x, dtype_z, device='cuda'):
    SIZE = 1
    # triton kernel
    @triton.jit
    def kernel(X, Z, SIZE: tl.constexpr):
        x_ptr = X + tl.arange(0, SIZE)
        z_ptr = Z + tl.arange(0, SIZE)
        x = tl.load(x_ptr)
        z = x.to(Z.dtype.element_ty)
        tl.store(z_ptr, z)

    if dtype_x in [torch.int32]:
        x = torch.randint(low=0, high=10, size=(SIZE, ), dtype=dtype_x, device=device)
    elif dtype_x in [torch.bfloat16, torch.float32]:
        x = torch.rand(SIZE, dtype=dtype_x, device=device)
    # reference result
    z_torch = x.to(dtype_z)
    # triton result
    z_tri = torch.empty(SIZE, dtype=dtype_z, device=device)
    kernel[(1, )](x, z_tri, SIZE)
    torch.testing.assert_close(z_torch, z_tri, rtol=0.0, atol=0.0)
