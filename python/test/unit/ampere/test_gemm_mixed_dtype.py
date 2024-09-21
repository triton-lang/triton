# Copyright (c) 2023 NVIDIA Corporation & Affiliates. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import itertools

import pytest
import torch
from torch.testing import assert_close

import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    lhs_ptr,  # (M, K)
    rhs_ptr,  # (K, N)
    out_ptr,  # (M, N)
    # shape information (strides)
    M, N, K,
    # block information
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr
):
    start_m = tl.program_id(0)  # start (axis m)
    start_n = tl.program_id(1)  # start (axis n)

    acc = tl.zeros([block_m, block_n], dtype=tl.float32)

    for start_k in range(0, K, block_k):
        lhs_tile = (start_m * block_m + tl.arange(0, block_m))[:, None] * K + \
            (start_k + tl.arange(0, block_k))[None, :]

        rhs_tile = (start_k + tl.arange(0, block_k))[:, None] * N + \
            (start_n * block_n + tl.arange(0, block_n))[None, :]

        lhs_mask = ((start_m * block_m + tl.arange(0, block_m)) < M)[:, None] \
            * ((start_k + tl.arange(0, block_k)) < K)[None, :]

        rhs_mask = ((start_n * block_n + tl.arange(0, block_n)) < N)[None, :] \
            * ((start_k + tl.arange(0, block_k)) < K)[:, None]

        lhs = tl.load(lhs_ptr + lhs_tile, mask=lhs_mask, other=0.0)
        rhs = tl.load(rhs_ptr + rhs_tile, mask=rhs_mask, other=0.0)

        acc = tl.dot(lhs, rhs.to(tl.float16), acc)

    out_tile = (start_m * block_m + tl.arange(0, block_m))[:, None] * N + \
        start_n * block_n + tl.arange(0, block_n)[None, :]

    mask = ((start_m * block_m + tl.arange(0, block_m)) < M)[:, None] & \
        ((start_n * block_n + tl.arange(0, block_n)) < N)[None, :]

    # print("acc ", acc)

    tl.store(out_ptr + out_tile, acc, mask=mask)


@pytest.mark.parametrize(
    'M,N,K,block_m,block_n,block_k,dtype_a,dtype_b',
    [
        [128, 128, 128, 16, 16, 16, torch.float16, torch.int8],
        [128, 128, 128, 32, 16, 16, torch.float16, torch.int8],
        [128, 128, 128, 16, 32, 16, torch.float16, torch.int8],
        [128, 128, 128, 32, 32, 16, torch.float16, torch.int8],
        [128, 128, 128, 16, 64, 16, torch.float16, torch.int8],
        [128, 128, 128, 64, 16, 16, torch.float16, torch.int8]
    ])
@pytest.mark.skipif(torch.cuda.get_device_capability()[0] != 8, reason="Requires compute capability == 8")
def test_gemm_mixed_dtype(M, N, K, block_m, block_n, block_k, dtype_a, dtype_b):
    lhs = torch.randn((M, K), dtype=dtype_a, device="cuda")
    rhs = torch.randint(0, 127, (K, N), dtype=dtype_b, device="cuda")
    out = torch.empty((M, N), dtype=dtype_a, device='cuda')
    print(M, N, K, block_m, block_n, block_k)
    matmul_kernel[(triton.cdiv(M, block_m),
                   triton.cdiv(N, block_n))](lhs, rhs, out,
                                             M=M, N=N, K=K,
                                             block_m=block_m,
                                             block_n=block_n,
                                             block_k=block_k)
    expected = torch.mm(lhs, rhs.to(dtype_a))
    assert_close(out.cpu(), expected.cpu())
