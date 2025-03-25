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

import pytest
import torch
from torch.testing import assert_close

import triton
import triton.language as tl


@triton.jit
def matmul_tma_load_store(  #
        a_ptr, b_ptr, c_ptr,  #
        M, N, K,  #
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,  #
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,  #
        OUTPUT_F16: tl.constexpr  #
):
    a_block_ptr = tl.make_block_ptr(base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak), offsets=(0, 0),
                                    block_shape=(BLOCK_M, BLOCK_K), order=(1, 0))
    b_block_ptr = tl.make_block_ptr(base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn), offsets=(0, 0),
                                    block_shape=(BLOCK_K, BLOCK_N), order=(0, 1))
    c_block_ptr = tl.make_block_ptr(base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn), offsets=(0, 0),
                                    block_shape=(BLOCK_M, BLOCK_N), order=(1, 0))
    a = tl.load(a_block_ptr)
    b = tl.load(b_block_ptr)

    c = tl.dot(a, b)
    if OUTPUT_F16:
        c = c.to(tl.float16)

    tl.store(c_block_ptr, c)


@pytest.mark.parametrize('M,N,K,NUM_CTAS,NUM_WARPS,TRANS_A,TRANS_B,OUTPUT_F16', [
    [64, 64, 16, 1, 4, False, True, False],
    [64, 64, 16, 1, 4, False, True, True],
    [128, 64, 32, 1, 4, False, True, False],
    [128, 64, 32, 1, 4, False, True, True],
    [64, 128, 32, 1, 4, False, True, False],
    [64, 128, 32, 1, 4, False, True, True],
    [128, 128, 64, 1, 4, False, True, False],
    [128, 128, 64, 1, 4, False, True, True],
])
def test_tma_load_store(M, N, K, NUM_CTAS, NUM_WARPS, TRANS_A, TRANS_B, OUTPUT_F16):
    if (TRANS_A):
        a = torch.randn((K, M), device='cuda', dtype=torch.float16).T
    else:
        a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    if (TRANS_B):
        b = torch.randn((N, K), device='cuda', dtype=torch.float16).T
    else:
        b = torch.randn((K, N), device='cuda', dtype=torch.float16)

    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    if OUTPUT_F16:
        c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    matmul_tma_load_store[(1, 1)](
        a_ptr=a, b_ptr=b, c_ptr=c,  #
        M=M, N=N, K=K,  #
        stride_am=a.stride(0), stride_ak=a.stride(1),  #
        stride_bk=b.stride(0), stride_bn=b.stride(1),  #
        stride_cm=c.stride(0), stride_cn=c.stride(1),  #
        BLOCK_M=M, BLOCK_N=N, BLOCK_K=K,  #
        num_warps=NUM_WARPS, num_ctas=NUM_CTAS,  #
        OUTPUT_F16=OUTPUT_F16)
    golden = torch.matmul(a, b)
    torch.set_printoptions(profile="full")
    assert_close(c, golden, rtol=1e-2, atol=1e-3, check_dtype=False)
