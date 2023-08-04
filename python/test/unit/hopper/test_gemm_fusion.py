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

import torch

import triton
import triton.language as tl


@triton.jit
def gemm_fusion_kernel(A, B, C, E,
                       M, N, K,
                       stride_am, stride_ak, stride_bn, stride_bk, stride_cn, stride_ck, stride_em, stride_ek,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):

    offs_m = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    a_tile_ptr = tl.make_block_ptr(
        base=A, shape=(
            M, K), strides=(
            stride_am, stride_ak), offsets=(
                0, 0), block_shape=(
                    BLOCK_M, BLOCK_K), order=(
                        1, 0))
    b_tile_ptr = tl.make_block_ptr(
        base=B, shape=(
            N, K), strides=(
            stride_bn, stride_bk), offsets=(
                0, 0), block_shape=(
                    BLOCK_N, BLOCK_K), order=(
                        1, 0))
    c_tile_ptr = tl.make_block_ptr(
        base=C, shape=(
            N, K), strides=(
            stride_cn, stride_ck), offsets=(
                0, 0), block_shape=(
                    BLOCK_N, BLOCK_K), order=(
                        1, 0))

    a = tl.load(a_tile_ptr)
    b = tl.load(b_tile_ptr)

    o_ab = tl.dot(a, tl.trans(b))

    c = tl.load(c_tile_ptr)
    o_ab = o_ab.to(tl.float16)

    e = tl.dot(o_ab, c)
    e = e.to(tl.float16)

    offs_e = offs_m[:, None] * stride_em + offs_k[None, :] * stride_ek

    tl.store(E + offs_e, e)


def test_gemm_fusion():
    M, N, K = 128, 128, 64
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64
    A = torch.empty(
        (M, K), dtype=torch.float16, device='cuda').normal_(
        mean=0.1, std=0.2)
    B = torch.empty(
        (N, K), dtype=torch.float16, device='cuda').normal_(
        mean=0.1, std=0.2)
    C = torch.empty(
        (N, K), dtype=torch.float16, device='cuda').normal_(
        mean=0.1, std=0.2)
    C = torch.ones((N, K), dtype=torch.float16, device='cuda')
    E = torch.empty((M, K), dtype=torch.float16, device='cuda')
    ref_out = torch.matmul(torch.matmul(A, B.T), C)
    num_warps = 4
    gemm_fusion_kernel[(1, )](A, B, C, E, M, N, K,
                              A.stride(0), A.stride(1), B.stride(0), B.stride(
                                  1), C.stride(0), C.stride(1), E.stride(0), E.stride(1),
                              BLOCK_M, BLOCK_N, BLOCK_K, num_warps=num_warps)

    torch.testing.assert_allclose(ref_out, E, atol=1e-2, rtol=0)
