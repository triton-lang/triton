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

import triton
import triton.language as tl


@triton.jit
def gemm_fusion_kernel(A, B, C, E,  #
                       M, N, K,  #
                       stride_am, stride_ak, stride_bn, stride_bk, stride_cn, stride_ck, stride_em, stride_ek,  #
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)

    a_tile_ptr = tl.make_block_ptr(base=A, shape=(M, K), strides=(stride_am, stride_ak), offsets=(pid * BLOCK_M, 0),
                                   block_shape=(BLOCK_M, BLOCK_K), order=(1, 0))
    b_tile_ptr = tl.make_block_ptr(base=B, shape=(N, K), strides=(stride_bn, stride_bk), offsets=(0, 0),
                                   block_shape=(BLOCK_N, BLOCK_K), order=(1, 0))
    c_tile_ptr = tl.make_block_ptr(base=C, shape=(N, K), strides=(stride_cn, stride_ck), offsets=(0, 0),
                                   block_shape=(BLOCK_N, BLOCK_K), order=(1, 0))
    e_tile_ptr = tl.make_block_ptr(base=E, shape=(M, K), strides=(stride_em, stride_ek), offsets=(pid * BLOCK_M, 0),
                                   block_shape=(BLOCK_M, BLOCK_K), order=(1, 0))

    acc_e = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    a = tl.load(a_tile_ptr)
    for i in range(0, N, BLOCK_N):
        b = tl.load(b_tile_ptr)
        o_ab = tl.dot(a, tl.trans(b))
        c = tl.load(c_tile_ptr)
        o_ab = o_ab.to(tl.float16)
        acc_e += tl.dot(o_ab, c)
        b_tile_ptr = tl.advance(b_tile_ptr, [BLOCK_N, 0])
        c_tile_ptr = tl.advance(c_tile_ptr, [BLOCK_N, 0])

    acc_e = acc_e.to(tl.float16)
    tl.store(e_tile_ptr, acc_e)


#TODO: blackwell mma pipeline regressed with https://github.com/openai/triton-private-blackwell/commit/5cc42002bc36fb94385481ed4dab178a143733be
@pytest.mark.skipif(torch.cuda.get_device_capability()[0] != 9, reason="only works on hopper")
def test_gemm_fusion():
    M, N, K = 4096, 4096, 64
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64
    A = torch.empty((M, K), dtype=torch.float16, device='cuda').normal_(mean=0.1, std=0.2)
    B = torch.empty((N, K), dtype=torch.float16, device='cuda').normal_(mean=0.1, std=0.2)
    C = torch.empty((N, K), dtype=torch.float16, device='cuda').normal_(mean=0.1, std=0.2)
    E = torch.empty((M, K), dtype=torch.float16, device='cuda')
    ref_out = torch.matmul(torch.matmul(A, B.T), C)
    num_warps = 4
    grid = (triton.cdiv(M, BLOCK_M), 1)
    gemm_fusion_kernel[grid](
        A, B, C, E, M, N, K,  #
        A.stride(0), A.stride(1),  #
        B.stride(0), B.stride(1),  #
        C.stride(0), C.stride(1),  #
        E.stride(0), E.stride(1),  #
        BLOCK_M, BLOCK_N, BLOCK_K,  #
        num_warps=num_warps)

    torch.testing.assert_close(ref_out, E, atol=1e-2, rtol=0)


@triton.jit
def batched_gemm_fusion(Q, K, V, Out,  #
                        stride_qz, stride_qh, stride_qm, stride_qk,  #
                        stride_kz, stride_kh, stride_kn, stride_kk,  #
                        stride_vz, stride_vh, stride_vk, stride_vn,  #
                        stride_oz, stride_oh, stride_om, stride_on,  #
                        Z, NH, N_CTX,  #
                        BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,  #
                        BLOCK_N: tl.constexpr):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    q_tile_ptr = tl.make_block_ptr(
        base=Q,
        shape=(Z, NH, N_CTX, BLOCK_DMODEL),
        strides=(stride_qz, stride_qh, stride_qm, stride_qk),
        offsets=(off_hz // NH, off_hz % NH, start_m, 0),
        block_shape=(1, 1, BLOCK_M, BLOCK_DMODEL),
        order=(3, 2, 1, 0),
    )
    k_tile_ptr = tl.make_block_ptr(
        base=K,
        shape=(Z, NH, N_CTX, BLOCK_DMODEL),
        strides=(stride_kz, stride_kh, stride_kn, stride_kk),
        offsets=(off_hz // NH, off_hz % NH, 0, 0),
        block_shape=(1, 1, BLOCK_N, BLOCK_DMODEL),
        order=(3, 2, 1, 0),
    )
    v_tile_ptr = tl.make_block_ptr(
        base=V,
        shape=(Z, NH, N_CTX, BLOCK_DMODEL),
        strides=(stride_vz, stride_vh, stride_vk, stride_vn),
        offsets=(off_hz // NH, off_hz % NH, 0, 0),
        block_shape=(1, 1, BLOCK_N, BLOCK_DMODEL),
        order=(3, 2, 1, 0),
    )
    o_tile_ptr = tl.make_block_ptr(
        base=Out,
        shape=(Z, NH, N_CTX, BLOCK_DMODEL),
        strides=(stride_oz, stride_oh, stride_om, stride_on),
        offsets=(off_hz // NH, off_hz % NH, start_m, 0),
        block_shape=(1, 1, BLOCK_M, BLOCK_DMODEL),
        order=(3, 2, 1, 0),
    )

    q = tl.load(q_tile_ptr, boundary_check=(0, 1, 2, 3))
    q = tl.reshape(q, (BLOCK_M, BLOCK_DMODEL), can_reorder=True)
    for i in range(0, N_CTX, BLOCK_N):
        k = tl.load(k_tile_ptr, boundary_check=(0, 1, 2, 3))
        k = tl.reshape(k, (BLOCK_N, BLOCK_DMODEL), can_reorder=True)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))

        p = qk.to(tl.float16)
        v = tl.load(v_tile_ptr, boundary_check=(0, 1, 2, 3))
        v = tl.reshape(v, (BLOCK_N, BLOCK_DMODEL), can_reorder=True)
        acc += tl.dot(p, v)

        k_tile_ptr = tl.advance(k_tile_ptr, [0, 0, BLOCK_N, 0])
        v_tile_ptr = tl.advance(v_tile_ptr, [0, 0, BLOCK_N, 0])

    acc = tl.reshape(acc, (1, 1, BLOCK_M, BLOCK_DMODEL), can_reorder=True)
    acc = acc.to(tl.float16)
    tl.store(o_tile_ptr, acc)


@pytest.mark.skip(reason="don't support 4d across stack, left for future")
def test_batched_gemm_fusion():
    Z = 4
    NH = 48
    H = 64
    N_CTX = 2048
    BLOCK_M, BLOCK_N, BLOCK_DMODEL = 128, 128, H
    torch.manual_seed(20)
    A = torch.empty((Z, NH, N_CTX, H), dtype=torch.float16, device='cuda').normal_(mean=0.1, std=0.2)
    B = torch.empty((Z, NH, N_CTX, H), dtype=torch.float16, device='cuda').normal_(mean=0.1, std=0.2)
    C = torch.empty((Z, NH, N_CTX, H), dtype=torch.float16, device='cuda').normal_(mean=0.1, std=0.2)
    E = torch.empty_like(A)
    BT = B.transpose(-1, -2)
    ref_out = torch.matmul(torch.matmul(A, BT), C)
    num_warps = 4
    grid = (triton.cdiv(N_CTX, BLOCK_M), B * NH)
    batched_gemm_fusion[grid](
        A, B, C, E,  #
        A.stride(0), A.stride(1), A.stride(2), A.stride(3),  #
        B.stride(0), B.stride(1), B.stride(2), B.stride(3),  #
        C.stride(0), C.stride(1), C.stride(2), C.stride(3),  #
        E.stride(0), E.stride(1), E.stride(2), E.stride(3),  #
        Z, NH, N_CTX,  #
        BLOCK_M, BLOCK_DMODEL, BLOCK_N, num_warps=num_warps)

    torch.testing.assert_close(ref_out, E, atol=1e-2, rtol=0)
