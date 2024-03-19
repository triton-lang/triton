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
from triton.runtime import driver


# kernel used to query max clusters for persistent kernel when NUM_CTAS > 1
@triton.jit
def empty_kernel(null, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pass


@triton.jit
def static_persistent_matmul_kernel(  #
        a_ptr, b_ptr, c_ptr,  #
        M, N, K,  #
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,  #
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,  #
        NUM_SMS: tl.constexpr  #
):
    start_tile = tl.program_id(axis=0)
    m_tiles = tl.cdiv(M, BLOCK_M)
    n_tiles = tl.cdiv(N, BLOCK_N)
    num_tiles = m_tiles * n_tiles
    offs_k = tl.arange(0, BLOCK_K)

    for tile_id in range(start_tile, num_tiles, NUM_SMS):
        pid_m = tile_id // n_tiles
        pid_n = tile_id % n_tiles
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
        offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        for k in range(0, K, BLOCK_K):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            accumulator += tl.dot(a, b)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        offs_cm = tl.arange(0, BLOCK_M) + pid_m * BLOCK_M
        offs_cn = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N

        c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
        tl.store(c_ptrs, accumulator)


@triton.jit
def static_persistent_tma_matmul_kernel(  #
        a_ptr, b_ptr, c_ptr,  #
        M, N, K,  #
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,  #
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,  #
        NUM_SMS: tl.constexpr  #
):
    start_tile = tl.program_id(axis=0)
    m_tiles = tl.cdiv(M, BLOCK_M)
    n_tiles = tl.cdiv(N, BLOCK_N)
    k_tiles = tl.cdiv(K, BLOCK_K)
    num_tiles = m_tiles * n_tiles

    pre_pid_m = start_tile // n_tiles
    pre_pid_n = start_tile % n_tiles

    block_offset_m = pre_pid_m * BLOCK_M
    block_offset_n = pre_pid_n * BLOCK_N
    a_tile_ptr = tl.make_block_ptr(base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
                                   offsets=(block_offset_m, 0), block_shape=(BLOCK_M, BLOCK_K), order=(1, 0))
    b_tile_ptr = tl.make_block_ptr(base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
                                   offsets=(0, block_offset_n), block_shape=(BLOCK_K, BLOCK_N), order=(0, 1))
    for tile_id in range(start_tile, num_tiles, NUM_SMS):
        pid_m = tile_id // n_tiles
        pid_n = tile_id % n_tiles
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        if tile_id >= NUM_SMS:
            a_tile_ptr = tl.advance(a_tile_ptr, [(pid_m - pre_pid_m) * BLOCK_M, -k_tiles * BLOCK_K])
            b_tile_ptr = tl.advance(b_tile_ptr, [-k_tiles * BLOCK_K, (pid_n - pre_pid_n) * BLOCK_N])

        for k in range(0, K, BLOCK_K):
            a = tl.load(a_tile_ptr)
            b = tl.load(b_tile_ptr)
            accumulator += tl.dot(a, b)
            a_tile_ptr = tl.advance(a_tile_ptr, [0, BLOCK_K])
            b_tile_ptr = tl.advance(b_tile_ptr, [BLOCK_K, 0])

        offs_m = tl.arange(0, BLOCK_M) + pid_m * BLOCK_M
        offs_n = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N

        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, accumulator)
        pre_pid_m = pid_m
        pre_pid_n = pid_n


@pytest.mark.parametrize('M,N,K,BLOCK_M,BLOCK_N,BLOCK_K,NUM_WARPS,NUM_CTAS,TRANS_A,TRANS_B,USE_TMA', [(
    *shape, use_tma
) for shape in [
    [4096, 4096, 64, 64, 64, 16, 4, 1, False, True],
    [4096, 4096, 64, 64, 64, 32, 4, 1, False, True
     ],
    [4096, 4096, 64, 256, 64, 16, 4, 1, False, True
     ],
    [4096, 4096, 64, 128, 128, 16, 4, 1, False, True
     ],
    # TODO: fix issue for 8-warp persistent kernel
    # [4096, 4096, 64, 128, 128, 16, 8, 1, False, True],
    # [4096, 4096, 64, 128, 256, 16, 8, 1, False, True],
] for use_tma in [False, True]])
@pytest.mark.skipif(torch.cuda.get_device_capability()[0] < 9, reason="Requires compute capability >= 9")
def test_user_defined_persistent_non_warp_specialized_gemm(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARPS, NUM_CTAS,
                                                           TRANS_A, TRANS_B, USE_TMA):
    if (TRANS_A):
        a = .1 * torch.randn((K, M), device='cuda', dtype=torch.float16).T
    else:
        a = .1 * torch.randn((M, K), device='cuda', dtype=torch.float16)

    if (TRANS_B):
        b = .1 * torch.randn((N, K), device='cuda', dtype=torch.float16).T
    else:
        b = .1 * torch.randn((K, N), device='cuda', dtype=torch.float16)
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)

    NUM_SMS = torch.cuda.get_device_properties('cuda').multi_processor_count
    grid = lambda META: (min(META['NUM_SMS'], triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N'])), )

    if USE_TMA:
        static_persistent_tma_matmul_kernel[grid](a_ptr=a, b_ptr=b, c_ptr=c, M=M, N=N, K=K, stride_am=a.stride(0),
                                                  stride_ak=a.stride(1), stride_bk=b.stride(0), stride_bn=b.stride(1),
                                                  stride_cm=c.stride(0), stride_cn=c.stride(1), BLOCK_M=BLOCK_M,
                                                  BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, NUM_SMS=NUM_SMS,
                                                  num_warps=NUM_WARPS, num_ctas=NUM_CTAS)
    else:
        static_persistent_matmul_kernel[grid](a_ptr=a, b_ptr=b, c_ptr=c, M=M, N=N, K=K, stride_am=a.stride(0),
                                              stride_ak=a.stride(1), stride_bk=b.stride(0), stride_bn=b.stride(1),
                                              stride_cm=c.stride(0), stride_cn=c.stride(1), BLOCK_M=BLOCK_M,
                                              BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, NUM_SMS=NUM_SMS, num_warps=NUM_WARPS,
                                              num_ctas=NUM_CTAS)

    th_c = torch.matmul(a, b)
    torch.testing.assert_close(th_c, c, atol=1e-2, rtol=0, check_dtype=False)


@triton.jit
def warp_specialized_matmul_kernel(  #
        a_ptr, b_ptr, c_ptr,  #
        M, N, K,  #
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,  #
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,  #
):
    tid = tl.program_id(axis=0)
    n_tiles = tl.cdiv(N, BLOCK_N)
    pid_m = tid // n_tiles
    pid_n = tid % n_tiles

    offs_k = tl.arange(0, BLOCK_K)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_am = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    offs_bn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    accumulator = accumulator.to(c_ptr.dtype.element_ty)

    offs_cm = tl.arange(0, BLOCK_M) + pid_m * BLOCK_M
    offs_cn = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N

    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    mask = (offs_cm < M)[:, None] & (offs_cn < N)[None, :]
    tl.store(c_ptrs, accumulator, mask=mask)


@triton.jit
def tma_warp_specialized_matmul_kernel(  #
        a_ptr, b_ptr, c_ptr,  #
        M, N, K,  #
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,  #
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,  #
):
    tid = tl.program_id(axis=0)
    n_tiles = tl.cdiv(N, BLOCK_N)
    pid_m = tid // n_tiles
    pid_n = tid % n_tiles

    block_offset_m = pid_m * BLOCK_M
    block_offset_n = pid_n * BLOCK_N
    a_tile_ptr = tl.make_block_ptr(base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
                                   offsets=(block_offset_m, 0), block_shape=(BLOCK_M, BLOCK_K), order=(1, 0))
    b_tile_ptr = tl.make_block_ptr(base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
                                   offsets=(0, block_offset_n), block_shape=(BLOCK_K, BLOCK_N), order=(0, 1))

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(a_tile_ptr)
        b = tl.load(b_tile_ptr)
        accumulator += tl.dot(a, b)
        a_tile_ptr = tl.advance(a_tile_ptr, [0, BLOCK_K])
        b_tile_ptr = tl.advance(b_tile_ptr, [BLOCK_K, 0])
    accumulator = accumulator.to(c_ptr.dtype.element_ty)

    offs_cm = tl.arange(0, BLOCK_M) + pid_m * BLOCK_M
    offs_cn = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N

    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    mask = (offs_cm < M)[:, None] & (offs_cn < N)[None, :]
    tl.store(c_ptrs, accumulator, mask=mask)


@pytest.mark.parametrize('M,N,K,BLOCK_M,BLOCK_N,BLOCK_K,NUM_CTAS,TRANS_A,TRANS_B,USE_TMA',
                         [(*shape, use_tma) for shape in [
                             [2048, 2048, 64, 64, 64, 16, 1, False, True],
                             [4096, 4096, 64, 64, 64, 16, 1, False, True],
                             [128, 4096, 64, 64, 64, 16, 1, False, True],
                             [4096, 128, 64, 64, 64, 16, 1, False, True],
                             [4096, 4096, 64, 64, 64, 32, 1, False, True],
                             [4096, 4096, 256, 128, 128, 16, 1, False, True],
                             [4096, 4096, 320, 128, 64, 64, 1, False, True],
                             [4096, 4096, 320, 64, 128, 64, 1, False, True],
                             [4096, 4096, 320, 128, 128, 64, 1, False, True],
                             [4096, 4096, 256, 256, 64, 16, 1, False, True],
                             [4096, 4096, 256, 256, 64, 64, 1, False, True],
                             [4096, 4096, 256, 64, 256, 16, 1, False, True],
                             [4096, 4096, 256, 64, 256, 64, 1, False, True],
                             [4096, 4096, 256, 256, 128, 16, 1, False, True],
                             [4096, 4096, 256, 256, 128, 64, 1, False, True],
                             [4096, 4096, 256, 128, 256, 16, 1, False, True],
                             [4096, 4096, 256, 128, 256, 64, 1, False, True],
                             # numCTAs > 1
                             [2048, 2048, 64, 128, 128, 64, 2, False, True],
                             [2048, 2048, 128, 256, 128, 64, 4, False, True],
                             [4096, 4096, 128, 256, 128, 64, 4, False, True],
                             [4096, 4096, 256, 128, 256, 64, 4, False, True],
                             [4096, 4096, 256, 256, 256, 64, 4, False, True],
                         ] for use_tma in [False, True]])
@pytest.mark.skipif(torch.cuda.get_device_capability()[0] < 9, reason="Requires compute capability >= 9")
def test_non_persistent_warp_specialized_gemm(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, NUM_CTAS, TRANS_A, TRANS_B, USE_TMA):
    if (TRANS_A):
        a = .1 * torch.randn((K, M), device='cuda', dtype=torch.float16).T
    else:
        a = .1 * torch.randn((M, K), device='cuda', dtype=torch.float16)

    if (TRANS_B):
        b = .1 * torch.randn((N, K), device='cuda', dtype=torch.float16).T
    else:
        b = .1 * torch.randn((K, N), device='cuda', dtype=torch.float16)

    c = torch.empty((M, N), device=a.device, dtype=torch.float32)

    grid = lambda META: (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), )

    if USE_TMA:
        tma_warp_specialized_matmul_kernel[grid](
            a, b, c,  #
            M, N, K,  #
            a.stride(0), a.stride(1),  #
            b.stride(0), b.stride(1),  #
            c.stride(0), c.stride(1),  #
            BLOCK_M, BLOCK_N, BLOCK_K,  #
            num_warps=4,  #
            num_ctas=NUM_CTAS,  #
            enable_warp_specialization=True)
    else:
        warp_specialized_matmul_kernel[grid](
            a, b, c,  #
            M, N, K,  #
            a.stride(0), a.stride(1),  #
            b.stride(0), b.stride(1),  #
            c.stride(0), c.stride(1),  #
            BLOCK_M, BLOCK_N, BLOCK_K,  #
            num_warps=4,  #
            num_ctas=NUM_CTAS,  #
            enable_warp_specialization=True)

    th_c = torch.matmul(a, b)
    torch.testing.assert_close(th_c, c, atol=1e-2, rtol=0, check_dtype=False)


@triton.jit
def static_persistent_warp_specialized_matmul_kernel(  #
        a_ptr, b_ptr, c_ptr,  #
        M, N, K,  #
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,  #
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,  #
        NUM_SMS: tl.constexpr  #
):
    start_tile = tl.program_id(axis=0)
    m_tiles = tl.cdiv(M, BLOCK_M)
    n_tiles = tl.cdiv(N, BLOCK_N)
    num_tiles = m_tiles * n_tiles
    offs_k = tl.arange(0, BLOCK_K)

    for tile_id in range(start_tile, num_tiles, NUM_SMS):
        pid_m = tile_id // n_tiles
        pid_n = tile_id % n_tiles
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
        offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        for k in range(0, K, BLOCK_K):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            accumulator += tl.dot(a, b)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        offs_cm = tl.arange(0, BLOCK_M) + pid_m * BLOCK_M
        offs_cn = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N

        c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
        tl.store(c_ptrs, accumulator)


@triton.jit
def static_persistent_tma_warp_specialized_matmul_kernel(  #
        a_ptr, b_ptr, c_ptr,  #
        M, N, K,  #
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,  #
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,  #
        NUM_SMS: tl.constexpr  #
):
    start_tile = tl.program_id(axis=0)
    m_tiles = tl.cdiv(M, BLOCK_M)
    n_tiles = tl.cdiv(N, BLOCK_N)
    k_tiles = tl.cdiv(K, BLOCK_K)
    num_tiles = m_tiles * n_tiles

    pre_pid_m = start_tile // n_tiles
    pre_pid_n = start_tile % n_tiles

    block_offset_m = pre_pid_m * BLOCK_M
    block_offset_n = pre_pid_n * BLOCK_N
    a_tile_ptr = tl.make_block_ptr(base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
                                   offsets=(block_offset_m, 0), block_shape=(BLOCK_M, BLOCK_K), order=(1, 0))
    b_tile_ptr = tl.make_block_ptr(base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
                                   offsets=(0, block_offset_n), block_shape=(BLOCK_K, BLOCK_N), order=(0, 1))
    for tile_id in range(start_tile, num_tiles, NUM_SMS):
        pid_m = tile_id // n_tiles
        pid_n = tile_id % n_tiles
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        if tile_id >= NUM_SMS:
            a_tile_ptr = tl.advance(a_tile_ptr, [(pid_m - pre_pid_m) * BLOCK_M, -k_tiles * BLOCK_K])
            b_tile_ptr = tl.advance(b_tile_ptr, [-k_tiles * BLOCK_K, (pid_n - pre_pid_n) * BLOCK_N])

        for k in range(0, K, BLOCK_K):
            a = tl.load(a_tile_ptr)
            b = tl.load(b_tile_ptr)
            accumulator += tl.dot(a, b)
            a_tile_ptr = tl.advance(a_tile_ptr, [0, BLOCK_K])
            b_tile_ptr = tl.advance(b_tile_ptr, [BLOCK_K, 0])

        offs_m = tl.arange(0, BLOCK_M) + pid_m * BLOCK_M
        offs_n = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N

        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, accumulator)
        pre_pid_m = pid_m
        pre_pid_n = pid_n


@pytest.mark.parametrize('M,N,K,BLOCK_M,BLOCK_N,BLOCK_K,NUM_CTAS,TRANS_A,TRANS_B,USE_TMA',
                         [(*shape, use_tma) for shape in [
                             [2048, 2048, 64, 64, 64, 16, 1, False, True],
                             [4096, 4096, 64, 64, 64, 16, 1, False, True],
                             [128, 4096, 64, 64, 64, 16, 1, False, True],
                             [4096, 128, 64, 64, 64, 16, 1, False, True],
                             [4096, 4096, 64, 64, 64, 32, 1, False, True],
                             [4096, 4096, 256, 128, 128, 16, 1, False, True],
                             [4096, 4096, 320, 128, 64, 64, 1, False, True],
                             [4096, 4096, 320, 64, 128, 64, 1, False, True],
                             [4096, 4096, 320, 128, 128, 64, 1, False, True],
                             [4096, 4096, 256, 256, 64, 16, 1, False, True],
                             [4096, 4096, 256, 256, 64, 64, 1, False, True],
                             [4096, 4096, 256, 64, 256, 16, 1, False, True],
                             [4096, 4096, 256, 64, 256, 64, 1, False, True],
                             [4096, 4096, 256, 256, 128, 16, 1, False, True],
                             [4096, 4096, 256, 256, 128, 64, 1, False, True],
                             [4096, 4096, 256, 128, 256, 16, 1, False, True],
                             [4096, 4096, 256, 128, 256, 64, 1, False, True],
                             # numCTAs > 1
                             [2048, 2048, 64, 128, 128, 64, 2, False, True],
                             [2048, 2048, 128, 256, 128, 64, 4, False, True],
                             [4096, 4096, 128, 256, 128, 64, 4, False, True],
                             [4096, 4096, 256, 128, 256, 64, 4, False, True],
                             [4096, 4096, 256, 256, 256, 64, 4, False, True],
                         ] for use_tma in [False, True]])
@pytest.mark.skipif(torch.cuda.get_device_capability()[0] < 9, reason="Requires compute capability >= 9")
def test_user_defined_persistent_warp_specialized_gemm(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, NUM_CTAS, TRANS_A, TRANS_B,
                                                       USE_TMA):
    if (TRANS_A):
        a = .1 * torch.randn((K, M), device='cuda', dtype=torch.float16).T
    else:
        a = .1 * torch.randn((M, K), device='cuda', dtype=torch.float16)

    if (TRANS_B):
        b = .1 * torch.randn((N, K), device='cuda', dtype=torch.float16).T
    else:
        b = .1 * torch.randn((K, N), device='cuda', dtype=torch.float16)
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)

    NUM_SMS = torch.cuda.get_device_properties('cuda').multi_processor_count
    grid = lambda META: (min(META['NUM_SMS'], triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N'])), )

    if USE_TMA:
        static_persistent_tma_warp_specialized_matmul_kernel[grid](
            a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), BLOCK_M,
            BLOCK_N, BLOCK_K, NUM_SMS, num_warps=4, num_ctas=NUM_CTAS,  #
            enable_warp_specialization=True)
    else:
        static_persistent_warp_specialized_matmul_kernel[grid](
            a, b, c,  #
            M, N, K,  #
            a.stride(0), a.stride(1),  #
            b.stride(0), b.stride(1),  #
            c.stride(0), c.stride(1),  #
            BLOCK_M, BLOCK_N, BLOCK_K, NUM_SMS,  #
            num_warps=4, num_ctas=NUM_CTAS,  #
            enable_warp_specialization=True)

    th_c = torch.matmul(a, b)
    torch.testing.assert_close(th_c, c, atol=1e-2, rtol=0, check_dtype=False)


@triton.jit
def static_persistent_matmul_no_scf_kernel(a_ptr, b_ptr, c_ptr,  #
                                           M, N, K,  #
                                           stride_am, stride_ak,  #
                                           stride_bk, stride_bn,  #
                                           stride_cm, stride_cn,  #
                                           BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,  #
                                           FLOAT16_OUTPUT: tl.constexpr, USE_TMA_EPILOGUE: tl.constexpr,  #
                                           NUM_SMS: tl.constexpr, USE_TMA_LOAD: tl.constexpr  #
                                           ):
    start_tile = tl.program_id(axis=0)
    m_tiles = tl.cdiv(M, BLOCK_M)
    n_tiles = tl.cdiv(N, BLOCK_N)
    num_tiles = m_tiles * n_tiles
    offs_k = tl.arange(0, BLOCK_K)
    pre_pid_m = start_tile // n_tiles
    pre_pid_n = start_tile % n_tiles
    block_offset_m = pre_pid_m * BLOCK_M
    block_offset_n = pre_pid_n * BLOCK_N

    if USE_TMA_LOAD:
        a_block_ptr = tl.make_block_ptr(base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
                                        offsets=(block_offset_m, 0), block_shape=(BLOCK_M, BLOCK_K), order=(1, 0))
        b_block_ptr = tl.make_block_ptr(base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
                                        offsets=(0, block_offset_n), block_shape=(BLOCK_K, BLOCK_N), order=(0, 1))
    if USE_TMA_EPILOGUE:
        c_block_ptr = tl.make_block_ptr(base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
                                        offsets=(block_offset_m, block_offset_n), block_shape=(BLOCK_M, BLOCK_N),
                                        order=(1, 0))

    for tile_id in range(start_tile, num_tiles, NUM_SMS):
        pid_m = tile_id // n_tiles
        pid_n = tile_id % n_tiles

        if USE_TMA_LOAD:
            a_block_ptr = tl.advance(a_block_ptr, [(pid_m - pre_pid_m) * BLOCK_M, 0])
            b_block_ptr = tl.advance(b_block_ptr, [0, (pid_n - pre_pid_n) * BLOCK_N])
            a = tl.load(a_block_ptr)
            b = tl.load(b_block_ptr)
        else:
            offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
            b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)

        c = tl.dot(a, b)

        if FLOAT16_OUTPUT:
            c = c.to(tl.float16)

        if USE_TMA_EPILOGUE:
            c_block_ptr = tl.advance(c_block_ptr, [(pid_m - pre_pid_m) * BLOCK_M, (pid_n - pre_pid_n) * BLOCK_N])
            tl.store(c_block_ptr, c)
        else:
            offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
            tl.store(c_ptrs, c)

        pre_pid_m = pid_m
        pre_pid_n = pid_n


@pytest.mark.parametrize(
    'M,N,K,NUM_CTAS,NUM_WARPS,TRANS_A,TRANS_B,OUTPUT_TYPE,USE_TMA_EPILOGUE,USE_TMA_LOAD',
    itertools.chain(*[[
        # numCTAs = 1, no TMA multicast:
        [64, 16, 16, 1, 4, False, True, "float16", USE_TMA_EPILOGUE, USE_TMA_LOAD],
        [64, 32, 16, 1, 4, False, True, "float16", USE_TMA_EPILOGUE, USE_TMA_LOAD],
        [64, 64, 16, 1, 4, False, True, "float16", USE_TMA_EPILOGUE, USE_TMA_LOAD],
        [64, 64, 16, 1, 4, False, True, "float32", USE_TMA_EPILOGUE, USE_TMA_LOAD],
        [64, 64, 32, 1, 4, False, True, "float32", USE_TMA_EPILOGUE, USE_TMA_LOAD],
        [64, 64, 64, 1, 4, False, True, "float32", USE_TMA_EPILOGUE, USE_TMA_LOAD],
        [128, 128, 16, 1, 4, False, True, "float16", USE_TMA_EPILOGUE, USE_TMA_LOAD],
        [128, 128, 16, 1, 4, False, True, "float32", USE_TMA_EPILOGUE, USE_TMA_LOAD],
        # small M, N
        [16, 16, 16, 1, 4, False, True, "float32", USE_TMA_EPILOGUE, USE_TMA_LOAD],
        [16, 32, 16, 1, 4, False, True, "float32", USE_TMA_EPILOGUE, USE_TMA_LOAD],
        [32, 16, 16, 1, 4, False, True, "float32", USE_TMA_EPILOGUE, USE_TMA_LOAD],
        [32, 32, 16, 1, 4, False, True, "float32", USE_TMA_EPILOGUE, USE_TMA_LOAD],
    ] for USE_TMA_EPILOGUE in [True, False] for USE_TMA_LOAD in [True, False]]))
@pytest.mark.skipif(torch.cuda.get_device_capability()[0] < 9, reason="Requires compute capability >= 9")
def test_static_persistent_matmul_no_scf_kernel(M, N, K, NUM_CTAS, NUM_WARPS, TRANS_A, TRANS_B, OUTPUT_TYPE,
                                                USE_TMA_EPILOGUE, USE_TMA_LOAD):
    if (TRANS_A):
        a = torch.randn((K, M), device='cuda', dtype=torch.float16).T
    else:
        a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    if (TRANS_B):
        b = torch.randn((N, K), device='cuda', dtype=torch.float16).T
    else:
        b = torch.randn((K, N), device='cuda', dtype=torch.float16)

    if OUTPUT_TYPE == "float16":
        c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    else:
        c = torch.empty((M, N), device=a.device, dtype=torch.float32)

    NUM_SMS = torch.cuda.get_device_properties('cuda').multi_processor_count

    # TODO: set `enable_warp_specialization=False` will lead to compilation error.
    static_persistent_matmul_no_scf_kernel[(NUM_SMS, )](
        a_ptr=a, b_ptr=b, c_ptr=c,  #
        M=M, N=N, K=K,  #
        stride_am=a.stride(0), stride_ak=a.stride(1),  #
        stride_bk=b.stride(0), stride_bn=b.stride(1),  #
        stride_cm=c.stride(0), stride_cn=c.stride(1),  #
        BLOCK_M=M if M < 128 else M // 2, BLOCK_N=N if N < 128 else N // 2, BLOCK_K=K, NUM_SMS=NUM_SMS,  #
        num_warps=NUM_WARPS,  #
        num_ctas=NUM_CTAS,  #
        FLOAT16_OUTPUT=(OUTPUT_TYPE == "float16"),  #
        USE_TMA_EPILOGUE=USE_TMA_EPILOGUE,  #
        USE_TMA_LOAD=USE_TMA_LOAD,  #
        enable_warp_specialization=True)
    a_f32 = a.to(torch.float32)
    b_f32 = b.to(torch.float32)
    golden = torch.matmul(a_f32, b_f32)
    torch.set_printoptions(profile="full")
    assert_close(c, golden, rtol=1e-2, atol=1e-3, check_dtype=False)


@triton.jit
def full_static_persistent_matmul_kernel(a_ptr, b_ptr, w_ptr, bias_ptr, z_ptr,  #
                                         M, N, K,  #
                                         stride_am, stride_ak,  #
                                         stride_bk, stride_bn,  #
                                         stride_wm, stride_wn,  #
                                         stride_zm, stride_zn,  #
                                         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                                         GROUP_SIZE_M: tl.constexpr,  #
                                         out_dtype: tl.constexpr, USE_TMA_STORE: tl.constexpr,  #
                                         ADD_MATRIX: tl.constexpr, ADD_ROWS: tl.constexpr, ADD_COLS: tl.constexpr,  #
                                         DO_SOFTMAX: tl.constexpr, CHAIN_DOT: tl.constexpr,  #
                                         A_ORDER_0: tl.constexpr, A_ORDER_1: tl.constexpr,  #
                                         B_ORDER_0: tl.constexpr, B_ORDER_1: tl.constexpr,  #
                                         NUM_SMS: tl.constexpr  #
                                         ):
    start_pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_tiles = num_pid_m * num_pid_n
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = start_pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pre_pid_m = first_pid_m + ((start_pid % num_pid_in_group) % group_size_m)
    pre_pid_n = (start_pid % num_pid_in_group) // group_size_m

    pre_block_offset_m = pre_pid_m * BLOCK_M
    pre_block_offset_n = pre_pid_n * BLOCK_N
    a_tile_ptr = tl.make_block_ptr(base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
                                   offsets=(pre_block_offset_m, 0), block_shape=(BLOCK_M, BLOCK_K),
                                   order=(A_ORDER_0, A_ORDER_1))
    b_tile_ptr = tl.make_block_ptr(base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
                                   offsets=(0, pre_block_offset_n), block_shape=(BLOCK_K, BLOCK_N),
                                   order=(B_ORDER_0, B_ORDER_1))
    w_tile_ptr = tl.make_block_ptr(base=w_ptr, shape=(N, N), strides=(stride_wm, stride_wn),
                                   offsets=(0, pre_block_offset_n), block_shape=(BLOCK_N, BLOCK_N), order=(0, 1))

    if USE_TMA_STORE:
        z_block_ptr = tl.make_block_ptr(base=z_ptr, shape=(M, N), strides=(stride_zm, stride_zn),
                                        offsets=(pre_block_offset_m, pre_block_offset_n),
                                        block_shape=(BLOCK_M, BLOCK_N), order=(1, 0))

    for tile_id in range(start_pid, num_tiles, NUM_SMS):
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m
        block_offset_m = pid_m * BLOCK_M
        block_offset_n = pid_n * BLOCK_N

        offs_m = block_offset_m + tl.arange(0, BLOCK_M)
        offs_n = block_offset_n + tl.arange(0, BLOCK_N)
        z_ptrs = z_ptr + offs_m[:, None] * stride_zm + offs_n[None, :] * stride_zn
        bias_ptrs = bias_ptr + offs_m[:, None] * stride_zm + offs_n[None, :] * stride_zn
        mask = (offs_m < M)[:, None] & (offs_n < N)[None, :]

        # TODO: lib/Dialect/TritonGPU/Transforms/RewriteTensorPointer.cpp does not support scf.if yet.
        # if tile_id >= NUM_SMS:
        #     a_tile_ptr = tl.advance(a_tile_ptr, [(pid_m - pre_pid_m) * BLOCK_M, -tl.cdiv(K, BLOCK_K) * BLOCK_K])
        #     b_tile_ptr = tl.advance(b_tile_ptr, [-tl.cdiv(K, BLOCK_K) * BLOCK_K, (pid_n - pre_pid_n) * BLOCK_N])

        a_tile_ptr = tl.advance(a_tile_ptr, [(pid_m - pre_pid_m) * BLOCK_M, 0])
        b_tile_ptr = tl.advance(b_tile_ptr, [0, (pid_n - pre_pid_n) * BLOCK_N])
        z = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            a = tl.load(a_tile_ptr, boundary_check=(0, 1))
            b = tl.load(b_tile_ptr, boundary_check=(0, 1))
            z += tl.dot(a, b)
            a_tile_ptr = tl.advance(a_tile_ptr, [0, BLOCK_K])
            b_tile_ptr = tl.advance(b_tile_ptr, [BLOCK_K, 0])
        a_tile_ptr = tl.advance(a_tile_ptr, [0, -tl.cdiv(K, BLOCK_K) * BLOCK_K])
        b_tile_ptr = tl.advance(b_tile_ptr, [-tl.cdiv(K, BLOCK_K) * BLOCK_K, 0])

        if (out_dtype == tl.constexpr(tl.float16)):
            z = z.to(tl.float16)

        if ADD_MATRIX:
            z += tl.load(bias_ptrs, mask=mask)
        if ADD_ROWS:
            ZRs = bias_ptr + offs_m * stride_zm
            z += tl.load(ZRs)[:, None]
        if ADD_COLS:
            ZCs = bias_ptr + offs_n * stride_zn
            z += tl.load(ZCs)[None, :]
        if DO_SOFTMAX:
            max = tl.max(z, 1)
            z = z - max[:, None]
            num = tl.exp(z.to(tl.float32)).to(max.dtype)
            den = tl.sum(num, 1)
            z = num / den[:, None]
        if CHAIN_DOT:
            w = tl.load(w_tile_ptr)
            w_tile_ptr = tl.advance(w_tile_ptr, [0, (pid_n - pre_pid_n) * BLOCK_N])
            z = tl.dot(z.to(w.dtype), w)
            if (out_dtype == tl.constexpr(tl.float16)):
                z = z.to(tl.float16)

        if USE_TMA_STORE:
            z_block_ptr = tl.advance(z_block_ptr, [(pid_m - pre_pid_m) * BLOCK_M, (pid_n - pre_pid_n) * BLOCK_N])
            tl.store(z_block_ptr, z, boundary_check=(0, 1))
        else:
            tl.store(z_ptrs, z, mask=mask)

        pre_pid_m = pid_m
        pre_pid_n = pid_n


@pytest.mark.parametrize(
    'BLOCK_M,BLOCK_N,BLOCK_K,NUM_WARPS,NUM_CTAS,M,N,K,TRANS_A,TRANS_B,epilogue,out_dtype,USE_TMA_STORE,NUM_STAGES,ENABLE_WS',
    [
        # corner shapes
        (128, 128, 64, 4, 1, *shape_w_c, 'none', out_dtype, use_tma_store, 3, enable_ws) for shape_w_c in [
            [4096, 1, 1024, False, False],
            [2048, 204, 1000, True, False],
            [16, 524288, 32, False, True],
        ] for out_dtype in ['float16', 'float32'] for use_tma_store in [False, True] for enable_ws in [True]
    ] + [
        # softmax epilogue
        (*shape_w_c, trans_a, trans_b, epilogue, out_dtype, use_tma_store, num_stages, enable_ws)
        # softmax works for one CTA
        for shape_w_c in [
            [64, 64, 16, 4, 1, 64, 64, 64],
            [128, 128, 64, 4, 1, None, None, None],
            [16, 16, 64, 4, 1, 16, 16, 64],
            # TODO: enable when num_warps != 4 is supported.
            # [64, 64, 32, 8, 1, 64, 64, 64],
            [128, 128, 64, 4, 1, 128, 128, 128],
        ]
        for epilogue in ['softmax']
        for out_dtype in ['float16', 'float32']
        for use_tma_store in [False, True]
        for trans_a in [False]
        for trans_b in [True]
        for num_stages in [3]
        for enable_ws in [True]
    ] + [
        # loop over tile shapes and transpose combinations
        (*shape_w_c, trans_a, trans_b, 'none', out_dtype, use_tma_store, num_stages, enable_ws) for shape_w_c in [
            [64, 64, 32, 4, 1, 128, 256, 64],
            [128, 128, 16, 4, 4, 512, 256, 64],
            [128, 256, 32, 4, 8, 256, 256, 192],
            [512, 256, 32, 4, 8, 1024, 256, 192],
            # BLOCK_K >= 128
            [64, 128, 128, 4, 1, 512, 256, 256],
            [128, 128, 128, 4, 1, 256, 256, 192],
            [128, 128, 128, 4, 2, 256, 256, 192],
            # small BLOCK_M and BLOCK_K
            [16, 32, 32, 4, 1, 128, 256, 64],
            [32, 32, 16, 4, 1, 256, 256, 192],
            [16, 32, 64, 4, 4, 512, 256, 64],
        ] for out_dtype in ['float32'] for use_tma_store in [False] for trans_a in [False, True] for trans_b in
        [False, True] for num_stages in [3] for enable_ws in [True]
    ] + [
        # loop over epilogues besides of softmax
        (*shape_w_c, trans_a, trans_b, epilogue, out_dtype, use_tma_store, num_stages, enable_ws) for shape_w_c in [
            [64, 64, 16, 4, 1, 128, 128, 64],
            *[[256, 64, 16, num_warps, num_ctas, 256, 256, 64] for num_warps in [4] for num_ctas in [1, 2, 4]],
            # for chain-dot
            [128, 128, 64, 4, 1, None, None, None],
            [64, 64, 16, 4, 1, None, None, None],
            # small BLOCK_M and BLOCK_K
            [16, 16, 64, 4, 1, 128, 128, 64],
            *[[16, 32, 64, num_warps, num_ctas, 256, 256, 256] for num_warps in [4] for num_ctas in [1, 2]],
            #  # TODO: enable when num_warps != 4 is supported.
            #  # repeat
            #  # [64, 64, 32, 8, 1, 128, 256, 64],
            #  # [64, 64, 16, 8, 2, 128, 128, 64],
            # irregular shape
            [128, 128, 64, 4, 1, 500, 200, 128],
            [128, 128, 64, 4, 1, 513, 193, 192],
        ] for epilogue in ['none', 'add-matrix', 'add-rows', 'add-cols', 'chain-dot'] for out_dtype in
        ['float16', 'float32'] for use_tma_store in [False, True] for trans_a in [False] for trans_b in [True] for
        num_stages in [3] for enable_ws in [True] if not (epilogue == 'chain-dot' and
                                                          (shape_w_c[5] is not None or shape_w_c[0] != shape_w_c[1]))
    ] + [
        # loop over instr shapes & pipeline stages
        (64, n, 16, 4, 1, 512, 256, 256, False, True, 'none', out_dtype, use_tma_store, num_stages, enable_ws)
        for n in [16, 32, 64, 128, 256]
        for out_dtype in ['float32']
        for use_tma_store in [False]
        for num_stages in [2, 4, 5, 7]
        for enable_ws in [True]
    ] + [
        # irregular shapes
        (*shape_w_c, *shape, False, True, 'none', out_dtype, use_tma_store, num_stages, enable_ws)
        for shape_w_c in [[128, 128, 64, 4, 1], [256, 128, 64, 4, 2], [128, 128, 128, 4, 2]]
        for shape in [
            [512, 360, 1024],
            [360, 4096, 512],
        ]
        for out_dtype in ['float32']
        for use_tma_store in [False, True]
        for num_stages in [3, 4]
        for enable_ws in [True]
    ] + [
        # larger NUM_CTAS
        [1024, 128, 64, 4, 8, 1300, 1800, 3000, False, False, 'none', 'float16', True, 5, True],
        [512, 256, 64, 4, 8, 800, 30000, 10000, True, True, 'none', 'float16', True, 4, True],
        [1024, 128, 64, 4, 8, 1800, 10000, 15000, True, True, 'none', 'float16', True, 5, True],
        [512, 256, 64, 4, 8, 1300, 1800, 3000, False, False, 'none', 'float16', True, 5, True],
        [128, 1024, 64, 4, 8, 800, 30000, 10000, True, True, 'none', 'float16', True, 5, True],
        [512, 256, 64, 4, 8, 1800, 10000, 15000, True, True, 'none', 'float16', True, 5, True],
    ])
@pytest.mark.skipif(torch.cuda.get_device_capability()[0] < 9, reason="Requires compute capability >= 9")
def test_full_static_persistent_matmul_kernel(BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARPS, NUM_CTAS, M, N, K, TRANS_A, TRANS_B,
                                              epilogue, out_dtype, USE_TMA_STORE, NUM_STAGES, ENABLE_WS):
    if '-'.join(
            map(str, [
                BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARPS, NUM_CTAS, M, N, K, epilogue, out_dtype, USE_TMA_STORE, NUM_STAGES,
                ENABLE_WS
            ])) in [
                '128-128-128-4-1-256-256-192-none-float32-True-3-True',
            ]:
        pytest.skip('out of resource: shared memory, Required: 263168')

    if '-'.join(map(str, [BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARPS, NUM_CTAS, M, N, K, TRANS_A, TRANS_B])) in [
            '16-32-64-4-4-512-256-64-True-False',
            '16-32-64-4-4-512-256-64-True-True',
            '16-32-64-4-4-512-256-64-False-False',
            '16-32-64-4-4-512-256-64-False-True',
    ]:
        pytest.skip('shapePerCTA[1] < 16 not supported')

    if '-'.join(map(str, [BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARPS, NUM_CTAS, M, N, K, TRANS_B])) in [
            '16-32-64-4-1-256-256-256-False',
            '16-32-64-4-2-256-256-256-False',
            '16-32-64-4-2-256-256-256-True',
            '16-32-64-8-2-256-256-256-False',
            '16-32-64-8-2-256-256-256-True',
    ]:
        pytest.skip('Known legacy issue, ldmatrix can only support x4')

    if epilogue == 'chain-dot':
        pytest.skip('known failure: Assertion !region.empty() && unexpected empty region.')

    M = BLOCK_M if M is None else M
    N = BLOCK_N if N is None else N
    K = BLOCK_K if K is None else K

    if (TRANS_A):
        a = torch.randn((K, M), device='cuda', dtype=torch.float16).T
        a_order = [0, 1]
    else:
        a = torch.randn((M, K), device='cuda', dtype=torch.float16)
        a_order = [1, 0]

    if (TRANS_B):
        b = torch.randn((N, K), device='cuda', dtype=torch.float16).T
        b_order = [0, 1]
    else:
        b = torch.randn((K, N), device='cuda', dtype=torch.float16)
        b_order = [1, 0]

    if out_dtype == 'float16' and epilogue != 'softmax':
        # TODO: for out_dtype == 'float16' and epilogue == 'softmax', it will
        # fail with the following error: 'llvm.fmul' op requires the same type
        # for all operands and results
        out_dtype = tl.float16
        torch_out_dtype = torch.float16
    else:
        out_dtype = tl.float32
        torch_out_dtype = torch.float32

    # avoid out of memory
    if epilogue in ['add-matrix', 'add-rows', 'add-cols']:
        bias = torch.randn((M, N), device='cuda', dtype=torch_out_dtype)
    else:
        bias = torch.randn((1, 1), device='cuda', dtype=torch_out_dtype)

    if epilogue == 'chain-dot':
        w = torch.randn((N, N), device='cuda', dtype=torch.float16).T
    else:
        w = torch.randn((1, 1), device='cuda', dtype=torch.float16).T

    z = torch.full((M, N), 1., device='cuda', dtype=torch_out_dtype)

    # torch result
    a_f32 = a.to(torch.float32)
    b_f32 = b.to(torch.float32)
    dot = torch.matmul(a_f32, b_f32)

    def process_epilogue(d, bias, w, epilogue):
        if epilogue == 'add-matrix':
            ref = d + bias
        elif epilogue == 'add-rows':
            ref = d + bias[:, 0][:, None]
        elif epilogue == 'add-cols':
            ref = d + bias[0, :][None, :]
        elif epilogue == 'softmax':
            num = torch.exp(d - torch.max(d, dim=-1, keepdims=True)[0])
            denom = torch.sum(num, dim=-1, keepdims=True)
            ref = num / denom
            # ref = torch.softmax(d, 1)
        elif epilogue == 'chain-dot':
            ref = torch.matmul(d, w.to(torch.float32))
        else:
            ref = d
        return ref

    golden = process_epilogue(dot, bias, w, epilogue)

    NUM_SMS = torch.cuda.get_device_properties('cuda').multi_processor_count
    if NUM_CTAS > 1:
        src = triton.compiler.ASTSource(fn=empty_kernel, signature="i32", constants={"BLOCK_M": 64, "BLOCK_N": 64})
        null_kernel = triton.compile(src)
        null_kernel._init_handles()
        device = driver.get_current_device()
        max_shared_mem = driver.utils.get_device_properties(device)["max_shared_mem"]
        num_clusters = driver.utils.cu_occupancy_max_active_clusters(null_kernel.function, max_shared_mem, NUM_CTAS, 1,
                                                                     1)
        NUM_SMS = num_clusters

    def grid(META):
        return (min(NUM_SMS, triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N'])), )

    full_static_persistent_matmul_kernel[grid](
        a_ptr=a, b_ptr=b, w_ptr=w, bias_ptr=bias, z_ptr=z,  #
        M=M, N=N, K=K,  #
        stride_am=a.stride(0), stride_ak=a.stride(1),  #
        stride_bk=b.stride(0), stride_bn=b.stride(1),  #
        stride_wm=w.stride(0), stride_wn=w.stride(1),  #
        stride_zm=z.stride(0), stride_zn=z.stride(1),  #
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_SIZE_M=8,  #
        out_dtype=out_dtype,  #
        USE_TMA_STORE=USE_TMA_STORE,  #
        ADD_MATRIX=epilogue == 'add-matrix',  #
        ADD_ROWS=epilogue == 'add-rows',  #
        ADD_COLS=epilogue == 'add-cols',  #
        DO_SOFTMAX=epilogue == 'softmax',  #
        CHAIN_DOT=epilogue == 'chain-dot',  #
        A_ORDER_0=a_order[0], A_ORDER_1=a_order[1],  #
        B_ORDER_0=b_order[0], B_ORDER_1=b_order[1],  #
        num_warps=NUM_WARPS, num_ctas=NUM_CTAS, num_stages=NUM_STAGES,  #
        enable_warp_specialization=ENABLE_WS,  #
        NUM_SMS=NUM_SMS)

    torch.set_printoptions(profile="full")
    golden = torch.nn.functional.normalize(golden)
    z = torch.nn.functional.normalize(z)
    assert_close(z, golden, rtol=1e-2, atol=1e-3, check_dtype=False)
