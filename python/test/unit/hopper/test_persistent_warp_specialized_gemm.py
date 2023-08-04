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
def static_persistent_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_SM: tl.constexpr,
):
    start_tile = tl.program_id(axis=0)
    m_tiles = tl.cdiv(M, BLOCK_M)
    n_tiles = tl.cdiv(N, BLOCK_N)
    num_tiles = m_tiles * n_tiles
    offs_k = tl.arange(0, BLOCK_K)

    for tile_id in range(start_tile, num_tiles, NUM_SM):
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
def static_persistent_tma_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_SM: tl.constexpr,
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
    a_tile_ptr = tl.make_block_ptr(base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak), offsets=(block_offset_m, 0), block_shape=(BLOCK_M, BLOCK_K), order=(1, 0))
    b_tile_ptr = tl.make_block_ptr(base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn), offsets=(0, block_offset_n), block_shape=(BLOCK_K, BLOCK_N), order=(0, 1))
    for tile_id in range(start_tile, num_tiles, NUM_SM):
        pid_m = tile_id // n_tiles
        pid_n = tile_id % n_tiles
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        if tile_id >= NUM_SM:
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


@pytest.mark.parametrize('M,N,K,BLOCK_M,BLOCK_N,BLOCK_K,NUM_WARPS,NUM_CTAS,TRANS_A,TRANS_B', [
    [4096, 4096, 64, 64, 64, 16, 4, 1, False, True],
    [4096, 4096, 64, 64, 64, 32, 4, 1, False, True],
    [4096, 4096, 64, 256, 64, 16, 4, 1, False, True],
    [4096, 4096, 64, 128, 128, 16, 4, 1, False, True],
    # TODO: fix issue for 8-warp persistent kernel
    # [4096, 4096, 64, 128, 128, 16, 8, 1, False, True],
    # [4096, 4096, 64, 128, 256, 16, 8, 1, False, True],
])
@pytest.mark.skipif(torch.cuda.get_device_capability()[0] < 9, reason="Requires compute capability >= 9")
def test_user_defined_persistent_non_warp_specialized_gemm(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARPS, NUM_CTAS, TRANS_A, TRANS_B):
    # TODO: fix RewriteTensorPtrPass
    pytest.skip('RewriteTensorPtrPass issue')

    if (TRANS_A):
        a = .1 * torch.randn((K, M), device='cuda', dtype=torch.float16).T
    else:
        a = .1 * torch.randn((M, K), device='cuda', dtype=torch.float16)

    if (TRANS_B):
        b = .1 * torch.randn((N, K), device='cuda', dtype=torch.float16).T
    else:
        b = .1 * torch.randn((K, N), device='cuda', dtype=torch.float16)
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)

    num_SMs = torch.cuda.get_device_properties('cuda').multi_processor_count
    grid = lambda META: (num_SMs,)

    def call_vintage():
        static_persistent_matmul_kernel[grid](a_ptr=a, b_ptr=b, c_ptr=c, M=M, N=N, K=K, stride_am=a.stride(0), stride_ak=a.stride(1), stride_bk=b.stride(0), stride_bn=b.stride(1), stride_cm=c.stride(0), stride_cn=c.stride(1), BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, NUM_SM=num_SMs, num_warps=NUM_WARPS, num_ctas=NUM_CTAS)
        return c

    def call_stylish():
        static_persistent_tma_matmul_kernel[grid](a_ptr=a, b_ptr=b, c_ptr=c, M=M, N=N, K=K, stride_am=a.stride(0), stride_ak=a.stride(1), stride_bk=b.stride(0), stride_bn=b.stride(1), stride_cm=c.stride(0), stride_cn=c.stride(1), BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, NUM_SM=num_SMs, num_warps=NUM_WARPS, num_ctas=NUM_CTAS)
        return c

    th_c = torch.matmul(a, b)

    # Test using old style of ptr calculation
    tt_c = call_vintage()
    torch.testing.assert_allclose(th_c, tt_c, atol=1e-2, rtol=0)

    # Cealr c
    c = torch.randn((M, N), device=a.device, dtype=torch.float32)

    # Test using make_block_ptr
    tt_c = call_stylish()
    torch.testing.assert_allclose(th_c, tt_c, atol=1e-2, rtol=0)


@triton.autotune(
    configs=[
        triton.Config({}, num_stages=3, num_warps=4, enable_warp_specialization=True),
        triton.Config({}, num_stages=3, num_warps=4, enable_warp_specialization=False),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def warp_specialized_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
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


@triton.autotune(
    configs=[
        triton.Config({}, num_stages=3, num_warps=4, enable_warp_specialization=True),
        triton.Config({}, num_stages=3, num_warps=4, enable_warp_specialization=False),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def tma_warp_specialized_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
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


@pytest.mark.parametrize('M,N,K,BLOCK_M,BLOCK_N,BLOCK_K,NUM_CTAS,TRANS_A,TRANS_B', [
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
    # TODO: the following cases fail for warp specialization
    # [4096, 4096, 256, 128, 256, 16, 1, False, True],
    # [4096, 4096, 256, 128, 256, 64, 1, False, True],
])
@pytest.mark.skipif(torch.cuda.get_device_capability()[0] < 9, reason="Requires compute capability >= 9")
def test_non_persistent_warp_specialized_gemm(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, NUM_CTAS, TRANS_A, TRANS_B):
    pytest.skip('hang')

    if (TRANS_A):
        a = .1 * torch.randn((K, M), device='cuda', dtype=torch.float16).T
    else:
        a = .1 * torch.randn((M, K), device='cuda', dtype=torch.float16)

    if (TRANS_B):
        b = .1 * torch.randn((N, K), device='cuda', dtype=torch.float16).T
    else:
        b = .1 * torch.randn((K, N), device='cuda', dtype=torch.float16)

    c = torch.empty((M, N), device=a.device, dtype=torch.float32)

    grid = lambda META: (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    def call_vintage():
        warp_specialized_matmul_kernel[grid](
            a, b, c,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_M, BLOCK_N, BLOCK_K)
        return c

    def call_stylish():
        tma_warp_specialized_matmul_kernel[grid](
            a, b, c,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_M, BLOCK_N, BLOCK_K)
        return c

    th_c = torch.matmul(a, b)

    # Test using old style of ptr calculation
    tt_c = call_vintage()
    torch.testing.assert_allclose(th_c, tt_c, atol=1e-2, rtol=0)

    # Cealr c
    c = torch.randn((M, N), device=a.device, dtype=torch.float32)

    # Test using make_block_ptr
    # TODO: There are some cases failing even in non-warp-specialized way
    fail_cases = [(2048, 2048, 64, 64, 64, 16, 1, False, True),
                  (4096, 4096, 64, 64, 64, 16, 1, False, True),
                  (4096, 4096, 64, 64, 64, 32, 1, False, True),]
    if (M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, NUM_CTAS, TRANS_A, TRANS_B) in fail_cases:
        pytest.skip('Pytest skips case [{}] for tma_warp_specialized_matmul_kernel() since they are known to fail'.format(
            ', '.join(str([M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, NUM_CTAS, TRANS_A, TRANS_B]))))
    tt_c = call_stylish()
    torch.testing.assert_allclose(th_c, tt_c, atol=1e-2, rtol=0)


@triton.autotune(
    configs=[
        triton.Config({}, num_stages=3, num_warps=4, enable_warp_specialization=True),
        triton.Config({}, num_stages=3, num_warps=4, enable_warp_specialization=False),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def static_persistent_warp_specialized_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_SM: tl.constexpr,
):
    start_tile = tl.program_id(axis=0)
    m_tiles = tl.cdiv(M, BLOCK_M)
    n_tiles = tl.cdiv(N, BLOCK_N)
    num_tiles = m_tiles * n_tiles
    offs_k = tl.arange(0, BLOCK_K)

    for tile_id in range(start_tile, num_tiles, NUM_SM):
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


@triton.autotune(
    configs=[
        triton.Config({}, num_stages=3, num_warps=4, enable_warp_specialization=True),
        # triton.Config({}, num_stages=3, num_warps=4, enable_warp_specialization=False),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def static_persistent_tma_warp_specialized_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_SM: tl.constexpr,
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
    a_tile_ptr = tl.make_block_ptr(base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak), offsets=(block_offset_m, 0), block_shape=(BLOCK_M, BLOCK_K), order=(1, 0))
    b_tile_ptr = tl.make_block_ptr(base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn), offsets=(0, block_offset_n), block_shape=(BLOCK_K, BLOCK_N), order=(0, 1))
    for tile_id in range(start_tile, num_tiles, NUM_SM):
        pid_m = tile_id // n_tiles
        pid_n = tile_id % n_tiles
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        if tile_id >= NUM_SM:
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


@pytest.mark.parametrize('M,N,K,BLOCK_M,BLOCK_N,BLOCK_K,NUM_CTAS,TRANS_A,TRANS_B', [
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
    # TODO: the following cases fail for warp specialization
    # [4096, 4096, 256, 128, 256, 16, 1, False, True],
    # [4096, 4096, 256, 128, 256, 64, 1, False, True],
])
@pytest.mark.skipif(torch.cuda.get_device_capability()[0] < 9, reason="Requires compute capability >= 9")
def test_user_defined_persistent_warp_specialized_gemm(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, NUM_CTAS, TRANS_A, TRANS_B):
    # TODO: fix RewriteTensorPtrPass
    pytest.skip('RewriteTensorPtrPass issue')
    if (TRANS_A):
        a = .1 * torch.randn((K, M), device='cuda', dtype=torch.float16).T
    else:
        a = .1 * torch.randn((M, K), device='cuda', dtype=torch.float16)

    if (TRANS_B):
        b = .1 * torch.randn((N, K), device='cuda', dtype=torch.float16).T
    else:
        b = .1 * torch.randn((K, N), device='cuda', dtype=torch.float16)
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)

    num_SMs = torch.cuda.get_device_properties('cuda').multi_processor_count
    grid = lambda META: (num_SMs,)

    def call_vintage():
        static_persistent_warp_specialized_matmul_kernel[grid](
            a, b, c,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_M, BLOCK_N, BLOCK_K, num_SMs)
        return c

    def call_stylish():
        static_persistent_tma_warp_specialized_matmul_kernel[grid](
            a, b, c,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_M, BLOCK_N, BLOCK_K, num_SMs)
        return c

    th_c = torch.matmul(a, b)

    # Test using old style of ptr calculation
    tt_c = call_vintage()
    torch.testing.assert_allclose(th_c, tt_c, atol=1e-2, rtol=0)

    # Cealr c
    c = torch.randn((M, N), device=a.device, dtype=torch.float32)

    # Test using make_block_ptr
    tt_c = call_stylish()
    torch.testing.assert_allclose(th_c, tt_c, atol=1e-2, rtol=0)
