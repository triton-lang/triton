# Copyright (c) 2025, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.
# Licensed under the MIT License

import pytest
import torch
import os

import triton
import triton.language as tl
from torch.testing import assert_close

torch.manual_seed(0)


@pytest.mark.parametrize('M, N, K, AT, BT, ACol, BCol, num_warps, disable_sme, dataType',
                         [(M, N, K, AT, BT, ACol, BCol, num_warps, disable_sme, dataType)
                          for M in [32, 64, 128]
                          for N in [32, 64]
                          for K in [32, 64]
                          for AT in [False, True]
                          for BT in [False, True]
                          for ACol in [False, True]
                          for BCol in [False, True]
                          for num_warps in [1, 2, 4]
                          for disable_sme in ["0", "1"]
                          for dataType in ["float16", "bfloat16"]])
def test_sme_and_swizzle_layout_trans(M, N, K, AT, BT, ACol, BCol, num_warps, disable_sme, dataType, device='cuda'):

    @triton.jit
    def kernel(
        A,
        B,
        C,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        A_T: tl.constexpr,
        B_T: tl.constexpr,
    ):
        off_m = tl.arange(0, BLOCK_M)
        off_mk = tl.arange(0, BLOCK_K)
        if A_T:
            off_m = tl.arange(0, BLOCK_K)
            off_mk = tl.arange(0, BLOCK_M)
        off_n = tl.arange(0, BLOCK_N)
        off_nk = tl.arange(0, BLOCK_K)
        if B_T:
            off_n = tl.arange(0, BLOCK_K)
            off_nk = tl.arange(0, BLOCK_N)
        off_cm = tl.arange(0, BLOCK_M)
        off_cn = tl.arange(0, BLOCK_N)
        a = A + off_m[:, None] * stride_am + off_mk[None, :] * stride_ak
        b = B + off_nk[:, None] * stride_bk + off_n[None, :] * stride_bn
        C = C + off_cm[:, None] * stride_cm + off_cn[None, :] * stride_cn
        x = tl.load(a)
        y = tl.load(b)
        if A_T:
            x = tl.trans(x)
        if B_T:
            y = tl.trans(y)
        z = tl.dot(x, y)
        tl.store(C, z)

    os.environ['TRITON_DISABLE_SME'] = disable_sme  #when disable_sme=1, this test swizzle trans
    #run test
    dataType = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dataType]
    a = .1 * torch.randn((K, M) if (AT ^ ACol) else (M, K), device='cuda', dtype=dataType)
    b = .1 * torch.randn((N, K) if (BT ^ BCol) else (K, N), device='cuda', dtype=dataType)

    tt_c = .1 * torch.randn((M, N), device='cuda', dtype=dataType)
    tt_a = a
    tt_b = b

    if ACol:
        tt_a = a.t()
    if BCol:
        tt_b = b.t()

    # triton result
    kernel[(1, 1)](tt_a, tt_b, tt_c, tt_a.stride(0), tt_a.stride(1), tt_b.stride(0), tt_b.stride(1), tt_c.stride(0),
                   tt_c.stride(1), BLOCK_M=M, BLOCK_N=N, BLOCK_K=K, A_T=AT, B_T=BT, num_warps=num_warps)

    th_a = a.t() if (AT ^ ACol) else a
    th_b = b.t() if (BT ^ BCol) else b
    #torch result
    th_c = torch.matmul(th_a, th_b)
    assert_close(tt_c, th_c, atol=1e-2, rtol=0)


@pytest.mark.parametrize('M, N, K, AT, BT, CT, num_warps, dataType', [(M, N, K, AT, BT, CT, num_warps, dataType)
                                                                      for M in [32, 64, 128]
                                                                      for N in [32, 64]
                                                                      for K in [32, 64]
                                                                      for AT in [False, True]
                                                                      for BT in [False, True]
                                                                      for CT in [False, True]
                                                                      for num_warps in [1, 2, 4]
                                                                      for dataType in ["float16", "bfloat16"]])
def test_multi_dot_trans(M, N, K, AT, BT, CT, num_warps, dataType, device='cuda'):

    @triton.jit
    def kernel(
        A,
        B,
        C,
        D,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        stride_dm,
        stride_dn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        A_T: tl.constexpr,
        B_T: tl.constexpr,
        C_T: tl.constexpr,
    ):
        off_m = tl.arange(0, BLOCK_M)
        off_mk = tl.arange(0, BLOCK_K)
        if A_T:
            off_m = tl.arange(0, BLOCK_K)
            off_mk = tl.arange(0, BLOCK_M)
        off_n = tl.arange(0, BLOCK_N)
        off_nk = tl.arange(0, BLOCK_K)
        if B_T:
            off_n = tl.arange(0, BLOCK_K)
            off_nk = tl.arange(0, BLOCK_N)
        off_cm = tl.arange(0, BLOCK_M)
        off_cn = tl.arange(0, BLOCK_N)
        if C_T:
            off_cm = tl.arange(0, BLOCK_N)
            off_cn = tl.arange(0, BLOCK_M)
        off_dn = tl.arange(0, BLOCK_N)
        a = A + off_m[:, None] * stride_am + off_mk[None, :] * stride_ak
        b = B + off_nk[:, None] * stride_bk + off_n[None, :] * stride_bn
        c = C + off_cm[:, None] * stride_cm + off_cn[None, :] * stride_cn
        x = tl.load(a)
        y = tl.load(b)
        w = tl.load(c)
        if A_T:
            x = tl.trans(x)
        if B_T:
            y = tl.trans(y)
        if C_T:
            w = tl.trans(w)
        z = tl.dot(x, y)
        z = z.to(C.dtype.element_ty)
        p = tl.dot(tl.trans(z), w)
        D = D + off_dn[:, None] * stride_dm + off_dn[None, :] * stride_dn
        tl.store(D, p)

    #run test
    dataType = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dataType]
    a = .1 * torch.randn((K, M) if AT else (M, K), device='cuda', dtype=dataType)
    b = .1 * torch.randn((N, K) if BT else (K, N), device='cuda', dtype=dataType)
    c = .1 * torch.randn((N, M) if CT else (M, N), device='cuda', dtype=dataType)
    d = .1 * torch.randn((N, N), device='cuda', dtype=dataType)
    # triton result
    kernel[(1, 1)](a, b, c,
                   d, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), d.stride(0),
                   d.stride(1), BLOCK_M=M, BLOCK_N=N, BLOCK_K=K, A_T=AT, B_T=BT, C_T=CT, num_warps=num_warps)
    ta = a.t() if AT else a
    tb = b.t() if BT else b
    tc = c.t() if CT else c
    #torch result
    th_c = torch.matmul(torch.matmul(ta, tb).t(), tc)
    assert_close(d, th_c, atol=1e-2, rtol=0)
