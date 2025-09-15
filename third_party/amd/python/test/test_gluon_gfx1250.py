# ruff: noqa: E402
import hip

hip.hip.hipInit(0)
# Needed for internal dev flow for now; will remove later

import re
import pytest
import torch

import triton
from triton.backends.compiler import GPUTarget
from triton.experimental import gluon
import triton.experimental.gluon.language as ttgl


@gluon.jit
def gemm_kernel(a_ptr, b_ptr, c_ptr,  #
                M, N, K,  #
                stride_am, stride_ak,  #
                stride_bk, stride_bn,  #
                stride_cm, stride_cn,  #
                BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr, BLOCK_K: ttgl.constexpr):

    BLOCKED_LAYOUT: ttgl.constexpr = ttgl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0])
    WMMA_LAYOUT: ttgl.constexpr = ttgl.amd.AMDWMMALayout(3, True, [2, 2], [16, 16, 32])

    pid = ttgl.program_id(axis=0)
    num_pid_m = ttgl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    offs_am = pid_m * BLOCK_M + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, BLOCKED_LAYOUT))
    offs_ak = ttgl.arange(0, BLOCK_K, layout=ttgl.SliceLayout(0, BLOCKED_LAYOUT))
    offs_a = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak

    offs_bk = ttgl.arange(0, BLOCK_K, layout=ttgl.SliceLayout(1, BLOCKED_LAYOUT))
    offs_bn = pid_n * BLOCK_N + ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, BLOCKED_LAYOUT))
    offs_b = offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    accumulator = ttgl.zeros((BLOCK_M, BLOCK_N), dtype=ttgl.float32, layout=WMMA_LAYOUT)
    for k in range(0, ttgl.cdiv(K, BLOCK_K)):
        mask_a = (offs_ak[None, :] < K - k * BLOCK_K) & (offs_am[:, None] < M)
        mask_b = (offs_bk[:, None] < K - k * BLOCK_K) & (offs_bn[None, :] < N)

        a = ttgl.load(a_ptr + offs_a, mask=mask_a, other=0.0)
        b = ttgl.load(b_ptr + offs_b, mask=mask_b, other=0.0)

        a = ttgl.convert_layout(a, ttgl.DotOperandLayout(0, WMMA_LAYOUT, 8))
        b = ttgl.convert_layout(b, ttgl.DotOperandLayout(1, WMMA_LAYOUT, 8))

        accumulator = ttgl.amd.gfx1250.wmma(a, b, accumulator)

        offs_a += BLOCK_K * stride_ak
        offs_b += BLOCK_K * stride_bk

    offs_cm = pid_m * BLOCK_M + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, WMMA_LAYOUT))
    offs_cn = pid_n * BLOCK_N + ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, WMMA_LAYOUT))
    offs_c = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    ttgl.store(c_ptr + offs_c, accumulator, mask=mask_c)


@pytest.mark.parametrize("BLOCK_M,BLOCK_N,BLOCK_K", [(32, 32, 32), (64, 64, 64), (128, 128, 64)])
def test_compile_gemm_bf16(BLOCK_M, BLOCK_N, BLOCK_K):
    k = triton.compile(
        gluon._runtime.GluonASTSource(
            fn=gemm_kernel, signature={
                "a_ptr": "*bf16", "b_ptr": "*bf16", "c_ptr": "*fp32", "M": "i32", "N": "i32", "K": "i32", "stride_am":
                "i32", "stride_ak": "i32", "stride_bk": "i32", "stride_bn": "i32", "stride_cm": "i32", "stride_cn":
                "i32", "BLOCK_M": "constexpr", "BLOCK_N": "constexpr", "BLOCK_K": "constexpr"
            }, constexprs={"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "BLOCK_K": BLOCK_K}),
        target=GPUTarget("hip", 'gfx1250', 32))

    amdgcn = k.asm["amdgcn"]
    pattern = r"v_wmma_f32_16x16x32_bf16"
    assert re.search(pattern, amdgcn), "The AMDGCN assembly does not contain the expected WMMA instruction."


@pytest.mark.parametrize("M,N,K", [
    (128, 128, 64),
    (256, 256, 128),
    (256, 128, 64),
    (128, 256, 64),
    (120, 120, 60),
])
@pytest.mark.parametrize("BLOCK_M,BLOCK_N,BLOCK_K", [(32, 32, 32), (64, 64, 64), (128, 128, 64)])
def test_runtime_gemm_bf16(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K):
    torch.manual_seed(42)
    a = torch.randn((M, K), dtype=torch.bfloat16)
    b = torch.randn((K, N), dtype=torch.bfloat16)
    c = torch.zeros((M, N), dtype=torch.float32)
    stride_am, stride_ak = a.stride(0), a.stride(1)
    stride_bk, stride_bn = b.stride(0), b.stride(1)
    stride_cm, stride_cn = c.stride(0), c.stride(1)

    a_device = a.cuda()
    b_device = b.cuda()
    c_device = c.cuda()
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1)
    gemm_kernel[grid](
        a_device, b_device, c_device,  #
        M, N, K,  #
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,  #
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)

    c_triton = c_device.cpu()
    c_torch = a.to(torch.float32) @ b.to(torch.float32)
    torch.testing.assert_close(c_triton, c_torch, rtol=1e-4, atol=1e-4)
