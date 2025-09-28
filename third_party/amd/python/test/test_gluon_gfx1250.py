# ruff: noqa: E402
import hip

hip.hip.hipInit(0)
# Needed for internal dev flow for now; will remove later

import re
import pytest
import torch

import triton
from triton.backends.compiler import GPUTarget
from triton._internal_testing import str_to_triton_dtype
from triton.experimental import gluon
import triton.experimental.gluon.language as ttgl


@gluon.jit
def gemm_kernel(a_ptr, b_ptr, c_ptr,  #
                M, N, K,  #
                stride_am, stride_ak,  #
                stride_bk, stride_bn,  #
                stride_cm, stride_cn,  #
                BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr, BLOCK_K: ttgl.constexpr,  #
                INSTR_SHAPE_K: ttgl.constexpr, K_WIDTH: ttgl.constexpr):

    BLOCKED_LAYOUT: ttgl.constexpr = ttgl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0])
    WMMA_LAYOUT: ttgl.constexpr = ttgl.amd.AMDWMMALayout(3, True, [2, 2], [16, 16, INSTR_SHAPE_K])

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

    accumulator = ttgl.zeros((BLOCK_M, BLOCK_N), dtype=c_ptr.type.element_ty, layout=WMMA_LAYOUT)
    for k in range(0, ttgl.cdiv(K, BLOCK_K)):
        mask_a = (offs_ak[None, :] < K - k * BLOCK_K) & (offs_am[:, None] < M)
        mask_b = (offs_bk[:, None] < K - k * BLOCK_K) & (offs_bn[None, :] < N)

        a = ttgl.load(a_ptr + offs_a, mask=mask_a, other=0.0)
        b = ttgl.load(b_ptr + offs_b, mask=mask_b, other=0.0)

        a = ttgl.convert_layout(a, ttgl.DotOperandLayout(0, WMMA_LAYOUT, K_WIDTH))
        b = ttgl.convert_layout(b, ttgl.DotOperandLayout(1, WMMA_LAYOUT, K_WIDTH))
        accumulator = ttgl.amd.gfx1250.wmma(a, b, accumulator)

        offs_a += BLOCK_K * stride_ak
        offs_b += BLOCK_K * stride_bk

    offs_cm = pid_m * BLOCK_M + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, WMMA_LAYOUT))
    offs_cn = pid_n * BLOCK_N + ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, WMMA_LAYOUT))
    offs_c = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    ttgl.store(c_ptr + offs_c, accumulator, mask=mask_c)


def get_test_gemm_block_mnk():
    return [
        (m, n, k) for (m, n) in [(32, 32), (64, 64)] \
                  for k in [32, 64, 128, 256]
    ]


def get_test_gemm_variants():
    return  [
        # float32 * float32 -> float32
        ("float32", "float32", 4),
        # bfloat16/float16 * bfloat16/float16 -> float32
        *[(a, a, 32) for a in ["bfloat16", "float16"]],
        # float8e4m3/float8e5m2 * float8e4m3/float8e5m2 -> float32/float16
        *[(a, b, k) for a in ["float8_e4m3fn", "float8_e5m2"] \
                       for b in ["float8_e4m3fn", "float8_e5m2"] \
                       for k in [64, 128]],
    ]


def get_test_gemm_shapes():
    return [
        (256, 256, 256),
        (250, 250, 250),
    ]


@pytest.mark.parametrize("BLOCK_M,BLOCK_N,BLOCK_K", get_test_gemm_block_mnk())
@pytest.mark.parametrize("a_dtype,b_dtype,k_dim", get_test_gemm_variants())
def test_compile_gemm(BLOCK_M, BLOCK_N, BLOCK_K, a_dtype, b_dtype, k_dim):
    if BLOCK_K < k_dim:
        pytest.skip("Skip tests where BLOCK_K < k_dim")

    a_dtype = str_to_triton_dtype(a_dtype).name
    b_dtype = str_to_triton_dtype(b_dtype).name

    k = triton.compile(
        gluon._runtime.GluonASTSource(
            fn=gemm_kernel, signature={
                "a_ptr": f"*{a_dtype}", "b_ptr": f"*{b_dtype}", "c_ptr": "*fp32",  #
                "M": "i32", "N": "i32", "K": "i32",  #
                "stride_am": "i32", "stride_ak": "i32",  #
                "stride_bk": "i32", "stride_bn": "i32",  #
                "stride_cm": "i32", "stride_cn": "i32",  #
                "BLOCK_M": "constexpr", "BLOCK_N": "constexpr", "BLOCK_K": "constexpr",  #
                "INSTR_SHAPE_K": "constexpr", "K_WIDTH": "constexpr"
            }, constexprs={
                "BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "BLOCK_K": BLOCK_K,  #
                "INSTR_SHAPE_K": k_dim, "K_WIDTH": 2 if a_dtype == "fp32" else 8
            }), target=GPUTarget("hip", 'gfx1250', 32))
    amdgcn = k.asm["amdgcn"]

    wmma_pattern = "v_wmma_"
    wmma_pattern += "f32_"
    wmma_pattern += "16x16x" + str(k_dim) + "_"
    if a_dtype == "fp32":
        wmma_pattern += "f32"
    if a_dtype in ("fp16", "bf16"):
        a_ty = "f16" if a_dtype == "fp16" else "bf16"
        wmma_pattern += a_ty
    if a_dtype in ("fp8e4nv", "fp8e5"):
        a_ty = "fp8" if a_dtype == "fp8e4nv" else "bf8"
        b_ty = "fp8" if b_dtype == "fp8e4nv" else "bf8"
        wmma_pattern += a_ty + "_" + b_ty

    assert re.search(wmma_pattern, amdgcn), "The AMDGCN assembly does not contain the expected WMMA instruction."


@pytest.mark.parametrize("M,N,K", get_test_gemm_shapes())
@pytest.mark.parametrize("BLOCK_M,BLOCK_N,BLOCK_K", get_test_gemm_block_mnk())
@pytest.mark.parametrize("a_dtype,b_dtype,k_dim", get_test_gemm_variants())
def test_runtime_gemm(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, a_dtype, b_dtype, k_dim):
    if BLOCK_K < k_dim:
        pytest.skip("Skip tests where BLOCK_K < k_dim")
    if a_dtype == 'float8_e4m3fn' or b_dtype == 'float8_e4m3fn':
        pytest.skip("Skip float8_e4m3fn tests for now due to accuracy issue")

    torch.manual_seed(42)

    def create_operand(shape, dtype):
        if dtype in (torch.float16, torch.bfloat16, torch.float32):
            return torch.randn(shape, dtype=dtype)
        elif dtype == torch.float8_e5m2:
            # range from min normal (0 00001 00) to max normal (0 11110 11)
            return torch.randint(0x04, 0x7B, shape, dtype=torch.uint8).view(dtype)
        else:
            # range from min normal (0 0001 000) to max normal (0 1110 111)
            assert dtype == torch.float8_e4m3fn
            return torch.randint(0x08, 0x77, shape, dtype=torch.uint8).view(dtype)

    a_dtype = getattr(torch, a_dtype)
    b_dtype = getattr(torch, b_dtype)

    a = create_operand((M, K), a_dtype)
    b = create_operand((K, N), b_dtype)
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
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,  #
        INSTR_SHAPE_K=k_dim, K_WIDTH=2 if a_dtype == torch.float32 else 8)

    c_triton = c_device.cpu()
    c_torch = a.to(torch.float32) @ b.to(torch.float32)
    torch.testing.assert_close(c_triton, c_torch, rtol=1e-4, atol=1e-4)
