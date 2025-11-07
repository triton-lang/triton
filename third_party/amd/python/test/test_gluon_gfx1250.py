# ruff: noqa: E402
import hip

hip.hip.hipInit(0)
# Needed for internal dev flow for now; will remove later

import re
import pytest
import torch

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton._internal_testing import str_to_triton_dtype
from triton._internal_testing import is_hip_gfx1250
from triton.tools.mxfp import MXFP4Tensor, MXScaleTensor
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
        # NOTE: we always use transposed=True for wmma layout, which will swap A and B
        wmma_pattern += b_ty + "_" + a_ty
    assert re.search(wmma_pattern, amdgcn)


@pytest.mark.parametrize("M,N,K", get_test_gemm_shapes())
@pytest.mark.parametrize("BLOCK_M,BLOCK_N,BLOCK_K", get_test_gemm_block_mnk())
@pytest.mark.parametrize("a_dtype,b_dtype,k_dim", get_test_gemm_variants())
def test_runtime_gemm(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, a_dtype, b_dtype, k_dim):
    if BLOCK_K < k_dim:
        pytest.skip("Skip tests where BLOCK_K < k_dim")

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


@gluon.jit
def dot_mxfp_gluon_kernel(a_base, stride_am, stride_ak, a_scale, b_base, stride_bk, stride_bn, b_scale, out,
                          BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr, BLOCK_K: ttgl.constexpr,
                          type_a: ttgl.constexpr, type_b: ttgl.constexpr):
    DIV_FACTOR_A: ttgl.constexpr = 2 if type_a == "e2m1" else 1
    DIV_FACTOR_B: ttgl.constexpr = 2 if type_b == "e2m1" else 1
    PACKED_BLOCK_K_A: ttgl.constexpr = BLOCK_K // DIV_FACTOR_A
    PACKED_BLOCK_K_B: ttgl.constexpr = BLOCK_K // DIV_FACTOR_B
    SCALE_BLOCK_K: ttgl.constexpr = BLOCK_K // 32

    scale_blocked_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [8, 4], [4, 1], [1, 0])
    a_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 16], [8, 4], [4, 1], [1, 0])
    a_scale_linear_layout: ttgl.constexpr = ttgl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2]], lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [0, 0]], warp_bases=[[0, 0], [16, 0]],
        block_bases=[], shape=[32, 4])
    b_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 16], [16, 2], [4, 1], [1, 0])
    b_scale_linear_layout: ttgl.constexpr = ttgl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2]], lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [0, 0]], warp_bases=[[16, 0], [0, 0]],
        block_bases=[], shape=[32, 4])

    wmma_layout: ttgl.constexpr = ttgl.amd.AMDWMMALayout(version=3, transposed=True, warps_per_cta=[2, 2],
                                                         instr_shape=[16, 16, 128])
    wmma_layout_packed: ttgl.constexpr = ttgl.amd.AMDWMMALayout(version=3, transposed=True, warps_per_cta=[2, 2],
                                                                instr_shape=[16, 16, 64])

    zero = ttgl.zeros([BLOCK_M, BLOCK_N], dtype=ttgl.float32, layout=wmma_layout)

    offs_am = ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, a_layout))
    offs_ak = ttgl.arange(0, PACKED_BLOCK_K_A, layout=ttgl.SliceLayout(0, a_layout))
    a_offsets = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
    a = ttgl.load(a_base + a_offsets)
    a = ttgl.convert_layout(
        a,
        ttgl.DotOperandLayout(operand_index=0, parent=wmma_layout_packed if type_a == "e2m1" else wmma_layout,
                              k_width=16))

    offs_bk = ttgl.arange(0, PACKED_BLOCK_K_B, layout=ttgl.SliceLayout(1, b_layout))
    offs_bn = ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, b_layout))
    b_offsets = offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    b = ttgl.load(b_base + b_offsets)
    b = ttgl.convert_layout(
        b,
        ttgl.DotOperandLayout(operand_index=1, parent=wmma_layout_packed if type_b == "e2m1" else wmma_layout,
                              k_width=16))

    if a_scale is not None:
        offs_scale_am = ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, scale_blocked_layout))
        off_scale_ak = ttgl.arange(0, SCALE_BLOCK_K, layout=ttgl.SliceLayout(0, scale_blocked_layout))
        a_scale_offsets = offs_scale_am[:, None] * SCALE_BLOCK_K + off_scale_ak[None, :]
        scale_a = ttgl.load(a_scale + a_scale_offsets)
    else:
        scale_a = ttgl.full([BLOCK_M, SCALE_BLOCK_K], 127, dtype=ttgl.int8, layout=scale_blocked_layout)

    if b_scale is not None:
        offs_scale_bn = ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(1, scale_blocked_layout))
        offs_scale_bk = ttgl.arange(0, SCALE_BLOCK_K, layout=ttgl.SliceLayout(0, scale_blocked_layout))
        b_scale_offsets = offs_scale_bn[:, None] * SCALE_BLOCK_K + offs_scale_bk[None, :]
        scale_b = ttgl.load(b_scale + b_scale_offsets)
    else:
        scale_b = ttgl.full([BLOCK_N, SCALE_BLOCK_K], 127, dtype=ttgl.int8, layout=scale_blocked_layout)

    scale_a = ttgl.convert_layout(scale_a, a_scale_linear_layout)
    scale_b = ttgl.convert_layout(scale_b, b_scale_linear_layout)
    c = ttgl.amd.gfx1250.wmma_scaled(a, scale_a, type_a, b, scale_b, type_b, zero)
    c = c.to(out.dtype.element_ty)

    offs_cm = ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, wmma_layout))
    offs_cn = ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, wmma_layout))
    out_offsets = offs_cm[:, None] * BLOCK_N + offs_cn[None, :]
    out = out + out_offsets
    ttgl.store(out, c)


def torch_gemm_mxfp(a, b, a_scale, b_scale, scale_block, M, N, K):
    a_scale_f32 = a_scale.to(torch.float32).repeat_interleave(scale_block, dim=1)[:M, :K]
    b_scale_f32 = b_scale.to(torch.float32).repeat_interleave(scale_block, dim=1).T.contiguous()[:K, :N]

    a_f32 = a.to(torch.float32)
    b_f32 = b.to(torch.float32)

    return torch.matmul(a_f32 * a_scale_f32, b_f32 * b_scale_f32).to(torch.float32)


@pytest.mark.parametrize("BLOCK_M, BLOCK_N, BLOCK_K", [(16, 16, 128), (32, 32, 128), (32, 32, 256), (32, 32, 512),
                                                       (64, 64, 128), (128, 128, 256)])
@pytest.mark.parametrize("mxfp_type", ["e2m1"])
@pytest.mark.parametrize("hasScale", [True, False])
def test_compile_amd_wmma_scaled(BLOCK_M, BLOCK_N, BLOCK_K, mxfp_type, hasScale):
    k = triton.compile(
        gluon._runtime.GluonASTSource(
            fn=dot_mxfp_gluon_kernel, signature={
                "a_base": "*u8", "stride_am": "i32", "stride_ak": "i32", "a_scale": "*u8", "b_base": "*u8", "stride_bk":
                "i32", "stride_bn": "i32", "b_scale": "*u8", "out": "*fp32", "BLOCK_M": "constexpr", "BLOCK_N":
                "constexpr", "BLOCK_K": "constexpr", "type_a": "constexpr", "type_b": "constexpr"
            }, constexprs={
                "BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "BLOCK_K": BLOCK_K, "type_a": mxfp_type, "type_b": mxfp_type
            }), target=GPUTarget("hip", 'gfx1250', 32))
    amdgcn = k.asm["amdgcn"]
    assert "v_wmma_scale_f32_16x16x128_f8f6f4" in amdgcn, "The AMDGCN assembly does not contain the expected scaled WMMA instruction."


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires GFX1250")
@pytest.mark.parametrize("BLOCK_M, BLOCK_N, BLOCK_K", [(16, 16, 128), (32, 32, 128), (32, 32, 256), (32, 32, 512),
                                                       (64, 64, 128), (128, 128, 256)])
@pytest.mark.parametrize("mxfp_type", ["e2m1"])
@pytest.mark.parametrize("hasScale", [True, False])
def test_runtime_amd_wmma_scaled(BLOCK_M, BLOCK_N, BLOCK_K, mxfp_type, hasScale):

    @triton.jit
    def dot_mxfp_triton_kernel(a_base, stride_am, stride_ak, a_scale, b_base, stride_bk, stride_bn, b_scale, out,
                               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                               type_a: tl.constexpr, type_b: tl.constexpr):
        DIV_FACTOR_A: tl.constexpr = 2 if type_a == "e2m1" else 1
        DIV_FACTOR_B: tl.constexpr = 2 if type_b == "e2m1" else 1
        PACKED_BLOCK_K_A: tl.constexpr = BLOCK_K // DIV_FACTOR_A
        PACKED_BLOCK_K_B: tl.constexpr = BLOCK_K // DIV_FACTOR_B
        a_ptr = a_base + tl.arange(0, BLOCK_M)[:, None] * stride_am + \
                tl.arange(0, PACKED_BLOCK_K_A)[None, :] * stride_ak
        b_ptr = b_base + tl.arange(0, PACKED_BLOCK_K_B)[:, None] * stride_bk + \
                tl.arange(0, BLOCK_N)[None, :] * stride_bn

        a = tl.load(a_ptr)
        b = tl.load(b_ptr)
        SCALE_BLOCK_K: tl.constexpr = BLOCK_K // 32

        if a_scale is not None:
            scale_a_ptr = a_scale + tl.arange(0, BLOCK_M)[:, None] * SCALE_BLOCK_K + tl.arange(0,
                                                                                               SCALE_BLOCK_K)[None, :]
            a_scale = tl.load(scale_a_ptr)
        if b_scale is not None:
            scale_b_ptr = b_scale + tl.arange(0, BLOCK_N)[:, None] * SCALE_BLOCK_K + tl.arange(0,
                                                                                               SCALE_BLOCK_K)[None, :]
            b_scale = tl.load(scale_b_ptr)
        c = tl.dot_scaled(a, a_scale, type_a, b, b_scale, type_b)
        out_ptr = out + tl.arange(0, BLOCK_M)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
        tl.store(out_ptr, c)

    def torch_gemm_mxfp(a, b, a_scale, b_scale, scale_block, M, N, K):
        a_scale_f32 = a_scale.to(torch.float32).repeat_interleave(scale_block, dim=1)[:M, :K]
        b_scale_f32 = b_scale.to(torch.float32).repeat_interleave(scale_block, dim=1).T.contiguous()[:K, :N]

        a_f32 = a.to(torch.float32)
        b_f32 = b.to(torch.float32)

        return torch.matmul(a_f32 * a_scale_f32, b_f32 * b_scale_f32).to(torch.float32)

    torch.manual_seed(0)

    type_a = mxfp_type
    type_b = mxfp_type

    a_mxfp4 = MXFP4Tensor(size=(BLOCK_M, BLOCK_K)).random()
    b_mxfp4 = MXFP4Tensor(size=(BLOCK_K, BLOCK_N)).random()

    scale_a_size = (BLOCK_M, (BLOCK_K + 32 - 1) // 32)
    scale_b_size = (BLOCK_N, (BLOCK_K + 32 - 1) // 32)

    if hasScale:
        scale_a_mxfp4 = MXScaleTensor(size=scale_a_size).random(high=32.0)
        scale_b_mxfp4 = MXScaleTensor(size=scale_b_size).random(high=32.0)
    else:
        scale_a_mxfp4 = torch.ones(scale_a_size, dtype=torch.float32)
        scale_b_mxfp4 = torch.ones(scale_b_size, dtype=torch.float32)

    c_torch = torch_gemm_mxfp(a_mxfp4, b_mxfp4, scale_a_mxfp4, scale_b_mxfp4, 32, BLOCK_M, BLOCK_N, BLOCK_K)

    a = a_mxfp4.to_packed_tensor(dim=1).data.contiguous().cuda()
    b = b_mxfp4.to_packed_tensor(dim=0).data.contiguous().cuda()

    if hasScale:
        scale_a = scale_a_mxfp4.data.cuda()
        scale_b = scale_b_mxfp4.data.cuda()
    else:
        scale_a = None
        scale_b = None

    c = torch.zeros((BLOCK_M, BLOCK_N), dtype=torch.float32).cuda()
    pgm = dot_mxfp_gluon_kernel[(1, )](a, *a.stride(), scale_a, b, *b.stride(), scale_b, c, BLOCK_M, BLOCK_N, BLOCK_K,
                                       type_a, type_b)
    assert "v_wmma_scale_f32_16x16x128_f8f6f4" in pgm.asm[
        "amdgcn"], "The AMDGCN assembly does not contain the expected scaled WMMA instruction."

    c_ref = torch.zeros((BLOCK_M, BLOCK_N), dtype=torch.float32).cuda()
    dot_mxfp_triton_kernel[(1, )](a, *a.stride(), scale_a, b, *b.stride(), scale_b, c_ref, BLOCK_M, BLOCK_N, BLOCK_K,
                                  type_a, type_b)

    torch.testing.assert_close(c.cpu(), c_ref.cpu(), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(c.cpu(), c_torch, rtol=1e-5, atol=1e-5)


@gluon.jit
def tensor_copy_kernel(a_ptr, b_ptr, M, N,  #
                       BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr, NUM_BUFFERS: ttgl.constexpr,
                       BLOCKED_LAYOUT: ttgl.constexpr):
    SHARED_LAYOUT: ttgl.constexpr = ttgl.PaddedSharedLayout.with_identity_for([[32, 4]], [BLOCK_M, BLOCK_N], [1, 0])

    pid = ttgl.program_id(axis=0)
    num_pid_m = ttgl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    a_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(base=a_ptr, shape=(M, N), strides=(N, 1),
                                                         block_shape=(BLOCK_M, BLOCK_N), layout=SHARED_LAYOUT)
    a_buffer = ttgl.allocate_shared_memory(a_desc.dtype, [NUM_BUFFERS] + a_desc.block_shape, a_desc.layout)

    idx_m = pid_m * BLOCK_M
    for i in ttgl.static_range(0, NUM_BUFFERS):
        idx_n = pid_n * (BLOCK_N * NUM_BUFFERS) + i * BLOCK_N
        ttgl.amd.gfx1250.tdm.async_load(a_desc, [idx_m, idx_n], a_buffer.index(i))

    ttgl.amd.gfx1250.tdm.async_wait(0)

    for i in ttgl.static_range(0, NUM_BUFFERS):
        idx_n = pid_n * (BLOCK_N * NUM_BUFFERS) + i * BLOCK_N
        a = a_buffer.index(i).load(layout=BLOCKED_LAYOUT)

        offs_bm = idx_m + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, BLOCKED_LAYOUT))
        offs_bn = idx_n + ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, BLOCKED_LAYOUT))
        offs_b = (offs_bm[:, None] * N) + offs_bn[None, :]
        mask_b = (offs_bm[:, None] < M) & (offs_bn[None, :] < N)

        ttgl.store(b_ptr + offs_b, a, mask=mask_b)


@pytest.mark.parametrize("BLOCK_M,BLOCK_N", [(32, 32), (32, 64), (64, 64)])
@pytest.mark.parametrize("NUM_BUFFERS", [1, 2])
def test_compile_tensor_copy(BLOCK_M, BLOCK_N, NUM_BUFFERS):
    BLOCKED_LAYOUT = ttgl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0])
    k = triton.compile(
        gluon._runtime.GluonASTSource(
            fn=tensor_copy_kernel, signature={
                "a_ptr": "*fp16", "b_ptr": "*fp16", "M": "i32", "N": "i32",  #
                "BLOCK_M": "constexpr", "BLOCK_N": "constexpr", "NUM_BUFFERS": "constexpr",  #
                "BLOCKED_LAYOUT": "constexpr"
            }, constexprs={
                "BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "NUM_BUFFERS": NUM_BUFFERS, "BLOCKED_LAYOUT": BLOCKED_LAYOUT
            }), target=GPUTarget("hip", 'gfx1250', 32))

    amdgcn = k.asm["amdgcn"]
    for pattern in ("tensor_load_to_lds", "s_wait_tensorcnt 0x0"):
        assert re.search(pattern, amdgcn)


@pytest.mark.parametrize("BLOCK_M,BLOCK_N", [(32, 32), (32, 64), (64, 64), (1, 512), (256, 2)])
@pytest.mark.parametrize("NUM_BUFFERS", [1, 2])
@pytest.mark.parametrize("NUM_WARPS", [4, 8])
@pytest.mark.parametrize("M,N", [(1024, 1024), (1000, 1000)])
def test_runtime_tensor_copy(M, N, BLOCK_M, BLOCK_N, NUM_BUFFERS, NUM_WARPS):
    blocked_layout = ttgl.BlockedLayout([1, 8], [4, 8], [NUM_WARPS, 1], [1, 0])

    torch.manual_seed(42)
    a = torch.randint(0x0, 0xFFFF, (M, N), dtype=torch.uint16)
    b = torch.zeros_like(a)

    a_device = a.cuda()
    b_device = b.cuda()
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N * NUM_BUFFERS), 1)
    tensor_copy_kernel[grid](a_device, b_device, M, N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, NUM_BUFFERS=NUM_BUFFERS,
                             BLOCKED_LAYOUT=blocked_layout, num_warps=NUM_WARPS)

    b_triton = b_device.cpu()
    assert torch.equal(b_triton, a)


@gluon.jit
def tensor_fill_kernel(a_ptr, M, N, BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr, NUM_BUFFERS: ttgl.constexpr):
    SHARED_LAYOUT: ttgl.constexpr = ttgl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    BLOCKED_LAYOUT: ttgl.constexpr = ttgl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0])

    pid = ttgl.program_id(axis=0)
    num_pid_m = ttgl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    a_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(base=a_ptr, shape=(M, N), strides=(N, 1),
                                                         block_shape=(BLOCK_M, BLOCK_N), layout=SHARED_LAYOUT)
    a_buffer = ttgl.allocate_shared_memory(a_desc.dtype, [NUM_BUFFERS] + a_desc.block_shape, a_desc.layout)

    idx_m = pid_m * BLOCK_M
    for i in ttgl.static_range(0, NUM_BUFFERS):
        idx_n = pid_n * (BLOCK_N * NUM_BUFFERS) + i * BLOCK_N
        vm = idx_m + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, BLOCKED_LAYOUT))
        vn = idx_n + ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, BLOCKED_LAYOUT))
        v = (vm[:, None] * N) + vn[None, :]
        v = v.to(a_desc.dtype)
        a_buffer.index(i).store(v)

    for i in ttgl.static_range(0, NUM_BUFFERS):
        idx_n = pid_n * (BLOCK_N * NUM_BUFFERS) + i * BLOCK_N
        ttgl.amd.gfx1250.tdm.async_store(a_desc, [idx_m, idx_n], a_buffer.index(i))

    ttgl.amd.gfx1250.tdm.async_wait(0)


@pytest.mark.parametrize("BLOCK_M,BLOCK_N", [(32, 32), (32, 64), (64, 64)])
@pytest.mark.parametrize("NUM_BUFFERS", [1, 2])
def test_compile_tensor_fill(BLOCK_M, BLOCK_N, NUM_BUFFERS):
    k = triton.compile(
        gluon._runtime.GluonASTSource(
            fn=tensor_fill_kernel, signature={
                "a_ptr": "*fp16", "M": "i32", "N": "i32",  #
                "BLOCK_M": "constexpr", "BLOCK_N": "constexpr", "NUM_BUFFERS": "constexpr"
            }, constexprs={"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "NUM_BUFFERS": NUM_BUFFERS}),
        target=GPUTarget("hip", 'gfx1250', 32))

    amdgcn = k.asm["amdgcn"]

    for pattern in ("tensor_store_from_lds", "s_wait_tensorcnt 0x0"):
        assert re.search(pattern, amdgcn)


@pytest.mark.parametrize("BLOCK_M,BLOCK_N", [(32, 32), (32, 64), (64, 64)])
@pytest.mark.parametrize("NUM_BUFFERS", [1, 2])
@pytest.mark.parametrize("M,N", [(1024, 1024), (1000, 1000)])
def test_runtime_tensor_fill(M, N, BLOCK_M, BLOCK_N, NUM_BUFFERS):
    a = torch.zeros((M, N), dtype=torch.uint16)

    a_device = a.cuda()
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N * NUM_BUFFERS), 1)
    tensor_fill_kernel[grid](a_device, M, N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, NUM_BUFFERS=NUM_BUFFERS)

    a_triton = a_device.cpu()
    a_ref = torch.arange(M, dtype=torch.int16).unsqueeze(1) * N + \
            torch.arange(N, dtype=torch.int16).unsqueeze(0)
    a_ref = a_ref.to(torch.uint16)
    assert torch.equal(a_triton, a_ref)


@gluon.jit
def mxgemm_kernel(a_ptr, b_ptr, c_ptr, a_scale, b_scale, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm,
                  stride_cn, stride_scale, DTYPE_A: ttgl.constexpr, DTYPE_B: ttgl.constexpr,
                  SCALE_BLOCK: ttgl.constexpr, BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr,
                  BLOCK_K: ttgl.constexpr, GROUP_SIZE_M: ttgl.constexpr):
    DIV_FACTOR_A: ttgl.constexpr = 2 if DTYPE_A == "e2m1" else 1
    DIV_FACTOR_B: ttgl.constexpr = 2 if DTYPE_B == "e2m1" else 1
    BLOCK_K_SCALE: ttgl.constexpr = BLOCK_K // SCALE_BLOCK
    BLOCK_K_PACKED_A: ttgl.constexpr = BLOCK_K // DIV_FACTOR_A
    BLOCK_K_PACKED_B: ttgl.constexpr = BLOCK_K // DIV_FACTOR_B

    BLOCKED_LAYOUT: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [8, 4], [4, 1], [1, 0])
    A_BLOCKED_LAYOUT: ttgl.constexpr = ttgl.BlockedLayout([1, 16], [8, 4], [4, 1], [1, 0])
    B_BLOCKED_LAYOUT: ttgl.constexpr = ttgl.BlockedLayout([1, 16], [16, 2], [4, 1], [1, 0])

    WMMA_LAYOUT: ttgl.constexpr = ttgl.amd.AMDWMMALayout(3, transposed=True, warps_per_cta=[2, 2],
                                                         instr_shape=[16, 16, 128])
    WMMA_LAYOUT_PACKED: ttgl.constexpr = ttgl.amd.AMDWMMALayout(3, transposed=True, warps_per_cta=[2, 2],
                                                                instr_shape=[16, 16, 64])

    DOT_LAYOUT_A: ttgl.constexpr = ttgl.DotOperandLayout(
        operand_index=0, parent=WMMA_LAYOUT_PACKED if DTYPE_A == "e2m1" else WMMA_LAYOUT, k_width=16)
    DOT_LAYOUT_B: ttgl.constexpr = ttgl.DotOperandLayout(
        operand_index=1, parent=WMMA_LAYOUT_PACKED if DTYPE_B == "e2m1" else WMMA_LAYOUT, k_width=16)
    A_SCALE_LINEAR_LAYOUT: ttgl.constexpr = ttgl.amd.gfx1250.get_wmma_scale_layout(DOT_LAYOUT_A,
                                                                                   [BLOCK_M, BLOCK_K_SCALE])
    B_SCALE_LINEAR_LAYOUT: ttgl.constexpr = ttgl.amd.gfx1250.get_wmma_scale_layout(DOT_LAYOUT_B,
                                                                                   [BLOCK_N, BLOCK_K_SCALE])

    pid = ttgl.program_id(axis=0)
    num_pid_m = ttgl.cdiv(M, BLOCK_M)
    num_pid_n = ttgl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, A_BLOCKED_LAYOUT))) % M
    offs_ak = ttgl.arange(0, BLOCK_K_PACKED_A, layout=ttgl.SliceLayout(0, A_BLOCKED_LAYOUT))
    offs_bk = ttgl.arange(0, BLOCK_K_PACKED_B, layout=ttgl.SliceLayout(1, B_BLOCKED_LAYOUT))
    offs_bn = (pid_n * BLOCK_N + ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, B_BLOCKED_LAYOUT))) % N

    offs_scale_am = (pid_m * BLOCK_M + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, BLOCKED_LAYOUT))) % M
    offs_scale_ak = ttgl.arange(0, BLOCK_K_SCALE, layout=ttgl.SliceLayout(0, BLOCKED_LAYOUT))
    offs_scale_bn = (pid_n * BLOCK_N + ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(1, BLOCKED_LAYOUT))) % N
    offs_scale_bk = ttgl.arange(0, BLOCK_K_SCALE, layout=ttgl.SliceLayout(0, BLOCKED_LAYOUT))

    a_scale_ptr = a_scale + offs_scale_am[:, None] * stride_scale + offs_scale_ak[None, :]
    b_scale_ptr = b_scale + offs_scale_bn[:, None] * stride_scale + offs_scale_bk[None, :]
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = ttgl.zeros((BLOCK_M, BLOCK_N), dtype=ttgl.float32, layout=WMMA_LAYOUT)
    for k in range(0, ttgl.cdiv(K, BLOCK_K)):
        k_remaining_a = K - k * BLOCK_K_PACKED_A
        k_remaining_b = K - k * BLOCK_K_PACKED_B
        valid_k_a = offs_ak < k_remaining_a
        valid_k_b = offs_bk < k_remaining_b

        scale_a = ttgl.load(a_scale_ptr)
        scale_b = ttgl.load(b_scale_ptr)
        scale_a = ttgl.convert_layout(scale_a, A_SCALE_LINEAR_LAYOUT)
        scale_b = ttgl.convert_layout(scale_b, B_SCALE_LINEAR_LAYOUT)

        a = ttgl.load(a_ptrs, mask=valid_k_a[None, :], other=0.0)
        b = ttgl.load(b_ptrs, mask=valid_k_b[:, None], other=0.0)
        a = ttgl.convert_layout(a, DOT_LAYOUT_A)
        b = ttgl.convert_layout(b, DOT_LAYOUT_B)

        accumulator = ttgl.amd.gfx1250.wmma_scaled(a, scale_a, DTYPE_A, b, scale_b, DTYPE_B, accumulator)

        a_ptrs += BLOCK_K_PACKED_A * stride_ak
        b_ptrs += BLOCK_K_PACKED_B * stride_bk

        a_scale_ptr += BLOCK_K_SCALE
        b_scale_ptr += BLOCK_K_SCALE

    offs_cm = pid_m * BLOCK_M + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, WMMA_LAYOUT))
    offs_cn = pid_n * BLOCK_N + ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, WMMA_LAYOUT))
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    ttgl.store(c_ptrs, accumulator, mask=c_mask)


@pytest.mark.parametrize("BLOCK_M, BLOCK_N, BLOCK_K", [(32, 32, 64), (32, 32, 128)])
@pytest.mark.parametrize("DTYPE_A", ["float8_e5m2", "float8_e4m3", "float4"])
@pytest.mark.parametrize("DTYPE_B", ["float8_e5m2", "float8_e4m3", "float4"])
def test_compile_mxgemm(BLOCK_M, BLOCK_N, BLOCK_K, DTYPE_A, DTYPE_B):
    scale_block = 32

    if BLOCK_K < 128:
        pytest.skip("NYI: don't support block shape smaller than instr shape")

    triton_dtype_converter = {'float8_e5m2': "fp8e5", "float8_e4m3": "fp8e4nv", "float4": "u8"}
    dot_scaled_dtype_converter = {'float8_e5m2': "e5m2", "float8_e4m3": "e4m3", "float4": "e2m1"}

    k = triton.compile(
        gluon._runtime.GluonASTSource(
            fn=mxgemm_kernel, signature={
                "a_ptr": f"*{triton_dtype_converter[DTYPE_A]}", "b_ptr": f"*{triton_dtype_converter[DTYPE_B]}", "c_ptr":
                "*fp32", "a_scale": "*u8", "b_scale": "*u8", "M": "i32", "N": "i32", "K": "i32", "stride_am": "i32",
                "stride_ak": "i32", "stride_bk": "i32", "stride_bn": "i32", "stride_cm": "i32", "stride_cn": "i32",
                "stride_scale": "i32", "DTYPE_A": "constexpr", "DTYPE_B": "constexpr", "SCALE_BLOCK": "constexpr",
                "BLOCK_M": "constexpr", "BLOCK_N": "constexpr", "BLOCK_K": "constexpr", "GROUP_SIZE_M": "constexpr"
            }, constexprs={
                "DTYPE_A": dot_scaled_dtype_converter[DTYPE_A], "DTYPE_B": dot_scaled_dtype_converter[DTYPE_B],
                "SCALE_BLOCK": scale_block, "BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "BLOCK_K": BLOCK_K, "GROUP_SIZE_M":
                1
            }), target=GPUTarget("hip", 'gfx1250', 32))

    amdgcn = k.asm["amdgcn"]
    pattern = "v_wmma_scale_f32_16x16x128_f8f6f4"
    assert re.search(pattern, amdgcn), f"Can't find instruction {pattern} in AMDGCN assembly"


def init_mxfp_data(dtype, d0: int, d1: int):
    if dtype == 'e2m1':
        return MXFP4Tensor(size=(d0, d1)).random()
    elif dtype == "e5m2":
        return torch.randint(20, 40, (d0, d1), dtype=torch.uint8).view(torch.float8_e5m2)
    elif dtype == "e4m3":
        return torch.randint(20, 40, (d0, d1), dtype=torch.uint8).view(torch.float8_e4m3fn)
    else:
        raise NotImplementedError(f"NYI: unsupported dtype: {dtype}")


@pytest.mark.parametrize("M, N, K", [(32, 32, 128), (128, 128, 512), (1, 8192, 512)])
@pytest.mark.parametrize("BLOCK_M, BLOCK_N, BLOCK_K", [(32, 32, 128), (64, 64, 128), (64, 64, 64)])
@pytest.mark.parametrize("DTYPE_A", ["e5m2", "e4m3", "e2m1"])
@pytest.mark.parametrize("DTYPE_B", ["e5m2", "e4m3", "e2m1"])
def test_runtime_mxgemm(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, DTYPE_A, DTYPE_B):
    scale_block = 32

    torch.manual_seed(0)

    a = init_mxfp_data(DTYPE_A, M, K)
    b = init_mxfp_data(DTYPE_B, K, N)
    a_size = (M, (K + scale_block - 1) // scale_block)
    b_size = (N, (K + scale_block - 1) // scale_block)
    a_scale = MXScaleTensor(size=a_size).random(low=1.0, high=32.0)
    b_scale = MXScaleTensor(size=b_size).random(low=1.0, high=32.0)

    c_ref = torch_gemm_mxfp(a, b, a_scale, b_scale, scale_block, M, N, K)

    a_scale = a_scale.data
    b_scale = b_scale.data

    # mxfp4 input needs packed along the k dim, i.e., two mxfp4 are packed in one uint8
    if DTYPE_A in ['e2m1', 'e2m3', 'e3m2']:
        a = a.to_packed_tensor(dim=1)
    if DTYPE_B in ['e2m1', 'e2m3', 'e3m2']:
        b = b.to_packed_tensor(dim=0)

    c_d = torch.zeros(M, N, dtype=torch.float32).cuda()
    a_d = a.data.contiguous().cuda()
    b_d = b.data.contiguous().cuda()
    a_scale_d = a_scale.cuda()
    b_scale_d = b_scale.cuda()

    stride_am, stride_ak = a_d.stride(0), a_d.stride(1)
    stride_bk, stride_bn = b_d.stride(0), b_d.stride(1)
    stride_cm, stride_cn = c_d.stride(0), c_d.stride(1)
    stride_scale = a_scale_d.stride(0)

    numBlocks = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    grid = [numBlocks, 1, 1]
    group_size_m = 1

    mxgemm_kernel[grid](a_d, b_d, c_d, a_scale_d, b_scale_d, M, N, K, stride_am, stride_ak, stride_bk, stride_bn,
                        stride_cm, stride_cn, stride_scale, DTYPE_A, DTYPE_B, scale_block, BLOCK_M, BLOCK_N, BLOCK_K,
                        group_size_m, num_warps=4, num_ctas=1)

    torch.testing.assert_close(c_d.cpu(), c_ref.cpu(), rtol=1e-5, atol=1e-8)


@gluon.jit
def async_load_and_write_back_kernel(a_ptr, out_ptr, M, N, BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr,
                                     blocked_layout: ttgl.constexpr, shared_layout: ttgl.constexpr):
    pid = ttgl.program_id(axis=0)
    num_pid_m = ttgl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    offs_m = pid_m * BLOCK_M + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, blocked_layout))
    offs_n = pid_n * BLOCK_N + ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, blocked_layout))

    a_ptrs = a_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    buffer = ttgl.allocate_shared_memory(a_ptr.type.element_ty, [BLOCK_M, BLOCK_N], shared_layout)
    ttgl.amd.gfx1250.async_copy.global_to_shared(buffer, a_ptrs)
    ttgl.amd.gfx1250.async_copy.commit_group()
    ttgl.amd.gfx1250.async_copy.wait_group(0)

    res = buffer.load(blocked_layout)

    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    ttgl.store(out_ptrs, res, mask)


ASYNC_COPY_TEST_PARAM_SIZE = pytest.mark.parametrize("M,N", [(128, 128), (1024, 1024), (1008, 1008)])
# We require the vec size to determine if we can use async_copy (>=4bytes), if it's a coalesced layout just assume 16
ASYNC_COPY_TEST_PARAM_SHARED_LAYOUT = pytest.mark.parametrize("vec_size, shared_layout", [
    (16, ttgl.SwizzledSharedLayout(1, 1, 1, [1, 0])),
    (4, ttgl.SwizzledSharedLayout(4, 2, 4, [1, 0])),
    (8, ttgl.SwizzledSharedLayout(8, 2, 4, [1, 0])),
    (16, ttgl.SwizzledSharedLayout(16, 2, 4, [1, 0])),
    (4, ttgl.PaddedSharedLayout.with_identity_for([[4, 4], [8, 4]], [128, 128], [1, 0])),
    (4,
     ttgl.PaddedSharedLayout([[4, 1]], [[0, 1], [0, 2], [0, 8], [0, 4], [16, 0], [32, 0], [0, 16], [0, 32], [0, 64],
                                        [1, 0], [2, 0], [4, 0], [8, 0], [64, 0]], [], [128, 128])),
    (1, ttgl.SwizzledSharedLayout(1, 1, 1, [0, 1])),
    (1, ttgl.SwizzledSharedLayout(4, 2, 4, [0, 1])),
    (1, ttgl.SwizzledSharedLayout(8, 2, 4, [0, 1])),
    (1, ttgl.SwizzledSharedLayout(16, 2, 4, [0, 1])),
    (1, ttgl.PaddedSharedLayout.with_identity_for([[4, 4]], [128, 128], [0, 1])),
    (1, ttgl.PaddedSharedLayout.with_identity_for([[4, 1]], [128, 128], [0, 1])),
])
ASYNC_COPY_TEST_PARAM_DTYPE = pytest.mark.parametrize("dtype", [
    # Test from 1 byte -> 8 bytes dtypes
    torch.float64, torch.float32, torch.float16, torch.float8_e4m3fn
])


def _test_runtime_async_copy_layouts(M, N, vec_size, shared_layout, dtype):
    BLOCK_M = 128
    BLOCK_N = 128
    blocked_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0])

    if dtype == torch.float8_e4m3fn:
        # range from min normal (0 00001 00) to max normal (0 11110 11)
        a = torch.randint(0x04, 0x7B, (M, N), dtype=torch.uint8).view(dtype)
    else:
        a = torch.rand((M, N), dtype=dtype)
    out = torch.empty_like(a)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1)
    out_handle = out.cuda()

    blocked_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0])
    run_kernel = lambda: async_load_and_write_back_kernel[grid](a.cuda(), out_handle, M, N, BLOCK_M, BLOCK_N,
                                                                blocked_layout, shared_layout)

    if (vec_size * dtype.itemsize) < 4:
        # If we have less than 4 contiguous bytes we expect to abort compilation
        with pytest.raises(RuntimeError):
            run_kernel()
    else:
        run_kernel()
        out_tri = out_handle.cpu()
        out_ref = a.cpu()
        assert torch.equal(out_tri, out_ref)


@ASYNC_COPY_TEST_PARAM_SIZE
@ASYNC_COPY_TEST_PARAM_SHARED_LAYOUT
@ASYNC_COPY_TEST_PARAM_DTYPE
def test_runtime_async_copy(M, N, vec_size, shared_layout, dtype):
    _test_runtime_async_copy_layouts(M, N, vec_size, shared_layout, dtype, False)


@gluon.jit
def scaled_wmma_scale_preshuffle(a_base, stride_am, stride_ak, a_scale, b_base, stride_bk, stride_bn, b_scale, out,
                                 stride_scale, BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr,
                                 BLOCK_K: ttgl.constexpr, type_a: ttgl.constexpr, type_b: ttgl.constexpr,
                                 TRANSPOSED_WMMA: ttgl.constexpr):
    DIV_FACTOR_A: ttgl.constexpr = 2 if type_a == "e2m1" else 1
    DIV_FACTOR_B: ttgl.constexpr = 2 if type_b == "e2m1" else 1
    PACKED_BLOCK_K_A: ttgl.constexpr = BLOCK_K // DIV_FACTOR_A
    PACKED_BLOCK_K_B: ttgl.constexpr = BLOCK_K // DIV_FACTOR_B
    SCALE_BLOCK_K: ttgl.constexpr = BLOCK_K // 32
    SCALE_KWIDTH: ttgl.constexpr = 4 if SCALE_BLOCK_K >= 4 else SCALE_BLOCK_K

    tiles_per_warp: ttgl.constexpr = [2, 2]
    NON_K_PRESHUFFLE_BLOCK_SIZE: ttgl.constexpr = 128

    scale_blocked_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [8, 4], [4, 1], [1, 0])
    a_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 16], [8, 4], [4, 1], [1, 0])
    b_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 16], [16, 2], [4, 1], [1, 0])

    wmma_layout: ttgl.constexpr = ttgl.amd.AMDWMMALayout(version=3, transposed=TRANSPOSED_WMMA, warps_per_cta=[2, 2],
                                                         instr_shape=[16, 16, 128], tiles_per_warp=tiles_per_warp)
    wmma_layout_packed: ttgl.constexpr = ttgl.amd.AMDWMMALayout(version=3, transposed=TRANSPOSED_WMMA,
                                                                warps_per_cta=[2, 2], instr_shape=[16, 16, 64],
                                                                tiles_per_warp=tiles_per_warp)

    operand_a_layout: ttgl.constexpr = ttgl.DotOperandLayout(
        operand_index=0, parent=wmma_layout_packed if type_a == "e2m1" else wmma_layout, k_width=16)
    operand_b_layout: ttgl.constexpr = ttgl.DotOperandLayout(
        operand_index=1, parent=wmma_layout_packed if type_b == "e2m1" else wmma_layout, k_width=16)

    a_scale_linear_layout: ttgl.constexpr = ttgl.amd.gfx1250.get_wmma_scale_layout(operand_a_layout,
                                                                                   [BLOCK_M, SCALE_BLOCK_K])
    b_scale_linear_layout: ttgl.constexpr = ttgl.amd.gfx1250.get_wmma_scale_layout(operand_b_layout,
                                                                                   [BLOCK_N, SCALE_BLOCK_K])

    zero = ttgl.zeros([BLOCK_M, BLOCK_N], dtype=ttgl.float32, layout=wmma_layout)

    offs_am = ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, a_layout))
    offs_ak = ttgl.arange(0, PACKED_BLOCK_K_A, layout=ttgl.SliceLayout(0, a_layout))
    a_offsets = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
    a = ttgl.load(a_base + a_offsets)
    a = ttgl.convert_layout(a, operand_a_layout)

    offs_bk = ttgl.arange(0, PACKED_BLOCK_K_B, layout=ttgl.SliceLayout(1, b_layout))
    offs_bn = ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, b_layout))
    b_offsets = offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    b = ttgl.load(b_base + b_offsets)
    b = ttgl.convert_layout(b, operand_b_layout)

    offs_scale_am = ttgl.arange(0, BLOCK_M // NON_K_PRESHUFFLE_BLOCK_SIZE,
                                layout=ttgl.SliceLayout(1, scale_blocked_layout))
    off_scale_ak = ttgl.arange(0, SCALE_BLOCK_K * NON_K_PRESHUFFLE_BLOCK_SIZE,
                               layout=ttgl.SliceLayout(0, scale_blocked_layout))
    a_scale_offsets = offs_scale_am[:, None] * stride_scale + off_scale_ak[None, :]
    scale_a = ttgl.load(a_scale + a_scale_offsets)

    offs_scale_bn = ttgl.arange(0, BLOCK_N // NON_K_PRESHUFFLE_BLOCK_SIZE,
                                layout=ttgl.SliceLayout(1, scale_blocked_layout))
    offs_scale_bk = ttgl.arange(0, SCALE_BLOCK_K * NON_K_PRESHUFFLE_BLOCK_SIZE,
                                layout=ttgl.SliceLayout(0, scale_blocked_layout))
    b_scale_offsets = offs_scale_bn[:, None] * stride_scale + offs_scale_bk[None, :]
    scale_b = ttgl.load(b_scale + b_scale_offsets)

    scale_a = scale_a.reshape(BLOCK_M // NON_K_PRESHUFFLE_BLOCK_SIZE, SCALE_BLOCK_K // SCALE_KWIDTH, 32, 4,
                              SCALE_KWIDTH).trans(0, 3, 2, 1, 4).reshape(BLOCK_M, SCALE_BLOCK_K)
    scale_b = scale_b.reshape(BLOCK_N // NON_K_PRESHUFFLE_BLOCK_SIZE, SCALE_BLOCK_K // SCALE_KWIDTH, 32, 4,
                              SCALE_KWIDTH).trans(0, 3, 2, 1, 4).reshape(BLOCK_N, SCALE_BLOCK_K)
    scale_a = ttgl.convert_layout(scale_a, a_scale_linear_layout)
    scale_b = ttgl.convert_layout(scale_b, b_scale_linear_layout)

    c = ttgl.amd.gfx1250.wmma_scaled(a, scale_a, type_a, b, scale_b, type_b, zero)
    c = c.to(out.dtype.element_ty)

    offs_cm = ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, wmma_layout))
    offs_cn = ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, wmma_layout))
    out_offsets = offs_cm[:, None] * BLOCK_N + offs_cn[None, :]
    out = out + out_offsets
    ttgl.store(out, c)


@pytest.mark.parametrize("M, N, K", [(128, 128, 64), (128, 128, 128), (256, 256, 256)])
@pytest.mark.parametrize("type_a", ["e5m2", "e2m1", "e4m3"])
@pytest.mark.parametrize("type_b", ["e5m2", "e2m1", "e4m3"])
@pytest.mark.parametrize("TRANSPOSED_WMMA", [True, False])
def test_compile_wmma_scale_preshuffle(M, N, K, type_a, type_b, TRANSPOSED_WMMA):
    dtype_converter = {'e5m2': "fp8e5", "e4m3": "fp8e4nv", "e2m1": "u8"}

    signature = {
        "a_base": f"*{dtype_converter[type_a]}", "stride_am": "i32", "stride_ak": "i32", "a_scale": "*i8", "b_base":
        f"*{dtype_converter[type_b]}", "stride_bk": "i32", "stride_bn": "i32", "b_scale": "*i8", "out": "*i32",
        "stride_scale": "i32", "BLOCK_M": "constexpr", "BLOCK_N": "constexpr", "BLOCK_K": "constexpr", "type_a":
        "constexpr", "type_b": "constexpr", "TRANSPOSED_WMMA": "constexpr"
    }

    constexprs = {
        "BLOCK_M": M, "BLOCK_N": N, "BLOCK_K": K, "type_a": type_a, "type_b": type_b, "TRANSPOSED_WMMA": TRANSPOSED_WMMA
    }
    k = triton.compile(
        src=gluon._runtime.GluonASTSource(fn=scaled_wmma_scale_preshuffle, signature=signature, constexprs=constexprs),
        target=GPUTarget("hip", 'gfx1250', 32))
    amdgcn = k.asm["amdgcn"]

    instr = "v_wmma_scale_f32_16x16x128_f8f6f4"
    scale_opsel_a = "matrix_a_scale:MATRIX_SCALE_ROW1"
    scale_opsel_b = "matrix_b_scale:MATRIX_SCALE_ROW1"
    for suffix in (scale_opsel_a, scale_opsel_b, f"{scale_opsel_a} {scale_opsel_b}"):
        pattern = f"{instr}.*{suffix}\n"
        assert re.search(pattern, amdgcn), f"Can't find pattern {pattern} in AMDGCN assembly"


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires GFX1250")
@pytest.mark.parametrize("M, N, K", [(128, 128, 64), (128, 128, 128), (256, 256, 256)])
@pytest.mark.parametrize("type_a", ["e5m2", "e2m1", "e4m3"])
@pytest.mark.parametrize("type_b", ["e5m2", "e2m1", "e4m3"])
@pytest.mark.parametrize("TRANSPOSED_WMMA", [True, False])
def test_runtime_wmma_scale_preshuffle(M, N, K, type_a, type_b, TRANSPOSED_WMMA):

    def pack_scale(x):
        NON_K, K_SCALE = x.shape
        num_chunk_m = NON_K // 128
        SCALE_KWIDTH = 4 if K_SCALE >= 4 else K_SCALE
        num_chunk_k = K_SCALE // SCALE_KWIDTH

        x = x.view(num_chunk_m, 4, 32, num_chunk_k, SCALE_KWIDTH)
        x = x.permute(0, 3, 2, 1, 4).contiguous()
        return x.view(NON_K // 128, K_SCALE * 128)

    torch.manual_seed(0)

    a = init_mxfp_data(type_a, M, K)
    b = init_mxfp_data(type_b, K, N)
    scale_a_size = (M, (K + 32 - 1) // 32)
    scale_b_size = (N, (K + 32 - 1) // 32)

    scale_a_mxfp4 = MXScaleTensor(size=scale_a_size).random(low=1.0, high=32.0)
    scale_b_mxfp4 = MXScaleTensor(size=scale_b_size).random(low=1.0, high=32.0)

    c_torch = torch_gemm_mxfp(a, b, scale_a_mxfp4, scale_b_mxfp4, 32, M, N, K)

    if type_a == "e2m1":
        a = a.to_packed_tensor(dim=1)

    if type_b == "e2m1":
        b = b.to_packed_tensor(dim=0)
    a = a.data.contiguous().cuda()
    b = b.data.contiguous().cuda()

    scale_a = scale_a_mxfp4.data
    scale_b = scale_b_mxfp4.data

    scale_a = pack_scale(scale_a)
    scale_b = pack_scale(scale_b)

    scale_a = scale_a.cuda()
    scale_b = scale_b.cuda()

    stride_scale = scale_a.stride(0)

    c = torch.zeros((M, N), dtype=torch.float32).cuda()
    scaled_wmma_scale_preshuffle[(1, )](a, *a.stride(), scale_a, b, *b.stride(), scale_b, c, stride_scale, M, N, K,
                                        type_a, type_b, TRANSPOSED_WMMA)

    torch.testing.assert_close(c.cpu(), c_torch, rtol=1e-5, atol=1e-5)
