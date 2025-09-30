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

    torch.manual_seed(0)

    type_a = mxfp_type
    type_b = mxfp_type

    DIV_FACTOR_A = 2 if type_a == "e2m1" else 1
    DIV_FACTOR_B = 2 if type_b == "e2m1" else 1

    x = torch.randint(20, 40, (BLOCK_M, BLOCK_K // DIV_FACTOR_A), dtype=torch.uint8).cuda()
    y = torch.randint(20, 40, (BLOCK_K // DIV_FACTOR_B, BLOCK_N), dtype=torch.uint8).cuda()

    if hasScale:
        min_scale, max_scale = (0, 142)
        scale_x = torch.randint(min_scale, max_scale + 1, (BLOCK_M, BLOCK_K // 32), dtype=torch.uint8).cuda()
        scale_y = torch.randint(min_scale, max_scale + 1, (BLOCK_N, BLOCK_K // 32), dtype=torch.uint8).cuda()
    else:
        scale_x = None
        scale_y = None

    def make_finite(x, dtype):
        if dtype not in ("e5m2", "e4m3"):
            return x
        mask = 0x7C if dtype == "e5m2" else 0x7F
        finite = torch.arange(x.numel(), dtype=torch.uint8).cuda().reshape_as(x) % mask
        x_finite = torch.where(x & mask == mask, finite | (0x80 & x), x)
        x.copy_(x_finite)
        return x

    x = make_finite(x, type_a)
    y = make_finite(y, type_b)

    z = torch.zeros((BLOCK_M, BLOCK_N), dtype=torch.float32).cuda()
    pgm = dot_mxfp_gluon_kernel[(1, )](x, *x.stride(), scale_x, y, *y.stride(), scale_y, z, BLOCK_M, BLOCK_N, BLOCK_K,
                                       type_a, type_b)
    assert "v_wmma_scale_f32_16x16x128_f8f6f4" in pgm.asm[
        "amdgcn"], "The AMDGCN assembly does not contain the expected scaled WMMA instruction."

    z_ref = torch.zeros((BLOCK_M, BLOCK_N), dtype=torch.float32).cuda()
    dot_mxfp_triton_kernel[(1, )](x, *x.stride(), scale_x, y, *y.stride(), scale_y, z_ref, BLOCK_M, BLOCK_N, BLOCK_K,
                                  type_a, type_b)

    torch.testing.assert_close(z.cpu(), z_ref.cpu(), rtol=1e-5, atol=1e-5)


def tensor_copy_kernel(a_ptr, b_ptr,  #
                       M, N,  #
                       BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr):
    SHARED_LAYOUT: ttgl.constexpr = ttgl.PaddedSharedLayout.with_identity_for([[32, 4]], [BLOCK_M, BLOCK_N], [1, 0])
    BLOCKED_LAYOUT: ttgl.constexpr = ttgl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0])

    pid = ttgl.program_id(axis=0)
    num_pid_m = ttgl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    a_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(base=a_ptr, shape=(M, N), strides=(N, 1),
                                                         block_shape=(BLOCK_M, BLOCK_N), layout=SHARED_LAYOUT)

    a_buffer = ttgl.allocate_shared_memory(a_desc.dtype, shape=a_desc.block_shape, layout=a_desc.layout)
    ttgl.amd.gfx1250.tdm.async_load(a_desc, [pid_m * BLOCK_M, pid_n * BLOCK_N], a_buffer)

    ttgl.amd.gfx1250.tdm.async_wait(0)
    a = a_buffer.load(layout=BLOCKED_LAYOUT)

    b_offsets = (pid_m * BLOCK_M + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, BLOCKED_LAYOUT)))[:, None] * N + \
                (pid_n * BLOCK_N + ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, BLOCKED_LAYOUT)))[None, :]
    ttgl.store(b_ptr + b_offsets, a)


@pytest.mark.parametrize("BLOCK_M,BLOCK_N", [(32, 32), (32, 64), (64, 64)])
def test_compile_tensor_copy(BLOCK_M, BLOCK_N):
    k = triton.compile(
        gluon._runtime.GluonASTSource(
            fn=tensor_copy_kernel, signature={
                "a_ptr": "*bf16", "b_ptr": "*bf16", "M": "i32", "N": "i32", "BLOCK_M": "constexpr", "BLOCK_N":
                "constexpr"
            }, constexprs={"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N}), target=GPUTarget("hip", 'gfx1250', 32))

    amdgcn = k.asm["amdgcn"]

    tensor_pattern = r"tensor_load_to_lds"
    assert re.search(tensor_pattern, amdgcn)

    wait_pattern = r"s_wait_tensorcnt 0x0"
    assert re.search(wait_pattern, amdgcn)


@pytest.mark.parametrize("BLOCK_M,BLOCK_N", [(32, 32), (32, 64), (64, 64)])
def test_runtime_tensor_copy(BLOCK_M, BLOCK_N):
    M, N = 1024, 1024

    torch.manual_seed(42)
    a = torch.randint(0x0, 0xFFFF, (M, N), dtype=torch.uint16)
    b = torch.zeros_like(a)

    a_device = a.cuda()
    b_device = b.cuda()
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1)
    tensor_copy_kernel[grid](a_device, b_device, M, N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)

    b_triton = b_device.cpu()
    assert torch.equal(b_triton, a)
