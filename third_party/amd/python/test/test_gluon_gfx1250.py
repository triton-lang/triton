# ruff: noqa: E402
import hip

hip.hip.hipInit(0)
# Needed for internal dev flow for now; will remove later

import re
import pytest
import torch

import triton
import triton.language as tl
from triton.language.core import _aggregate as aggregate
from triton.backends.compiler import GPUTarget
from triton._internal_testing import is_hip_gfx1250, str_to_triton_dtype, numpy_random, to_triton, unwrap_tensor, dtypes_with_bfloat16, uint_dtypes
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


@pytest.mark.parametrize("a_dtype,b_dtype,k_dim", get_test_gemm_variants())
@pytest.mark.parametrize("BLOCK_M,BLOCK_N,BLOCK_K", get_test_gemm_block_mnk())
def test_compile_gemm(a_dtype, b_dtype, k_dim, BLOCK_M, BLOCK_N, BLOCK_K):
    if BLOCK_K < k_dim:
        pytest.skip("Skip tests where BLOCK_K < k_dim")

    a_dtype = str_to_triton_dtype(a_dtype).name
    b_dtype = str_to_triton_dtype(b_dtype).name

    signature = {
        "a_ptr": f"*{a_dtype}", "b_ptr": f"*{b_dtype}", "c_ptr": "*fp32",  #
        "M": "i32", "N": "i32", "K": "i32",  #
        "stride_am": "i32", "stride_ak": "i32",  #
        "stride_bk": "i32", "stride_bn": "i32",  #
        "stride_cm": "i32", "stride_cn": "i32",  #
        "BLOCK_M": "constexpr", "BLOCK_N": "constexpr", "BLOCK_K": "constexpr",  #
        "INSTR_SHAPE_K": "constexpr", "K_WIDTH": "constexpr"
    }
    constexprs = {
        "BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "BLOCK_K": BLOCK_K,  #
        "INSTR_SHAPE_K": k_dim, "K_WIDTH": 2 if a_dtype == "fp32" else 8
    }
    fn = gemm_kernel

    k = triton.compile(src=gluon._runtime.GluonASTSource(fn, signature, constexprs),
                       target=GPUTarget("hip", 'gfx1250', 32))
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


@pytest.mark.parametrize("a_dtype,b_dtype,k_dim", get_test_gemm_variants())
@pytest.mark.parametrize("BLOCK_M,BLOCK_N,BLOCK_K", get_test_gemm_block_mnk())
@pytest.mark.parametrize("M,N,K", get_test_gemm_shapes())
def test_runtime_gemm(a_dtype, b_dtype, k_dim, BLOCK_M, BLOCK_N, BLOCK_K, M, N, K):
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
def gemm_async_pipelined_kernel(a_ptr, b_ptr, c_ptr,  #
                                M, N, K,  #
                                stride_am, stride_ak,  #
                                stride_bk, stride_bn,  #
                                stride_cm, stride_cn,  #
                                BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr, BLOCK_K: ttgl.constexpr,  #
                                NUM_BUFFERS: ttgl.constexpr, USE_TDM: ttgl.constexpr):
    a_dtype: ttgl.constexpr = a_ptr.type.element_ty
    b_dtype: ttgl.constexpr = b_ptr.type.element_ty
    ttgl.static_assert(a_dtype.is_fp16() or a_dtype.is_bf16(), "Only fp16/bf16 supported for A")
    ttgl.static_assert(b_dtype.is_fp16() or b_dtype.is_bf16(), "Only fp16/bf16 supported for B")
    ttgl.static_assert(NUM_BUFFERS >= 2, "NUM_BUFFERS must be at least 2")

    BLOCKED_LAYOUT: ttgl.constexpr = ttgl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0])
    WMMA_LAYOUT: ttgl.constexpr = ttgl.amd.AMDWMMALayout(3, True, [2, 2], [16, 16, 32])
    SHARED_LAYOUT_A: ttgl.constexpr = ttgl.PaddedSharedLayout.with_identity_for([[BLOCK_K, 8]], [BLOCK_M, BLOCK_K],
                                                                                [1, 0])
    SHARED_LAYOUT_B: ttgl.constexpr = ttgl.PaddedSharedLayout.with_identity_for([[BLOCK_N, 8]], [BLOCK_K, BLOCK_N],
                                                                                [1, 0])
    OPERAND_LAYOUT_A: ttgl.constexpr = ttgl.DotOperandLayout(0, WMMA_LAYOUT, 8)
    OPERAND_LAYOUT_B: ttgl.constexpr = ttgl.DotOperandLayout(1, WMMA_LAYOUT, 8)

    pid = ttgl.program_id(axis=0)
    num_pid_m = ttgl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    # Descriptors for TDM
    a_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(  #
        base=a_ptr + pid_m * BLOCK_M * stride_am,  #
        shape=(M, K),  #
        strides=(stride_am, stride_ak),  #
        block_shape=(BLOCK_M, BLOCK_K),  #
        layout=SHARED_LAYOUT_A)
    b_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(  #
        base=b_ptr + pid_n * BLOCK_N * stride_bn,  #
        shape=(K, N),  #
        strides=(stride_bk, stride_bn),  #
        block_shape=(BLOCK_K, BLOCK_N),  #
        layout=SHARED_LAYOUT_B)

    # Pointers for AsyncCopy
    offs_ak = ttgl.arange(0, BLOCK_K, layout=ttgl.SliceLayout(0, BLOCKED_LAYOUT))
    offs_am = (pid_m * BLOCK_M + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, BLOCKED_LAYOUT))) % M
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak

    offs_bk = ttgl.arange(0, BLOCK_K, layout=ttgl.SliceLayout(1, BLOCKED_LAYOUT))
    offs_bn = (pid_n * BLOCK_N + ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, BLOCKED_LAYOUT))) % N
    b_ptrs = b_ptr + offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    a_buffer = ttgl.allocate_shared_memory(a_desc.dtype, shape=[NUM_BUFFERS] + a_desc.block_shape, layout=a_desc.layout)
    b_buffer = ttgl.allocate_shared_memory(b_desc.dtype, shape=[NUM_BUFFERS] + b_desc.block_shape, layout=b_desc.layout)

    load_idx = 0
    wmma_idx = 0
    accumulator = ttgl.zeros((BLOCK_M, BLOCK_N), dtype=c_ptr.type.element_ty, layout=WMMA_LAYOUT)

    for _ in ttgl.static_range(NUM_BUFFERS - 1):
        if USE_TDM:
            ttgl.amd.gfx1250.tdm.async_load(a_desc, [0, load_idx * BLOCK_K],  #
                                            a_buffer.index(load_idx % NUM_BUFFERS))
            ttgl.amd.gfx1250.tdm.async_load(b_desc, [load_idx * BLOCK_K, 0],  #
                                            b_buffer.index(load_idx % NUM_BUFFERS))
        else:
            mask_a = offs_ak[None, :] < K - load_idx * BLOCK_K
            ttgl.amd.gfx1250.async_copy.global_to_shared(a_buffer.index(load_idx % NUM_BUFFERS), a_ptrs, mask_a,
                                                         other=0.0)

            mask_b = offs_bk[:, None] < K - load_idx * BLOCK_K
            ttgl.amd.gfx1250.async_copy.global_to_shared(b_buffer.index(load_idx % NUM_BUFFERS), b_ptrs, mask_b,
                                                         other=0.0)
            ttgl.amd.gfx1250.async_copy.commit_group()

        load_idx += 1
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    for _ in range(0, ttgl.cdiv(K, BLOCK_K) - (NUM_BUFFERS - 1)):
        if USE_TDM:
            ttgl.amd.gfx1250.tdm.async_load(a_desc, [0, load_idx * BLOCK_K],  #
                                            a_buffer.index(load_idx % NUM_BUFFERS))
            ttgl.amd.gfx1250.tdm.async_load(b_desc, [load_idx * BLOCK_K, 0],  #
                                            b_buffer.index(load_idx % NUM_BUFFERS))
        else:
            mask_a = offs_ak[None, :] < K - load_idx * BLOCK_K
            ttgl.amd.gfx1250.async_copy.global_to_shared(a_buffer.index(load_idx % NUM_BUFFERS), a_ptrs, mask_a,
                                                         other=0.0)

            mask_b = offs_bk[:, None] < K - load_idx * BLOCK_K
            ttgl.amd.gfx1250.async_copy.global_to_shared(b_buffer.index(load_idx % NUM_BUFFERS), b_ptrs, mask_b,
                                                         other=0.0)
            ttgl.amd.gfx1250.async_copy.commit_group()

        load_idx += 1
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

        if USE_TDM:
            ttgl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 1) * 2)
        else:
            ttgl.amd.gfx1250.async_copy.wait_group((NUM_BUFFERS - 1))

        a = a_buffer.index(wmma_idx % NUM_BUFFERS).load(layout=OPERAND_LAYOUT_A)
        b = b_buffer.index(wmma_idx % NUM_BUFFERS).load(layout=OPERAND_LAYOUT_B)
        accumulator = ttgl.amd.gfx1250.wmma(a, b, accumulator)
        wmma_idx += 1

    for i in ttgl.static_range(NUM_BUFFERS - 1):
        if USE_TDM:
            ttgl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 2 - i) * 2)
        else:
            ttgl.amd.gfx1250.async_copy.wait_group((NUM_BUFFERS - 2 - i))

        a = a_buffer.index(wmma_idx % NUM_BUFFERS).load(layout=OPERAND_LAYOUT_A)
        b = b_buffer.index(wmma_idx % NUM_BUFFERS).load(layout=OPERAND_LAYOUT_B)
        accumulator = ttgl.amd.gfx1250.wmma(a, b, accumulator)
        wmma_idx += 1

    offs_cm = pid_m * BLOCK_M + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, WMMA_LAYOUT))
    offs_cn = pid_n * BLOCK_N + ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, WMMA_LAYOUT))
    offs_c = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    ttgl.store(c_ptr + offs_c, accumulator, mask=mask_c)


@pytest.mark.parametrize("BLOCK_M,BLOCK_N,BLOCK_K", [(m, n, k) for (m, n) in [(32, 32), (64, 64)] \
                                                               for k in [32, 64]])
@pytest.mark.parametrize("NUM_BUFFERS", [2, 4])
@pytest.mark.parametrize("ASYNC_LOAD_TYPE", ["ASYNC_COPY", "TDM"])
def test_compile_gemm_async_pipelined(BLOCK_M, BLOCK_N, BLOCK_K, NUM_BUFFERS, ASYNC_LOAD_TYPE):
    # Inner strides need to be constexpr (1) to get contiguity. Note the compiler frontend does the same for normal dispatches
    signature = {
        "a_ptr": "*fp16", "b_ptr": "*fp16", "c_ptr": "*fp32",  #
        "M": "i32", "N": "i32", "K": "i32",  #
        "stride_am": "i32", "stride_ak": "constexpr",  #
        "stride_bk": "i32", "stride_bn": "constexpr",  #
        "stride_cm": "i32", "stride_cn": "constexpr",  #
        "BLOCK_M": "constexpr", "BLOCK_N": "constexpr", "BLOCK_K": "constexpr",  #
        "NUM_BUFFERS": "constexpr", "USE_TDM": "constexpr"
    }

    constexprs = {
        "stride_ak": 1, "stride_bn": 1, "stride_cn": 1, "BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "BLOCK_K": BLOCK_K,
        "NUM_BUFFERS": NUM_BUFFERS, "USE_TDM": ASYNC_LOAD_TYPE == "TDM"
    }
    fn = gemm_async_pipelined_kernel

    # AsyncCopy requires >= 32 bits per lane so we have to pass divisibility for arguments used in pointer arithmetic
    attrs = []
    if ASYNC_LOAD_TYPE == "ASYNC_COPY":
        attrs = {k: [["tt.divisibility", 16]] for k in [(x, ) for x in range(11)]}

    k = triton.compile(src=gluon._runtime.GluonASTSource(fn, signature, constexprs, attrs=attrs),
                       target=GPUTarget("hip", 'gfx1250', 32))
    amdgcn = k.asm["amdgcn"]

    assert re.search("v_wmma_f32_16x16x32_f16", amdgcn)

    if ASYNC_LOAD_TYPE == "TDM":
        for cnt in range(NUM_BUFFERS - 1, -1, -1):
            assert re.search(f"s_wait_tensorcnt 0x{(cnt * 2):x}", amdgcn)
        assert len(re.findall("tensor_load_to_lds", amdgcn)) == NUM_BUFFERS * 2
    else:
        copy_instr_for_A = BLOCK_M // 4 // 4
        copy_isntr_for_B = BLOCK_K // 4 // 4
        copy_instr_per_iter = copy_instr_for_A + copy_isntr_for_B
        for cnt in range(NUM_BUFFERS - 1, -1, -1):
            assert re.search(f"s_wait_asynccnt 0x{(cnt * copy_instr_per_iter):x}", amdgcn)
        # Each instruction loads 4 rows per warp and we have 4 warps (see BlockedLayout in test)
        assert len(re.findall("global_load_async_to_lds", amdgcn)) == NUM_BUFFERS * copy_instr_per_iter


@pytest.mark.parametrize("BLOCK_M,BLOCK_N,BLOCK_K", [(m, n, k) for (m, n) in [(32, 32), (64, 64)] \
                                                               for k in [32, 64]])
@pytest.mark.parametrize("NUM_BUFFERS", [2, 4])
@pytest.mark.parametrize("M,N,K", [(256, 256, 512), (240, 240, 496), (250, 250, 510)])
@pytest.mark.parametrize("ASYNC_LOAD_TYPE", ["ASYNC_COPY", "TDM"])
def test_runtime_gemm_async_pipelined(BLOCK_M, BLOCK_N, BLOCK_K, NUM_BUFFERS, M, N, K, ASYNC_LOAD_TYPE):
    if triton.cdiv(K, BLOCK_K) < NUM_BUFFERS:
        pytest.skip("Skip tests where K/BLOCK_K < NUM_BUFFERS")

    if ASYNC_LOAD_TYPE == "ASYNC_COPY" and any([x % 16 != 0 for x in [M, N, K]]):
        pytest.skip("AsyncCopy tests need divisibility==16 to get vectorization information")

    torch.manual_seed(42)

    a = torch.randn((M, K), dtype=torch.float16)
    b = torch.randn((K, N), dtype=torch.float16)
    c = torch.zeros((M, N), dtype=torch.float32)
    stride_am, stride_ak = a.stride(0), a.stride(1)
    stride_bk, stride_bn = b.stride(0), b.stride(1)
    stride_cm, stride_cn = c.stride(0), c.stride(1)

    a_device = a.cuda()
    b_device = b.cuda()
    c_device = c.cuda()
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1)
    gemm_async_pipelined_kernel[grid](
        a_device, b_device, c_device,  #
        M, N, K,  #
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,  #
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,  #
        NUM_BUFFERS=NUM_BUFFERS, USE_TDM=ASYNC_LOAD_TYPE == "TDM")

    c_triton = c_device.cpu()
    c_torch = a.to(torch.float32) @ b.to(torch.float32)
    torch.testing.assert_close(c_triton, c_torch, rtol=1e-4, atol=1e-4)


@gluon.jit
def gemm_async_kernel(a_ptr, b_ptr, c_ptr,  #
                      M, N, K,  #
                      stride_am, stride_ak,  #
                      stride_bk, stride_bn,  #
                      stride_cm, stride_cn,  #
                      BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr, BLOCK_K: ttgl.constexpr,  #
                      INSTR_SHAPE_K: ttgl.constexpr, K_WIDTH: ttgl.constexpr, USE_TDM: ttgl.constexpr):

    BLOCKED_LAYOUT: ttgl.constexpr = ttgl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0])
    WMMA_LAYOUT: ttgl.constexpr = ttgl.amd.AMDWMMALayout(3, True, [2, 2], [16, 16, INSTR_SHAPE_K])
    SHARED_LAYOUT_A: ttgl.constexpr = ttgl.PaddedSharedLayout.with_identity_for([[32, 4]], [BLOCK_M, BLOCK_K], [1, 0])
    SHARED_LAYOUT_B: ttgl.constexpr = ttgl.PaddedSharedLayout.with_identity_for([[32, 4]], [BLOCK_K, BLOCK_N], [1, 0])

    pid = ttgl.program_id(axis=0)
    num_pid_m = ttgl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    # Descriptors for TDM
    a_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(base=a_ptr + pid_m * BLOCK_M * stride_am, shape=(M, K),
                                                         strides=(stride_am, stride_ak), block_shape=(BLOCK_M, BLOCK_K),
                                                         layout=SHARED_LAYOUT_A)
    b_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(base=b_ptr + pid_n * BLOCK_N * stride_bn, shape=(K, N),
                                                         strides=(stride_bk, stride_bn), block_shape=(BLOCK_K, BLOCK_N),
                                                         layout=SHARED_LAYOUT_B)

    # Pointers for AsyncCopy
    offs_ak = ttgl.arange(0, BLOCK_K, layout=ttgl.SliceLayout(0, BLOCKED_LAYOUT))
    offs_am = (pid_m * BLOCK_M + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, BLOCKED_LAYOUT))) % M
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak

    offs_bk = ttgl.arange(0, BLOCK_K, layout=ttgl.SliceLayout(1, BLOCKED_LAYOUT))
    offs_bn = (pid_n * BLOCK_N + ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, BLOCKED_LAYOUT))) % N
    b_ptrs = b_ptr + offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    a_buffer = ttgl.allocate_shared_memory(a_desc.dtype, shape=a_desc.block_shape, layout=a_desc.layout)
    b_buffer = ttgl.allocate_shared_memory(b_desc.dtype, shape=b_desc.block_shape, layout=b_desc.layout)

    accumulator = ttgl.zeros((BLOCK_M, BLOCK_N), dtype=c_ptr.type.element_ty, layout=WMMA_LAYOUT)
    for k in range(0, ttgl.cdiv(K, BLOCK_K)):
        if USE_TDM:
            ttgl.amd.gfx1250.tdm.async_load(a_desc, [0, k * BLOCK_K], a_buffer)
            ttgl.amd.gfx1250.tdm.async_load(b_desc, [k * BLOCK_K, 0], b_buffer)
            ttgl.amd.gfx1250.tdm.async_wait(0)
        else:
            mask_a = offs_ak[None, :] < K - k * BLOCK_K
            ttgl.amd.gfx1250.async_copy.global_to_shared(a_buffer, a_ptrs, mask_a, other=0.0)

            mask_b = offs_bk[:, None] < K - k * BLOCK_K
            ttgl.amd.gfx1250.async_copy.global_to_shared(b_buffer, b_ptrs, mask_b, other=0.0)
            ttgl.amd.gfx1250.async_copy.commit_group()
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
            ttgl.amd.gfx1250.async_copy.wait_group(0)

        a = a_buffer.load(layout=BLOCKED_LAYOUT)
        b = b_buffer.load(layout=BLOCKED_LAYOUT)

        a = ttgl.convert_layout(a, ttgl.DotOperandLayout(0, WMMA_LAYOUT, K_WIDTH))
        b = ttgl.convert_layout(b, ttgl.DotOperandLayout(1, WMMA_LAYOUT, K_WIDTH))
        accumulator = ttgl.amd.gfx1250.wmma(a, b, accumulator)

    offs_cm = pid_m * BLOCK_M + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, WMMA_LAYOUT))
    offs_cn = pid_n * BLOCK_N + ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, WMMA_LAYOUT))
    offs_c = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    ttgl.store(c_ptr + offs_c, accumulator, mask=mask_c)


@pytest.mark.parametrize("BLOCK_M,BLOCK_N,BLOCK_K", [(32, 32, 32), (64, 64, 64), (128, 128, 64)])
@pytest.mark.parametrize("a_dtype,b_dtype,k_dim", [
    ("bfloat16", "bfloat16", 32),
    ("float8_e5m2", "float8_e5m2", 64),
])
@pytest.mark.parametrize("ASYNC_LOAD_TYPE", ["ASYNC_COPY", "TDM"])
def test_compile_gemm_async(BLOCK_M, BLOCK_N, BLOCK_K, a_dtype, b_dtype, k_dim, ASYNC_LOAD_TYPE):
    if BLOCK_K < k_dim:
        pytest.skip("Skip tests where BLOCK_K < k_dim")

    a_dtype = str_to_triton_dtype(a_dtype).name
    b_dtype = str_to_triton_dtype(b_dtype).name

    # AsyncCopy requires >= 32 bits per lane so we have to pass divisibility for arguments used in pointer arithmetic
    attrs = []
    if ASYNC_LOAD_TYPE == "ASYNC_COPY":
        attrs = {(k, ): [["tt.divisibility", 16]] for k in [0, 1, 2, 3, 4, 5, 6, 8, 10]}

    k = triton.compile(
        # Inner strides need to be constexpr (1) to get contiguity. Note the compiler frontend does the same for normal dispatches
        gluon._runtime.GluonASTSource(
            fn=gemm_async_kernel, signature={
                "a_ptr": f"*{a_dtype}", "b_ptr": f"*{b_dtype}", "c_ptr": "*fp32",  #
                "M": "i32", "N": "i32", "K": "i32",  #
                "stride_am": "i32", "stride_ak": "constexpr",  #
                "stride_bk": "i32", "stride_bn": "constexpr",  #
                "stride_cm": "i32", "stride_cn": "constexpr",  #
                "BLOCK_M": "constexpr", "BLOCK_N": "constexpr", "BLOCK_K": "constexpr",  #
                "INSTR_SHAPE_K": "constexpr", "K_WIDTH": "constexpr", "USE_TDM": "constexpr"
            }, attrs=attrs, constexprs={
                "stride_ak": 1, "stride_bn": 1, "stride_cn": 1, "BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "BLOCK_K":
                BLOCK_K, "INSTR_SHAPE_K": k_dim, "K_WIDTH": 8, "USE_TDM": ASYNC_LOAD_TYPE == "TDM"
            }), target=GPUTarget("hip", 'gfx1250', 32))
    amdgcn = k.asm["amdgcn"]

    if ASYNC_LOAD_TYPE == "TDM":
        patterns = ("tensor_load_to_lds", "s_wait_tensorcnt 0x0")
    elif ASYNC_LOAD_TYPE == "ASYNC_COPY":
        patterns = ("global_load_async_to_lds", "s_wait_asynccnt 0x0")

    for pattern in patterns:
        assert re.search(pattern, amdgcn), f"Can't find {pattern} in amdgcn"


@pytest.mark.parametrize("M,N,K", [(256, 256, 128), (250, 250, 120)])
@pytest.mark.parametrize("BLOCK_M,BLOCK_N,BLOCK_K", [(32, 32, 32), (64, 64, 64), (128, 128, 64)])
@pytest.mark.parametrize("a_dtype,b_dtype,k_dim", [
    ("bfloat16", "bfloat16", 32),
    ("float8_e5m2", "float8_e5m2", 64),
])
@pytest.mark.parametrize("ASYNC_LOAD_TYPE", ["ASYNC_COPY", "TDM"])
def test_runtime_gemm_async(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, a_dtype, b_dtype, k_dim, ASYNC_LOAD_TYPE):
    if BLOCK_K < k_dim:
        pytest.skip("Skip tests where BLOCK_K < k_dim")
    if ASYNC_LOAD_TYPE == "ASYNC_COPY" and any([x % 16 != 0 for x in [M, N, K]]):
        pytest.skip("AsyncCopy tests need divisibility==16 to get vectorization information")

    torch.manual_seed(42)

    def create_operand(shape, dtype):
        if dtype == torch.bfloat16:
            return torch.randn(shape, dtype=dtype)
        else:
            assert dtype == torch.float8_e5m2
            return torch.randint(0x04, 0x7B, shape, dtype=torch.uint8).view(dtype)

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
    gemm_async_kernel[grid](
        a_device, b_device, c_device,  #
        M, N, K,  #
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,  #
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,  #
        INSTR_SHAPE_K=k_dim, K_WIDTH=8, USE_TDM=ASYNC_LOAD_TYPE == "TDM")

    c_triton = c_device.cpu()
    c_torch = a.to(torch.float32) @ b.to(torch.float32)
    torch.testing.assert_close(c_triton, c_torch, rtol=1e-4, atol=1e-4)


def torch_gemm_mxfp(a, b, a_scale, b_scale, scale_block, M, N, K):
    a_scale_f32 = a_scale.to(torch.float32).repeat_interleave(scale_block, dim=1)[:M, :K]
    b_scale_f32 = b_scale.to(torch.float32).repeat_interleave(scale_block, dim=1).T.contiguous()[:K, :N]

    a_f32 = a.to(torch.float32)
    b_f32 = b.to(torch.float32)

    return torch.matmul(a_f32 * a_scale_f32, b_f32 * b_scale_f32).to(torch.float32)


def create_mxfp_operand(operand: int, m: int, n: int, dtype: str):
    size = (m, n)
    if dtype == 'e4m3':
        v = torch.randint(20, 40, size, dtype=torch.uint8)
        v_ref = v.view(torch.float8_e4m3fn).to(torch.float32)
    elif dtype == 'e5m2':
        v = torch.randint(20, 40, size, dtype=torch.uint8)
        v_ref = v.view(torch.float8_e5m2).to(torch.float32)
    else:
        assert dtype == 'e2m1'
        pack_dim = 1 if operand == 0 else 0
        v_mxfp4 = MXFP4Tensor(size=size).random()
        v = v_mxfp4.to_packed_tensor(pack_dim)
        v_ref = v_mxfp4.to(torch.float32)
    return v, v_ref


def create_mxfp_scale(operand: int, m: int, n: int):
    pack_dim = 1 if operand == 0 else 0
    size = (m, n // 32) if pack_dim == 1 else (m // 32, n)
    scale = MXScaleTensor(size=tuple(size)).random(1 / 32, 32)
    scale_ref = scale.to(torch.float32).repeat_interleave(32, dim=pack_dim)
    return scale.data, scale_ref


def get_test_mxfp_block_mnk():
    return [(m, n, k) for m, n in [(16, 16), (32, 32), (64, 64)] for k in [64, 128, 256]]


def get_test_mxfp_variants():
    types = ["e2m1", "e4m3", "e5m2"]
    return [(a_type, b_type) for a_type in types for b_type in types]


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires GFX1250")
@pytest.mark.parametrize("M, N, K", get_test_mxfp_block_mnk())
@pytest.mark.parametrize("a_type, b_type", get_test_mxfp_variants())
def test_amd_wmma_scaled(M, N, K, a_type, b_type):

    @aggregate
    class Layout:
        load_a: ttgl.constexpr
        load_b: ttgl.constexpr
        load_scale: ttgl.constexpr
        a: ttgl.constexpr
        b: ttgl.constexpr
        a_scale: ttgl.constexpr
        b_scale: ttgl.constexpr
        acc: ttgl.constexpr

        @gluon.constexpr_function
        def _get_scale_layout(operand, scale_nonk, scale_k):
            # TODO: generalize scale layout generation
            assert scale_nonk in [16, 32, 64] and scale_k in [2, 4, 8]
            scale_reg = [[0, 1], [0, 2]]
            if scale_k == 2:
                scale_reg[1] = [0, 0]
            if scale_k == 8:
                scale_reg.append([0, 4])
            if scale_nonk == 64:
                scale_reg.append([32, 0])

            scale_lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 0]]

            scale_warp = [[0, 0], [16, 0]] if operand == 0 else [[16, 0], [0, 0]]
            if scale_nonk == 16:
                scale_warp = [[0, 0], [0, 0]]

            scale_shape = [scale_nonk, scale_k]

            return ttgl.DistributedLinearLayout(scale_reg, scale_lane, scale_warp, [], scale_shape)

        @gluon.constexpr_function
        def __init__(self, a_type, b_type, scale_nonk, scale_k):
            self.load_a = ttgl.constexpr(ttgl.BlockedLayout([1, 16], [8, 4], [4, 1], [1, 0]))
            self.load_b = ttgl.constexpr(ttgl.BlockedLayout([1, 16], [16, 2], [4, 1], [1, 0]))
            self.load_scale = ttgl.constexpr(ttgl.BlockedLayout([1, 1], [8, 4], [4, 1], [1, 0]))

            wmma_layout = ttgl.amd.AMDWMMALayout(version=3, transposed=True, warps_per_cta=[2, 2],
                                                 instr_shape=[16, 16, 128])
            wmma_layout_packed = ttgl.amd.AMDWMMALayout(version=3, transposed=True, warps_per_cta=[2, 2],
                                                        instr_shape=[16, 16, 64])
            a_layout = ttgl.DotOperandLayout(0, wmma_layout_packed if a_type == "e2m1" else wmma_layout, k_width=16)
            b_layout = ttgl.DotOperandLayout(1, wmma_layout_packed if b_type == "e2m1" else wmma_layout, k_width=16)
            self.a = ttgl.constexpr(a_layout)
            self.b = ttgl.constexpr(b_layout)
            self.a_scale = ttgl.constexpr(Layout._get_scale_layout(0, scale_nonk, scale_k))
            self.b_scale = ttgl.constexpr(Layout._get_scale_layout(1, scale_nonk, scale_k))

            self.acc = ttgl.constexpr(wmma_layout)

    @gluon.jit
    def kernel(c_ptr, a_ptr, a_scale_ptr, b_ptr, b_scale_ptr,  #
               a_type: ttgl.constexpr, b_type: ttgl.constexpr,  #
               BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr, BLOCK_K: ttgl.constexpr):
        DIV_FACTOR_A: ttgl.constexpr = 2 if a_type == "e2m1" else 1
        DIV_FACTOR_B: ttgl.constexpr = 2 if b_type == "e2m1" else 1

        ttgl.static_assert(BLOCK_M == BLOCK_N)
        layout: ttgl.constexpr = Layout(a_type, b_type, BLOCK_M, BLOCK_K // 32)

        offs_a_m = ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, layout.load_a))
        offs_a_k = ttgl.arange(0, BLOCK_K // DIV_FACTOR_A, layout=ttgl.SliceLayout(0, layout.load_a))
        offs_a = offs_a_m[:, None] * (BLOCK_K // DIV_FACTOR_A) + offs_a_k[None, :]
        a = ttgl.load(a_ptr + offs_a)
        a = ttgl.convert_layout(a, layout.a)

        offs_b_k = ttgl.arange(0, BLOCK_K // DIV_FACTOR_B, layout=ttgl.SliceLayout(1, layout.load_b))
        offs_b_n = ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, layout.load_b))
        offs_b = offs_b_k[:, None] * BLOCK_N + offs_b_n[None, :]
        b = ttgl.load(b_ptr + offs_b)
        b = ttgl.convert_layout(b, layout.b)

        offs_a_scale_m = ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, layout.load_scale))
        offs_a_scale_k = ttgl.arange(0, BLOCK_K // 32, layout=ttgl.SliceLayout(0, layout.load_scale))
        offs_a_scale = offs_a_scale_m[:, None] * (BLOCK_K // 32) + offs_a_scale_k[None, :]
        a_scale = ttgl.load(a_scale_ptr + offs_a_scale)
        a_scale = ttgl.convert_layout(a_scale, layout.a_scale)

        offs_b_scale_n = ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(1, layout.load_scale))
        offs_b_scale_k = ttgl.arange(0, BLOCK_K // 32, layout=ttgl.SliceLayout(0, layout.load_scale))
        offs_b_scale = offs_b_scale_n[:, None] * (BLOCK_K // 32) + offs_b_scale_k[None, :]
        b_scale = ttgl.load(b_scale_ptr + offs_b_scale)
        b_scale = ttgl.convert_layout(b_scale, layout.b_scale)

        zero = ttgl.zeros([BLOCK_M, BLOCK_N], dtype=ttgl.float32, layout=layout.acc)
        c = ttgl.amd.gfx1250.wmma_scaled(a, a_scale, a_type, b, b_scale, b_type, zero)
        c = c.to(c_ptr.dtype.element_ty)

        offs_cm = ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, layout.acc))
        offs_cn = ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, layout.acc))
        offs_c = offs_cm[:, None] * BLOCK_N + offs_cn[None, :]
        ttgl.store(c_ptr + offs_c, c)

    torch.manual_seed(0)
    a, a_ref = create_mxfp_operand(0, M, K, a_type)
    b, b_ref = create_mxfp_operand(1, K, N, b_type)
    a_scale, a_scale_ref = create_mxfp_scale(0, M, K)
    b_scale, b_scale_ref = create_mxfp_scale(1, K, N)
    b_scale = b_scale.permute(1, 0).contiguous()

    a, a_scale = a.cuda(), a_scale.cuda()
    b, b_scale = b.cuda(), b_scale.cuda()
    c = torch.zeros((M, N), dtype=torch.float32).cuda()
    pgm = kernel[(1, )](c, a, a_scale, b, b_scale, a_type, b_type, M, N, K, num_warps=4)
    assert "v_wmma_scale_f32_16x16x128_f8f6f4" in pgm.asm["amdgcn"]

    c_torch = (a_ref * a_scale_ref) @ (b_ref * b_scale_ref)
    torch.testing.assert_close(c.cpu(), c_torch, atol=1e-5, rtol=2e-5)


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires GFX1250")
@pytest.mark.parametrize("M, N, K", [(16, 16, 128), (32, 32, 128), (32, 32, 256), (32, 32, 512), (64, 64, 128),
                                     (128, 128, 256)])
@pytest.mark.parametrize("mxfp_type", ["e2m1"])
@pytest.mark.parametrize("hasScale", [True, False])
def test_amd_wmma_scaled_tdm(M, N, K, mxfp_type, hasScale):

    @triton.jit
    def scaled_wmma_tdm_triton_kernel(a_base, stride_am, stride_ak, a_scale, b_base, stride_bk, stride_bn, b_scale, out,
                                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                                      type_a: tl.constexpr, type_b: tl.constexpr):
        DIV_FACTOR_A: tl.constexpr = 2 if type_a == "e2m1" else 1
        DIV_FACTOR_B: tl.constexpr = 2 if type_b == "e2m1" else 1
        PACKED_BLOCK_K_A: tl.constexpr = BLOCK_K // DIV_FACTOR_A
        PACKED_BLOCK_K_B: tl.constexpr = BLOCK_K // DIV_FACTOR_B
        a_desc = tl.make_tensor_descriptor(base=a_base, shape=(BLOCK_M, PACKED_BLOCK_K_A),
                                           strides=(stride_am, stride_ak), block_shape=(BLOCK_M, PACKED_BLOCK_K_A))
        b_desc = tl.make_tensor_descriptor(base=b_base, shape=(PACKED_BLOCK_K_B, BLOCK_N),
                                           strides=(stride_bk, stride_bn), block_shape=(PACKED_BLOCK_K_B, BLOCK_N))
        a = a_desc.load([0, 0])
        b = b_desc.load([0, 0])
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

    @gluon.jit
    def scaled_wmma_tdm_gluon_kernel(a_base, stride_am, stride_ak, a_scale, b_base, stride_bk, stride_bn, b_scale, out,
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
            reg_bases=[[0, 1], [0, 2]], lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [0, 0]],
            warp_bases=[[0, 0], [16, 0]], block_bases=[], shape=[32, 4])
        b_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 16], [16, 2], [4, 1], [1, 0])
        b_scale_linear_layout: ttgl.constexpr = ttgl.DistributedLinearLayout(
            reg_bases=[[0, 1], [0, 2]], lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [0, 0]],
            warp_bases=[[16, 0], [0, 0]], block_bases=[], shape=[32, 4])
        SHARED_LAYOUT_A: ttgl.constexpr = ttgl.PaddedSharedLayout.with_identity_for([[32, 4]],
                                                                                    [BLOCK_M, PACKED_BLOCK_K_A], [1, 0])
        SHARED_LAYOUT_B: ttgl.constexpr = ttgl.PaddedSharedLayout.with_identity_for([[32, 4]],
                                                                                    [PACKED_BLOCK_K_B, BLOCK_N], [1, 0])

        wmma_layout: ttgl.constexpr = ttgl.amd.AMDWMMALayout(version=3, transposed=True, warps_per_cta=[2, 2],
                                                             instr_shape=[16, 16, 128])
        wmma_layout_packed: ttgl.constexpr = ttgl.amd.AMDWMMALayout(version=3, transposed=True, warps_per_cta=[2, 2],
                                                                    instr_shape=[16, 16, 64])

        zero = ttgl.zeros([BLOCK_M, BLOCK_N], dtype=ttgl.float32, layout=wmma_layout)

        a_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(base=a_base, shape=(BLOCK_M, PACKED_BLOCK_K_A),
                                                             strides=(stride_am, stride_ak),
                                                             block_shape=(BLOCK_M, PACKED_BLOCK_K_A),
                                                             layout=SHARED_LAYOUT_A)
        a_buffer = ttgl.allocate_shared_memory(a_desc.dtype, shape=a_desc.block_shape, layout=a_desc.layout)
        ttgl.amd.gfx1250.tdm.async_load(a_desc, [0, 0], a_buffer)
        ttgl.amd.gfx1250.tdm.async_wait(0)
        a = a_buffer.load(layout=a_layout)
        a = ttgl.convert_layout(
            a,
            ttgl.DotOperandLayout(operand_index=0, parent=wmma_layout_packed if type_a == "e2m1" else wmma_layout,
                                  k_width=16))

        b_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(base=b_base, shape=(PACKED_BLOCK_K_B, BLOCK_N),
                                                             strides=(stride_bk, stride_bn),
                                                             block_shape=(PACKED_BLOCK_K_B, BLOCK_N),
                                                             layout=SHARED_LAYOUT_B)
        b_buffer = ttgl.allocate_shared_memory(b_desc.dtype, shape=b_desc.block_shape, layout=b_desc.layout)
        ttgl.amd.gfx1250.tdm.async_load(b_desc, [0, 0], b_buffer)
        ttgl.amd.gfx1250.tdm.async_wait(0)
        b = b_buffer.load(layout=b_layout)
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

    torch.manual_seed(0)

    type_a = mxfp_type
    type_b = mxfp_type

    DIV_FACTOR_A = 2 if type_a == "e2m1" else 1
    DIV_FACTOR_B = 2 if type_b == "e2m1" else 1

    x = torch.randint(20, 40, (M, K // DIV_FACTOR_A), dtype=torch.uint8).cuda()
    y = torch.randint(20, 40, (K // DIV_FACTOR_B, N), dtype=torch.uint8).cuda()

    if hasScale:
        min_scale, max_scale = (0, 142)
        scale_x = torch.randint(min_scale, max_scale + 1, (M, K // 32), dtype=torch.uint8).cuda()
        scale_y = torch.randint(min_scale, max_scale + 1, (N, K // 32), dtype=torch.uint8).cuda()
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

    z = torch.zeros((M, N), dtype=torch.float32).cuda()
    pgm = scaled_wmma_tdm_gluon_kernel[(1, )](x, *x.stride(), scale_x, y, *y.stride(), scale_y, z, M, N, K, type_a,
                                              type_b)
    amdgcn = pgm.asm["amdgcn"]

    patterns = (
        "tensor_load_to_lds",
        "s_wait_tensorcnt 0x0",
    )
    for pattern in patterns:
        assert re.search(pattern, amdgcn), f"Can't find {pattern} in amdgcn"

    z_ref = torch.zeros((M, N), dtype=torch.float32).cuda()
    scaled_wmma_tdm_triton_kernel[(1, )](x, *x.stride(), scale_x, y, *y.stride(), scale_y, z_ref, M, N, K, type_a,
                                         type_b)

    torch.testing.assert_close(z.cpu(), z_ref.cpu(), rtol=1e-5, atol=1e-5)


@gluon.jit
def tensor_async_copy_kernel(a_ptr, b_ptr, M, N,  #
                             BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr, NUM_BUFFERS: ttgl.constexpr):
    num_warps: ttgl.constexpr = ttgl.num_warps()
    smem_layout: ttgl.constexpr = ttgl.PaddedSharedLayout.with_identity_for([[32, 4]], [BLOCK_M, BLOCK_N], [1, 0])
    block_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 8], [4, 8], [num_warps, 1], [1, 0])

    pid_m = ttgl.program_id(axis=0)
    pid_n = ttgl.program_id(axis=1)

    a_buffer = ttgl.allocate_shared_memory(a_ptr.type.element_ty, [NUM_BUFFERS, BLOCK_M, BLOCK_N], smem_layout)

    idx_m = pid_m * BLOCK_M
    for i in ttgl.static_range(0, NUM_BUFFERS):
        idx_n = pid_n * (BLOCK_N * NUM_BUFFERS) + i * BLOCK_N

        offs_am = idx_m + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, block_layout))
        offs_an = idx_n + ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, block_layout))
        a_ptrs = a_ptr + offs_am[:, None] * N + offs_an[None, :]
        a_mask = (offs_am[:, None] < M) & (offs_an[None, :] < N)
        ttgl.amd.gfx1250.async_copy.global_to_shared(a_buffer.index(i), a_ptrs, a_mask, other=0.0)
        ttgl.amd.gfx1250.async_copy.commit_group()

    ttgl.amd.gfx1250.async_copy.wait_group(0)

    for i in ttgl.static_range(0, NUM_BUFFERS):
        idx_n = pid_n * (BLOCK_N * NUM_BUFFERS) + i * BLOCK_N
        a = a_buffer.index(i).load(layout=block_layout)

        offs_bm = idx_m + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, block_layout))
        offs_bn = idx_n + ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, block_layout))
        offs_b = (offs_bm[:, None] * N) + offs_bn[None, :]
        b_mask = (offs_bm[:, None] < M) & (offs_bn[None, :] < N)
        ttgl.store(b_ptr + offs_b, a, mask=b_mask)


@gluon.jit
def tensor_device_tdm_copy_kernel(a_ptr, b_ptr, M, N,  #
                                  BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr, NUM_BUFFERS: ttgl.constexpr):
    num_warps: ttgl.constexpr = ttgl.num_warps()
    smem_layout: ttgl.constexpr = ttgl.PaddedSharedLayout.with_identity_for([[32, 4]], [BLOCK_M, BLOCK_N], [1, 0])
    block_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 8], [4, 8], [num_warps, 1], [1, 0])

    pid_m = ttgl.program_id(axis=0)
    pid_n = ttgl.program_id(axis=1)

    a_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(base=a_ptr, shape=(M, N), strides=(N, 1),
                                                         block_shape=(BLOCK_M, BLOCK_N), layout=smem_layout)
    a_buffer = ttgl.allocate_shared_memory(a_desc.dtype, [NUM_BUFFERS] + a_desc.block_shape, a_desc.layout)

    idx_m = pid_m * BLOCK_M
    for i in ttgl.static_range(0, NUM_BUFFERS):
        idx_n = pid_n * (BLOCK_N * NUM_BUFFERS) + i * BLOCK_N
        ttgl.amd.gfx1250.tdm.async_load(a_desc, [idx_m, idx_n], a_buffer.index(i))

    ttgl.amd.gfx1250.tdm.async_wait(0)

    for i in ttgl.static_range(0, NUM_BUFFERS):
        idx_n = pid_n * (BLOCK_N * NUM_BUFFERS) + i * BLOCK_N
        a = a_buffer.index(i).load(layout=block_layout)

        offs_bm = idx_m + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, block_layout))
        offs_bn = idx_n + ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, block_layout))
        offs_b = (offs_bm[:, None] * N) + offs_bn[None, :]
        b_mask = (offs_bm[:, None] < M) & (offs_bn[None, :] < N)
        ttgl.store(b_ptr + offs_b, a, mask=b_mask)


@gluon.jit
def tensor_host_tdm_copy_kernel(a_desc, b_ptr, M, N,  #
                                NUM_BUFFERS: ttgl.constexpr):
    num_warps: ttgl.constexpr = ttgl.num_warps()
    block_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 8], [4, 8], [num_warps, 1], [1, 0])

    pid_m = ttgl.program_id(axis=0)
    pid_n = ttgl.program_id(axis=1)

    a_buffer = ttgl.allocate_shared_memory(a_desc.dtype, [NUM_BUFFERS] + a_desc.block_shape, a_desc.layout)
    BLOCK_M: ttgl.constexpr = a_desc.block_shape[0]
    BLOCK_N: ttgl.constexpr = a_desc.block_shape[1]

    idx_m = pid_m * BLOCK_M
    for i in ttgl.static_range(0, NUM_BUFFERS):
        idx_n = pid_n * (BLOCK_N * NUM_BUFFERS) + i * BLOCK_N
        ttgl.amd.gfx1250.tdm.async_load(a_desc, [idx_m, idx_n], a_buffer.index(i))

    ttgl.amd.gfx1250.tdm.async_wait(0)

    for i in ttgl.static_range(0, NUM_BUFFERS):
        idx_n = pid_n * (BLOCK_N * NUM_BUFFERS) + i * BLOCK_N
        a = a_buffer.index(i).load(layout=block_layout)

        offs_bm = idx_m + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, block_layout))
        offs_bn = idx_n + ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, block_layout))
        offs_b = (offs_bm[:, None] * N) + offs_bn[None, :]
        b_mask = (offs_bm[:, None] < M) & (offs_bn[None, :] < N)
        ttgl.store(b_ptr + offs_b, a, mask=b_mask)


@pytest.mark.parametrize("BLOCK_M,BLOCK_N", [(32, 32), (32, 64), (64, 64), (1, 512), (256, 2)])
@pytest.mark.parametrize("NUM_BUFFERS", [2])
@pytest.mark.parametrize("NUM_WARPS", [4, 8])
@pytest.mark.parametrize("ASYNC_LOAD_TYPE", ["ASYNC_COPY", "DEVICE_TDM", "HOST_TDM"])
def test_compile_tensor_copy(BLOCK_M, BLOCK_N, NUM_BUFFERS, ASYNC_LOAD_TYPE, NUM_WARPS):
    attrs = None
    if ASYNC_LOAD_TYPE == "ASYNC_COPY":
        # AsyncCopy requires >= 32 bits per lane so we have to pass divisibility for arguments
        attrs = {k: [["tt.divisibility", 16]] for k in [(x, ) for x in range(4)]}
        fn = tensor_async_copy_kernel
        signature = {
            "a_ptr": "*fp16",
            "b_ptr": "*fp16",
            "M": "i32",
            "N": "i32",
            "BLOCK_M": "constexpr",
            "BLOCK_N": "constexpr",
            "NUM_BUFFERS": "constexpr",
        }
        constexprs = {"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "NUM_BUFFERS": NUM_BUFFERS}
    elif "DEVICE_TDM":
        fn = tensor_device_tdm_copy_kernel
        signature = {
            "a_ptr": "*fp16",
            "b_ptr": "*fp16",
            "M": "i32",
            "N": "i32",
            "BLOCK_M": "constexpr",
            "BLOCK_N": "constexpr",
            "NUM_BUFFERS": "constexpr",
        }
        constexprs = {"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "NUM_BUFFERS": NUM_BUFFERS}
    else:
        assert ASYNC_LOAD_TYPE == "HOST_TDM"
        fn = tensor_host_tdm_copy_kernel
        smem_layout = ttgl.PaddedSharedLayout.with_identity_for([[32, 4]], [BLOCK_M, BLOCK_N], [1, 0])
        signature = {
            "a_desc": f"tensordesc<fp16[{BLOCK_M}, {BLOCK_N}],{smem_layout}>",
            "b_ptr": "*fp16",
            "M": "i32",
            "N": "i32",
            "NUM_BUFFERS": "constexpr",
        }
        constexprs = {"NUM_BUFFERS": NUM_BUFFERS}

    k = triton.compile(
        gluon._runtime.GluonASTSource(fn, signature, constexprs, attrs),
        target=GPUTarget("hip", 'gfx1250', 32),
        options={"num_warps": NUM_WARPS},
    )
    amdgcn = k.asm["amdgcn"]

    if ASYNC_LOAD_TYPE in {"DEVICE_TDM", "HOST_TDM"}:
        pattern = {"tensor_load_to_lds", "s_wait_tensorcnt 0x0"}
    else:
        ASYNC_LOAD_TYPE == "ASYNC_COPY"
        pattern = {"global_load_async_to_lds", "s_wait_asynccnt 0x0"}
    for p in pattern:
        assert re.search(p, amdgcn), f"Can't find {p} in amdgcn"


@pytest.mark.parametrize("BLOCK_M,BLOCK_N", [(32, 32), (32, 64), (64, 64), (1, 512), (256, 2)])
@pytest.mark.parametrize("NUM_BUFFERS", [2])
@pytest.mark.parametrize("NUM_WARPS", [4, 8])
@pytest.mark.parametrize("ASYNC_LOAD_TYPE", ["ASYNC_COPY", "DEVICE_TDM", "HOST_TDM"])
@pytest.mark.parametrize("M,N", [(1024, 1024), (1008, 1008)])
def test_runtime_tensor_copy(M, N, BLOCK_M, BLOCK_N, NUM_BUFFERS, ASYNC_LOAD_TYPE, NUM_WARPS):
    if ASYNC_LOAD_TYPE == "ASYNC_COPY" and any([x % 16 != 0 for x in [M, N]]):
        pytest.skip("AsyncCopy tests need divisibility==16 to get vectorization information")

    torch.manual_seed(42)
    a = torch.randint(0x0, 0xFFFF, (M, N), dtype=torch.uint16)
    b = torch.zeros_like(a)

    a_device = a.cuda()
    b_device = b.cuda()
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N * NUM_BUFFERS))
    if ASYNC_LOAD_TYPE == "ASYNC_COPY":
        tensor_async_copy_kernel[grid](a_device, b_device, M, N, BLOCK_M, BLOCK_N, NUM_BUFFERS, num_warps=NUM_WARPS)
    elif ASYNC_LOAD_TYPE == "DEVICE_TDM":
        tensor_device_tdm_copy_kernel[grid](a_device, b_device, M, N, BLOCK_M, BLOCK_N, NUM_BUFFERS,
                                            num_warps=NUM_WARPS)
    else:
        assert ASYNC_LOAD_TYPE == "HOST_TDM"
        smem_layout = ttgl.PaddedSharedLayout.with_identity_for([[32, 4]], [BLOCK_M, BLOCK_N], [1, 0])
        a_desc = gluon.amd.gfx1250.TensorDescriptor.from_tensor(a_device, [BLOCK_M, BLOCK_N], layout=smem_layout)
        tensor_host_tdm_copy_kernel[grid](a_desc, b_device, M, N, NUM_BUFFERS, num_warps=NUM_WARPS)

    b_triton = b_device.cpu()
    assert torch.equal(b_triton, a)


@gluon.jit
def tensor_device_tdm_multi_cta_load_and_store_kernel(a_ptr, b_ptr, M, N,  #
                                                      BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr,
                                                      block_layout: ttgl.constexpr, smem_layout: ttgl.constexpr,
                                                      USE_TDM_LOAD: ttgl.constexpr, USE_TDM_STORE: ttgl.constexpr):
    pid_m = ttgl.program_id(axis=0)
    pid_n = ttgl.program_id(axis=1)
    idx_m = pid_m * BLOCK_M
    idx_n = pid_n * BLOCK_N

    a_buffer = ttgl.allocate_shared_memory(a_ptr.type.element_ty, (BLOCK_M, BLOCK_N), smem_layout)

    # Load data - either using TDM load or async_copy
    if USE_TDM_LOAD:
        a_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(base=a_ptr, shape=(M, N), strides=(N, 1),
                                                             block_shape=(BLOCK_M, BLOCK_N), layout=smem_layout)
        ttgl.amd.gfx1250.tdm.async_load(a_desc, [idx_m, idx_n], a_buffer)
        ttgl.amd.gfx1250.tdm.async_wait(0)
    else:
        offs_am = idx_m + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, block_layout))
        offs_an = idx_n + ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, block_layout))
        offs_a = (offs_am[:, None] * N) + offs_an[None, :]
        a_mask = (offs_am[:, None] < M) & (offs_an[None, :] < N)
        a_ptrs = a_ptr + offs_a
        ttgl.amd.gfx1250.async_copy.global_to_shared(a_buffer, a_ptrs, a_mask)
        ttgl.amd.gfx1250.async_copy.commit_group()
        ttgl.amd.gfx1250.async_copy.wait_group(0)

    # Store data - either using TDM store or local_load + store
    if USE_TDM_STORE:
        b_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(base=b_ptr, shape=(M, N), strides=(N, 1),
                                                             block_shape=(BLOCK_M, BLOCK_N), layout=smem_layout)
        ttgl.amd.gfx1250.tdm.async_store(b_desc, [idx_m, idx_n], a_buffer)
        ttgl.amd.gfx1250.tdm.async_wait(0)
    else:
        a = a_buffer.load(layout=block_layout)
        offs_bm = idx_m + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, block_layout))
        offs_bn = idx_n + ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, block_layout))
        offs_b = (offs_bm[:, None] * N) + offs_bn[None, :]
        b_mask = (offs_bm[:, None] < M) & (offs_bn[None, :] < N)
        ttgl.store(b_ptr + offs_b, a, mask=b_mask)


@pytest.mark.parametrize("USE_TDM_LOAD, USE_TDM_STORE", [(True, False), (False, True)])
@pytest.mark.parametrize("BLOCK_M,BLOCK_N", [(64, 64), (128, 64)])
@pytest.mark.parametrize("NUM_WARPS", [1, 4])
@pytest.mark.parametrize("M,N", [(64, 64), (576, 576)])
@pytest.mark.parametrize("CGALayout", [
    [[0, 0]],
    [[0, 1]],
    [[1, 0]],
    [[0, 1], [0, 0], [1, 0], [0, 0]],
    [[0, 1], [0, 2], [0, 0], [0, 0]],
    [[1, 0], [2, 0], [4, 0], [0, 0]],
])
def test_runtime_tensor_load_and_store_multi_cta(M, N, BLOCK_M, BLOCK_N, NUM_WARPS, CGALayout, USE_TDM_LOAD,
                                                 USE_TDM_STORE):
    torch.manual_seed(42)
    a = torch.randint(0x0, 0xFFFF, (M, N), dtype=torch.uint16)
    b = torch.zeros_like(a)

    a_device = a.cuda()
    b_device = b.cuda()
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    num_ctas = 2**len(CGALayout)
    smem_layout: ttgl.constexpr = ttgl.SwizzledSharedLayout(1, 1, 1, [1, 0], CGALayout)
    block_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 8], [4, 8], [NUM_WARPS, 1], [1, 0], CGALayout)
    tensor_device_tdm_multi_cta_load_and_store_kernel[grid](a_device, b_device, M, N, BLOCK_M, BLOCK_N, block_layout,
                                                            smem_layout, USE_TDM_LOAD, USE_TDM_STORE,
                                                            num_warps=NUM_WARPS, num_ctas=num_ctas)

    assert torch.equal(a, b_device.cpu())


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
def tensor_descriptor_load_store_nd_kernel_device_tdm(out_ptr, a_ptr, shape, strides, BLOCK_SHAPE, out_shape,
                                                      out_strides, SHARED_LAYOUT: ttgl.constexpr):
    ndim: ttgl.constexpr = len(BLOCK_SHAPE)
    desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(base=a_ptr, shape=shape, strides=strides,
                                                       block_shape=BLOCK_SHAPE, layout=SHARED_LAYOUT)

    offs = (0, ) * ndim
    block_shared = ttgl.allocate_shared_memory(desc.dtype, shape=desc.block_shape, layout=desc.layout)
    ttgl.amd.gfx1250.tdm.async_load(desc, offs, block_shared)
    ttgl.amd.gfx1250.tdm.async_wait(0)

    out_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(base=out_ptr, shape=out_shape, strides=out_strides,
                                                           block_shape=BLOCK_SHAPE, layout=SHARED_LAYOUT)

    ttgl.amd.gfx1250.tdm.async_store(out_desc, offs, block_shared)
    ttgl.amd.gfx1250.tdm.async_wait(0)


@gluon.jit
def tensor_descriptor_load_store_nd_kernel_host_tdm(out_desc, inp_desc):
    ndim: ttgl.constexpr = len(inp_desc.block_shape)
    offs = (0, ) * ndim
    block_shared = ttgl.allocate_shared_memory(inp_desc.dtype, shape=inp_desc.block_shape, layout=inp_desc.layout)
    ttgl.amd.gfx1250.tdm.async_load(inp_desc, offs, block_shared)
    ttgl.amd.gfx1250.tdm.async_wait(0)

    ttgl.amd.gfx1250.tdm.async_store(out_desc, offs, block_shared)
    ttgl.amd.gfx1250.tdm.async_wait(0)


@pytest.mark.parametrize("ndim", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("INNER_BLOCK", [4, 8, 16, 32, 64, 128])
@pytest.mark.parametrize("dtype_str", sorted(set(dtypes_with_bfloat16) - {"int64", "uint64", "float64"}))
@pytest.mark.parametrize("TDM_TYPE", ["DEVICE_TDM", "HOST_TDM"])
def test_tensor_descriptor_load_store_nd(dtype_str, ndim, INNER_BLOCK, TDM_TYPE):
    SHARED_LAYOUT: ttgl.constexpr = ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1,
                                                              order=[ndim - 1 - i for i in range(ndim)])

    alloc_shape = [1, 1, 3, 7, INNER_BLOCK][-ndim:]

    BLOCK_SHAPE = (2, 2, 4, 8, INNER_BLOCK)[-ndim:]
    inp = to_triton(numpy_random(alloc_shape, dtype_str), device="cpu", dst_type=dtype_str)
    inp.data = inp.data[..., :INNER_BLOCK - 3]
    out = inp.new_empty(BLOCK_SHAPE)
    # uint_dtypes require special handling because PyTorch only has full native support
    # for uint8. While PyTorch 2.1+ added limited support for uint16, uint32, and uint64,
    # they still lack complete functionality across all PyTorch ops. They are stored as
    # signed tensors with the same bit width and wrapped in TensorWrapper for reinterpretation
    # to unsigned. The .base attribute accesses the underlying signed tensor for CUDA transfer.
    if dtype_str in uint_dtypes:
        inp.base = inp.base.cuda()
        out.base = out.base.cuda()
    else:
        inp = inp.cuda()
        out = out.cuda()

    if TDM_TYPE == "DEVICE_TDM":
        constexpr_block_shape = tuple(ttgl.constexpr(v) for v in BLOCK_SHAPE)
        k = tensor_descriptor_load_store_nd_kernel_device_tdm[(1, )](out, inp, inp.shape,
                                                                     inp.stride(), constexpr_block_shape, out.shape,
                                                                     out.stride(), SHARED_LAYOUT)
    else:
        assert TDM_TYPE == "HOST_TDM"
        inp_desc = gluon.amd.gfx1250.TensorDescriptor.from_tensor(inp, list(BLOCK_SHAPE), layout=SHARED_LAYOUT)
        out_desc = gluon.amd.gfx1250.TensorDescriptor.from_tensor(out, list(BLOCK_SHAPE), layout=SHARED_LAYOUT)
        k = tensor_descriptor_load_store_nd_kernel_host_tdm[(1, )](out_desc, inp_desc)

    amdgcn = k.asm["amdgcn"]
    for pattern in ("tensor_load_to_lds", "tensor_store_from_lds", "s_wait_tensorcnt 0x0"):
        assert re.search(pattern, amdgcn)

    # Check in-bounds
    actual = unwrap_tensor(out.cpu())
    expect = unwrap_tensor(inp.cpu())
    idx = tuple(slice(None, s) for s in inp.shape)
    assert torch.equal(expect, actual[idx])

    # Check out-of-bounds
    actual[idx].zero_()
    expect = expect.new_zeros(BLOCK_SHAPE)
    assert torch.equal(expect, actual)


def test_tensor_descriptor_load_store_invalid_blocksize():
    """Test that TDM operations fail when block size exceeds 2^16 (65536)"""
    ndim = 2
    INNER_BLOCK = 2**17  # 131072, exceeds 2^16 limit
    dtype_str = 'float32'

    SHARED_LAYOUT: ttgl.constexpr = ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1,
                                                              order=[ndim - 1 - i for i in range(ndim)])

    alloc_shape = [7, INNER_BLOCK]
    BLOCK_SHAPE = (8, INNER_BLOCK)

    inp = to_triton(numpy_random(alloc_shape, dtype_str), device="cpu", dst_type=dtype_str)
    inp.data = inp.data[..., :INNER_BLOCK - 3]
    out = inp.new_empty(BLOCK_SHAPE)
    inp = inp.cuda()
    out = out.cuda()

    constexpr_block_shape = tuple(ttgl.constexpr(v) for v in BLOCK_SHAPE)

    # Expect compilation to fail due to block size exceeding maximum
    try:
        tensor_descriptor_load_store_nd_kernel_device_tdm[(1, )](out, inp, inp.shape,
                                                                 inp.stride(), constexpr_block_shape, out.shape,
                                                                 out.stride(), SHARED_LAYOUT)
        pytest.fail(
            f"Expected compilation to fail for block size {INNER_BLOCK} (2^17) > 65536 (2^16), but it succeeded")
    except Exception as e:
        error_msg = str(e)
        assert "error encountered during parsing" in error_msg.lower(), \
            f"Expected parsing error for block size > 65536, but got: {error_msg}"


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


@pytest.mark.parametrize("BLOCK_M, BLOCK_N, BLOCK_K", [(32, 32, 128), (64, 64, 128), (64, 64, 64)])
@pytest.mark.parametrize("DTYPE_A", ["float8_e5m2", "float8_e4m3", "float4"])
@pytest.mark.parametrize("DTYPE_B", ["float8_e5m2", "float8_e4m3", "float4"])
def test_compile_mxgemm(BLOCK_M, BLOCK_N, BLOCK_K, DTYPE_A, DTYPE_B):
    scale_block = 32

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
def cluster_load_and_write_back_kernel(a_ptr, out_ptr, M, N, BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr,
                                       blocked_layout: ttgl.constexpr):
    pid = ttgl.program_id(axis=0)
    num_pid_m = ttgl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    offs_m = pid_m * BLOCK_M + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, blocked_layout))
    offs_n = pid_n * BLOCK_N + ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, blocked_layout))

    a_ptrs = a_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    a = ttgl.load(a_ptrs, mask)

    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    ttgl.store(out_ptrs, a, mask)


@pytest.mark.parametrize("blocked_layout", [
    ttgl.BlockedLayout(size_per_thread=[1, 8], threads_per_warp=[4, 8], warps_per_cta=[1, 2], order=[1, 0],
                       cga_layout=[[0, 1]]),
    ttgl.BlockedLayout(size_per_thread=[1, 8], threads_per_warp=[4, 8], warps_per_cta=[2, 2], order=[1, 0],
                       cga_layout=[[1, 0]]),
    ttgl.BlockedLayout(size_per_thread=[1, 1], threads_per_warp=[4, 8], warps_per_cta=[4, 1], order=[1, 0],
                       cga_layout=[[1, 0], [2, 0], [0, 0], [0, 0]]),
    ttgl.BlockedLayout(size_per_thread=[1, 2], threads_per_warp=[4, 8], warps_per_cta=[1, 1], order=[1, 0],
                       cga_layout=[[0, 1], [0, 0], [1, 0], [0, 0]]),
    ttgl.BlockedLayout(size_per_thread=[1, 4], threads_per_warp=[4, 8], warps_per_cta=[2, 2], order=[1, 0],
                       cga_layout=[[1, 0], [2, 0], [0, 0], [0, 0]]),
    ttgl.BlockedLayout(size_per_thread=[1, 8], threads_per_warp=[4, 8], warps_per_cta=[1, 4], order=[1, 0],
                       cga_layout=[[0, 1], [0, 0], [1, 0], [0, 0]]),
    ttgl.BlockedLayout(size_per_thread=[1, 16], threads_per_warp=[4, 8], warps_per_cta=[2, 2], order=[1, 0],
                       cga_layout=[[0, 1], [0, 2], [0, 4], [0, 0]]),
])
@pytest.mark.parametrize("dtype", [
    # Test from 1 byte -> 8 bytes dtypes
    torch.float64, torch.float32, torch.float16, torch.float8_e4m3fn
])
def test_runtime_cluster_load(blocked_layout, dtype):
    M = 128
    N = 128
    BLOCK_M = 64
    BLOCK_N = 64
    num_ctas = 2**len(blocked_layout.cga_layout)

    if dtype == torch.float8_e4m3fn:
        # range from min normal (0 00001 00) to max normal (0 11110 11)
        a = torch.randint(0x04, 0x7B, (M, N), dtype=torch.uint8).view(dtype)
    else:
        a = torch.rand((M, N), dtype=dtype)
    out = torch.empty_like(a)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1)
    num_warps = blocked_layout.warps_per_cta[0] * blocked_layout.warps_per_cta[1]
    out_handle = out.cuda()
    cluster_load_and_write_back_kernel[grid](a.cuda(), out_handle, M, N, BLOCK_M, BLOCK_N, blocked_layout,
                                             num_warps=num_warps, num_ctas=num_ctas)
    out_tri = out_handle.cpu()
    out_ref = a.cpu()
    assert torch.equal(out_tri, out_ref)


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


def _test_runtime_async_copy_layouts(M, N, vec_size, shared_layout, dtype, use_mbarrier):
    BLOCK_M = 128
    BLOCK_N = 128

    if dtype == torch.float8_e4m3fn:
        # range from min normal (0 00001 00) to max normal (0 11110 11)
        a = torch.randint(0x04, 0x7B, (M, N), dtype=torch.uint8).view(dtype)
    else:
        a = torch.rand((M, N), dtype=dtype)
    out = torch.empty_like(a)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1)
    out_handle = out.cuda()

    if not use_mbarrier:
        blocked_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0])
        run_kernel = lambda: async_load_and_write_back_kernel[grid](a.cuda(), out_handle, M, N, BLOCK_M, BLOCK_N,
                                                                    blocked_layout, shared_layout)
    else:
        run_kernel = lambda: async_copy_mbarrier_kernel[grid](a.cuda(), out_handle, M, N, BLOCK_M, BLOCK_N,
                                                              shared_layout)

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


@pytest.mark.parametrize("blocked_layout", [
    ttgl.BlockedLayout(size_per_thread=[1, 8], threads_per_warp=[4, 8], warps_per_cta=[1, 1], order=[1, 0]),
    ttgl.BlockedLayout(size_per_thread=[1, 8], threads_per_warp=[4, 8], warps_per_cta=[1, 1], order=[1, 0],
                       cga_layout=[[0, 1]]),
    ttgl.BlockedLayout(size_per_thread=[1, 8], threads_per_warp=[4, 8], warps_per_cta=[1, 1], order=[1, 0],
                       cga_layout=[[1, 0]]),
    ttgl.BlockedLayout(size_per_thread=[1, 8], threads_per_warp=[4, 8], warps_per_cta=[1, 1], order=[1, 0],
                       cga_layout=[[0, 1], [0, 2], [0, 0], [0, 0]]),
    ttgl.BlockedLayout(size_per_thread=[1, 8], threads_per_warp=[4, 8], warps_per_cta=[1, 1], order=[1, 0],
                       cga_layout=[[0, 1], [0, 0], [1, 0], [0, 0]]),
    ttgl.BlockedLayout(size_per_thread=[1, 8], threads_per_warp=[4, 8], warps_per_cta=[1, 1], order=[1, 0],
                       cga_layout=[[0, 1], [0, 2], [0, 4], [0, 0]]),
])
def test_runtime_async_copy_layouts_multi_cta(blocked_layout):
    M = 1024
    N = 1024
    BLOCK_M = 128
    BLOCK_N = 128
    num_ctas = 2**len(blocked_layout.cga_layout)

    shared_layout = ttgl.SwizzledSharedLayout(1, 1, 1, [1, 0], blocked_layout.cga_layout)

    a = torch.rand((M, N), dtype=torch.float32)
    out = torch.empty_like(a)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1)
    out_handle = out.cuda()
    async_load_and_write_back_kernel[grid](a.cuda(), out_handle, M, N, BLOCK_M, BLOCK_N, blocked_layout, shared_layout,
                                           num_warps=1, num_ctas=num_ctas)
    out_tri = out_handle.cpu()
    out_ref = a.cpu()
    assert torch.equal(out_tri, out_ref)


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
    NON_K_PRESHUFFLE_BLOCK_SIZE: ttgl.constexpr = 64

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

    scale_a = scale_a.reshape(BLOCK_M // NON_K_PRESHUFFLE_BLOCK_SIZE, SCALE_BLOCK_K // SCALE_KWIDTH, 16, 4,
                              SCALE_KWIDTH).trans(0, 3, 2, 1, 4).reshape(BLOCK_M, SCALE_BLOCK_K)
    scale_b = scale_b.reshape(BLOCK_N // NON_K_PRESHUFFLE_BLOCK_SIZE, SCALE_BLOCK_K // SCALE_KWIDTH, 16, 4,
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
@pytest.mark.parametrize("M, N, K", [(64, 64, 64), (128, 128, 128), (256, 256, 256)])
@pytest.mark.parametrize("type_a", ["e5m2", "e2m1", "e4m3"])
@pytest.mark.parametrize("type_b", ["e5m2", "e2m1", "e4m3"])
@pytest.mark.parametrize("TRANSPOSED_WMMA", [True, False])
def test_runtime_wmma_scale_preshuffle(M, N, K, type_a, type_b, TRANSPOSED_WMMA):

    def pack_scale(x):
        PRESHUFFLE_FACTOR = 64
        NON_K, K_SCALE = x.shape
        num_chunk_m = NON_K // PRESHUFFLE_FACTOR
        SCALE_KWIDTH = 4 if K_SCALE >= 4 else K_SCALE
        num_chunk_k = K_SCALE // SCALE_KWIDTH

        x = x.view(num_chunk_m, 4, 16, num_chunk_k, SCALE_KWIDTH)
        x = x.permute(0, 3, 2, 1, 4).contiguous()
        return x.view(NON_K // PRESHUFFLE_FACTOR, K_SCALE * PRESHUFFLE_FACTOR)

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


@gluon.jit
def async_copy_mbarrier_kernel(a_ptr, out_ptr, M, N, BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr,
                               shared_layout: ttgl.constexpr):

    ASYNC_LOAD_BLOCKED_LAYOUT: ttgl.constexpr = ttgl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0])
    BLOCKED_LAYOUT: ttgl.constexpr = ttgl.BlockedLayout([1, 8], [4, 8], [2, 2], [1, 0])
    NUM_WARPS: ttgl.constexpr = 4
    WARP_SIZE: ttgl.constexpr = 32

    pid = ttgl.program_id(axis=0)
    num_pid_m = ttgl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    offs_m = pid_m * BLOCK_M + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, ASYNC_LOAD_BLOCKED_LAYOUT))
    offs_n = pid_n * BLOCK_N + ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, ASYNC_LOAD_BLOCKED_LAYOUT))

    out_offs_m = pid_m * BLOCK_M + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, BLOCKED_LAYOUT))
    out_offs_n = pid_n * BLOCK_N + ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, BLOCKED_LAYOUT))

    a_ptrs = a_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (out_offs_m[:, None] < M) & (out_offs_n[None, :] < N)

    mbar = ttgl.allocate_shared_memory(ttgl.int64, [1], ttgl.amd.gfx1250.mbarrier.MBarrierLayout())
    buffer = ttgl.allocate_shared_memory(a_ptr.type.element_ty, [BLOCK_M, BLOCK_N], shared_layout)
    # NOTE: Setting count = NUM_WARPS * WARP_SIZE * 2 is only for testing purposes, in order to also exercise the ttgl.amd.gfx1250.mbarrier.arrive API.
    # In practice, since we know that phase is initialized to 0, we can just set count = NUM_WARPS * WARP_SIZE and call directly ttgl.amd.gfx1250.mbarrier.wait(mbar, 0).
    ttgl.amd.gfx1250.mbarrier.init(mbar, count=NUM_WARPS * WARP_SIZE * 2)
    ttgl.amd.gfx1250.async_copy.global_to_shared(buffer, a_ptrs)
    ttgl.amd.gfx1250.async_copy.mbarrier_arrive(mbar)
    prior_phase = ttgl.amd.gfx1250.mbarrier.arrive(mbar)
    ttgl.amd.gfx1250.mbarrier.wait(mbar, prior_phase)

    res = buffer.load(BLOCKED_LAYOUT)

    out_ptrs = out_ptr + out_offs_m[:, None] * N + out_offs_n[None, :]
    ttgl.store(out_ptrs, res, mask)


@pytest.mark.parametrize("BLOCK_M,BLOCK_N", [(32, 32), (32, 64), (64, 64)])
def test_compile_async_copy_mbarrier(BLOCK_M, BLOCK_N):
    SHARED_LAYOUT = ttgl.SwizzledSharedLayout(8, 2, 4, [1, 0])
    signature = {
        "a_ptr": "*fp16", "out_ptr": "*fp16", "M": "i32", "N": "i32",  #
        "BLOCK_M": "constexpr", "BLOCK_N": "constexpr", "shared_layout": "constexpr"
    }
    constexprs = {"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "shared_layout": SHARED_LAYOUT}
    # AsyncCopy requires >= 32 bits per lane so we have to pass divisibility for arguments used in pointer arithmetic
    attrs = {k: [["tt.divisibility", 16]] for k in [(x, ) for x in range(4)]}
    k = triton.compile(
        gluon._runtime.GluonASTSource(fn=async_copy_mbarrier_kernel, signature=signature, attrs=attrs,
                                      constexprs=constexprs), target=GPUTarget("hip", 'gfx1250', 32))
    pattern = ("global_load_async_to_lds", "ds_atomic_async_barrier_arrive_b64", "ds_atomic_barrier_arrive_rtn_b64",
               "s_sleep")

    amdgcn = k.asm["amdgcn"]
    for pattern in pattern:
        assert re.search(pattern, amdgcn)

    assert not re.search("s_wait_asynccnt 0x0", amdgcn)


@ASYNC_COPY_TEST_PARAM_SIZE
@ASYNC_COPY_TEST_PARAM_SHARED_LAYOUT
@ASYNC_COPY_TEST_PARAM_DTYPE
def test_runtime_async_copy_mbarrier(M, N, vec_size, shared_layout, dtype):
    _test_runtime_async_copy_layouts(M, N, vec_size, shared_layout, dtype, True)


@gluon.jit
def tensor_async_copy_mbarrier_kernel(a_ptr, b_ptr, M, N,  #
                                      BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr, NUM_BUFFERS: ttgl.constexpr,
                                      BLOCKED_LAYOUT: ttgl.constexpr, NUM_WARPS: ttgl.constexpr):
    SHARED_LAYOUT: ttgl.constexpr = ttgl.PaddedSharedLayout.with_identity_for([[32, 4]], [BLOCK_M, BLOCK_N], [1, 0])
    WARP_SIZE: ttgl.constexpr = 32
    pid = ttgl.program_id(axis=0)
    num_pid_m = ttgl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    a_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(base=a_ptr, shape=(M, N), strides=(N, 1),
                                                         block_shape=(BLOCK_M, BLOCK_N), layout=SHARED_LAYOUT)
    bars = ttgl.allocate_shared_memory(ttgl.int64, [NUM_BUFFERS, 1], ttgl.amd.gfx1250.mbarrier.MBarrierLayout())
    a_buffer = ttgl.allocate_shared_memory(a_desc.dtype, [NUM_BUFFERS] + a_desc.block_shape, a_desc.layout)

    # NOTE: barrier count takes into account both warp count (NUM_WARPS which is used for TDM) + thread count (NUM_WARPS * WARP_SIZE which is used for mbarrier.arrive)
    # NOTE: Setting count = NUM_WARPS + NUM_WARPS * WARP_SIZE is only for testing purposes, in order to also exercise the ttgl.amd.gfx1250.mbarrier.arrive API.
    # In practice, since we know that phase is initialized to 0, we can just set count = NUM_WARPS and call directly ttgl.amd.gfx1250.mbarrier.wait(bars.index(i), 0).

    for i in ttgl.static_range(0, NUM_BUFFERS):
        ttgl.amd.gfx1250.mbarrier.init(bars.index(i), count=NUM_WARPS + NUM_WARPS * WARP_SIZE)

    idx_m = pid_m * BLOCK_M
    for i in ttgl.static_range(0, NUM_BUFFERS):
        idx_n = pid_n * (BLOCK_N * NUM_BUFFERS) + i * BLOCK_N
        ttgl.amd.gfx1250.tdm.async_load(a_desc, [idx_m, idx_n], a_buffer.index(i), mbarrier=bars.index(i))

    for i in ttgl.static_range(0, NUM_BUFFERS):
        prior_phase = ttgl.amd.gfx1250.mbarrier.arrive(bars.index(i))
        ttgl.amd.gfx1250.mbarrier.wait(bars.index(i), prior_phase)
        idx_n = pid_n * (BLOCK_N * NUM_BUFFERS) + i * BLOCK_N
        a = a_buffer.index(i).load(layout=BLOCKED_LAYOUT)

        offs_bm = idx_m + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, BLOCKED_LAYOUT))
        offs_bn = idx_n + ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, BLOCKED_LAYOUT))
        offs_b = (offs_bm[:, None] * N) + offs_bn[None, :]
        mask_b = (offs_bm[:, None] < M) & (offs_bn[None, :] < N)

        ttgl.store(b_ptr + offs_b, a, mask=mask_b)


@pytest.mark.parametrize("BLOCK_M,BLOCK_N", [(32, 32), (32, 64), (64, 64)])
@pytest.mark.parametrize("NUM_BUFFERS", [1, 2])
@pytest.mark.parametrize("NUM_WARPS", [4])
def test_compile_tensor_copy_mbarrier(BLOCK_M, BLOCK_N, NUM_BUFFERS, NUM_WARPS):
    BLOCKED_LAYOUT = ttgl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0])
    signature = {
        "a_ptr": "*fp16", "b_ptr": "*fp16", "M": "i32", "N": "i32",  #
        "BLOCK_M": "constexpr", "BLOCK_N": "constexpr", "NUM_BUFFERS": "constexpr", "BLOCKED_LAYOUT": "constexpr",
        "NUM_WARPS": "constexpr"
    }
    constexprs = {
        "BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "NUM_BUFFERS": NUM_BUFFERS, "BLOCKED_LAYOUT": BLOCKED_LAYOUT,
        "NUM_WARPS": NUM_WARPS
    }
    attrs = []
    k = triton.compile(
        gluon._runtime.GluonASTSource(fn=tensor_async_copy_mbarrier_kernel, signature=signature, attrs=attrs,
                                      constexprs=constexprs), target=GPUTarget("hip", 'gfx1250', 32))
    pattern = ("tensor_load_to_lds", "ds_atomic_barrier_arrive_rtn_b64", "s_sleep")

    amdgcn = k.asm["amdgcn"]
    for pattern in pattern:
        assert re.search(pattern, amdgcn)

    assert not re.search("s_wait_tensorcnt 0x0", amdgcn)


@pytest.mark.parametrize("BLOCK_M,BLOCK_N", [(32, 32), (32, 64), (64, 64), (1, 512), (256, 2)])
@pytest.mark.parametrize("NUM_BUFFERS", [1, 2])
@pytest.mark.parametrize("NUM_WARPS", [4, 8])
@pytest.mark.parametrize("M,N", [(1024, 1024), (1008, 1008), (1000, 1000)])
def test_runtime_tensor_copy_mbarrier(M, N, BLOCK_M, BLOCK_N, NUM_BUFFERS, NUM_WARPS):
    torch.manual_seed(42)
    a = torch.randint(0x0, 0xFFFF, (M, N), dtype=torch.uint16)
    b = torch.zeros_like(a)

    a_device = a.cuda()
    b_device = b.cuda()
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N * NUM_BUFFERS), 1)

    blocked_layout = ttgl.BlockedLayout([1, 8], [4, 8], [NUM_WARPS, 1], [1, 0])

    tensor_async_copy_mbarrier_kernel[grid](a_device, b_device, M, N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                                            NUM_BUFFERS=NUM_BUFFERS, BLOCKED_LAYOUT=blocked_layout, NUM_WARPS=NUM_WARPS,
                                            num_warps=NUM_WARPS)

    b_triton = b_device.cpu()
    assert torch.equal(b_triton, a)


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires GFX1250")
def test_tdm_load_pred():

    @gluon.jit
    def kernel(a_ptr, b_ptr):
        shared_layout: ttgl.constexpr = ttgl.PaddedSharedLayout.with_identity_for([[32, 4]], [16, 32], [1, 0])
        reg_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 4], [4, 8], [4, 1], [1, 0])

        desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(base=a_ptr, shape=(16, 64), strides=(64, 1),
                                                           block_shape=(16, 32), layout=shared_layout)
        smem = ttgl.allocate_shared_memory(desc.dtype, shape=desc.block_shape, layout=desc.layout)
        b_offs_m = ttgl.arange(0, 16, layout=ttgl.SliceLayout(1, reg_layout))
        b_offs_n = ttgl.arange(0, 32, layout=ttgl.SliceLayout(0, reg_layout))
        b_ptrs = b_ptr + b_offs_m[:, None] * 64 + b_offs_n[None, :]

        ttgl.amd.gfx1250.tdm.async_load(desc, [0, 0], smem, pred=False)
        ttgl.amd.gfx1250.tdm.async_wait(0)
        tile1 = smem.load(reg_layout)
        ttgl.store(b_ptrs, tile1)

        ttgl.amd.gfx1250.tdm.async_load(desc, [0, 32], smem, pred=True)
        ttgl.amd.gfx1250.tdm.async_wait(0)
        tile2 = smem.load(reg_layout)
        ttgl.store(b_ptrs + 32, tile2)

    a = torch.randint(0x0, 0xFFFF, (16, 64), dtype=torch.uint16)
    b = torch.zeros_like(a)

    a_device = a.cuda()
    b_device = b.cuda()
    kernel[(1, )](a_device, b_device)

    b = b_device.cpu()
    assert torch.equal(a[:, 32:], b[:, 32:]) and not torch.equal(a[:, :32], b[:, :32])


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires GFX1250")
@pytest.mark.parametrize("XBLOCK", [128])
def test_ws_store_wait_load(XBLOCK):
    """
    Tests warp specialization with mbarrier synchronization on GFX1250.

    This test validates the mbarrier wait/arrive mechanism for synchronizing data flow
    between two specialized warp groups using helper variables ready_bar and done_bar:
    - ws_producer (worker) partition: Stores data to shared memory and signals completion via ready_bar
    - ws_consumer (default) partition: Waits on ready_bar, loads the data, processes it, stores to
      a different shared memory location, and signals completion via done_bar

    The main kernel (executed by default warps) then waits for done_bar, loads the final result, and stores
    it to global memory. The test verifies data integrity by comparing the output with an expected
    arange pattern.
    """

    @gluon.jit
    def ws_consumer(smem, ready_bar, done_bar, layout: ttgl.constexpr):
        ttgl.amd.gfx1250.mbarrier.wait(ready_bar, phase=0)
        val = smem.index(0).load(layout)
        smem.index(1).store(val)
        ttgl.amd.gfx1250.mbarrier.arrive(done_bar, count=1)

    @gluon.jit
    def ws_producer(smem, ready_bar, XBLOCK: ttgl.constexpr, layout: ttgl.constexpr):
        smem.index(0).store(ttgl.arange(0, XBLOCK, layout).to(ttgl.float16))
        ttgl.amd.gfx1250.mbarrier.arrive(ready_bar, count=1)

    @gluon.jit
    def ws_kernel(output, XBLOCK: ttgl.constexpr):
        WARP_SIZE: ttgl.constexpr = 32
        smem_layout: ttgl.constexpr = ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0])
        blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32],
                                                            warps_per_cta=[4], order=[0])
        smem = ttgl.allocate_shared_memory(ttgl.float16, [2, XBLOCK], smem_layout)
        bar = ttgl.allocate_shared_memory(ttgl.int64, [2, 1], ttgl.amd.gfx1250.mbarrier.MBarrierLayout())
        for i in range(2):
            # we have 4 default warps and 4 worker warps and arrive on barrier once per thread
            ttgl.amd.gfx1250.mbarrier.init(bar.index(i), count=4 * WARP_SIZE)
        ready_bar = bar.index(0)
        done_bar = bar.index(1)
        # NOTE: We have 8 warps in total. worker_num_warps = [4] (num warps for ws_producer partition) and num_warps = 4 (num warps for consumer partition)
        ttgl.warp_specialize([
            (ws_consumer, (smem, ready_bar, done_bar, blocked_layout)),
            (ws_producer, (smem, ready_bar, XBLOCK, blocked_layout)),
        ], [4])
        ttgl.amd.gfx1250.mbarrier.wait(done_bar, phase=0)
        val = smem.index(1).load(blocked_layout)
        output_ptrs = output + ttgl.arange(0, XBLOCK, blocked_layout)
        ttgl.store(output_ptrs, val)

    output = torch.empty((XBLOCK, ), dtype=torch.float16).cuda()
    ws_kernel[(1, )](output, XBLOCK=XBLOCK, num_warps=4)
    torch_output = torch.arange(0, XBLOCK, dtype=torch.float16)
    output_ref = output.cpu()
    assert torch.equal(output_ref, torch_output)


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires GFX1250")
@pytest.mark.parametrize("XBLOCK", [128])
@pytest.mark.parametrize("NUM_ITERS", [10])
def test_ws_store_wait_load_loop(XBLOCK, NUM_ITERS):
    """
    Tests warp specialization with mbarrier synchronization in a loop and phase tracking on GFX1250.

    This test validates iterative producer-consumer synchronization using three mbarriers:
    - ready_bar: Signals that the producer has written data to shared memory
    - done_bar: Signals that the consumer has finished all iterations
    - empty_bar: Signals that the consumer has consumed data and buffer is empty

    - ws_producer (worker) partition: Waits for empty_bar, writes data, signals via ready_bar (loops NUM_ITERS times)
    - ws_consumer (default) partition: Waits for ready_bar, reads and accumulates data, signals via empty_bar (loops NUM_ITERS times)

    Both partitions track phases (1-bit parity phase which toggles between 0 for even and 1 for odd). After all iterations, the main kernel
    (executed by default warps) waits for done_bar, loads the accumulated result, and stores it to global memory.
    The test verifies that the output equals the expected arange pattern.
    """

    @gluon.jit
    def ws_consumer(smem, ready_bar, done_bar, empty_bar, XBLOCK: ttgl.constexpr, NUM_ITERS: ttgl.constexpr,
                    layout: ttgl.constexpr):
        acc = ttgl.zeros([XBLOCK], ttgl.float16, layout)
        phase = 0
        for _ in ttgl.static_range(NUM_ITERS):
            ttgl.amd.gfx1250.mbarrier.wait(ready_bar, phase=phase)
            phase = phase ^ 1
            val = smem.index(0).load(layout)
            acc += val
            ttgl.amd.gfx1250.mbarrier.arrive(empty_bar, count=1)

        smem.index(1).store(acc)
        ttgl.amd.gfx1250.mbarrier.arrive(done_bar, count=1)

    @gluon.jit
    def ws_producer(smem, ready_bar, empty_bar, XBLOCK: ttgl.constexpr, NUM_ITERS: ttgl.constexpr,
                    layout: ttgl.constexpr):
        val = ttgl.arange(0, XBLOCK, layout).to(ttgl.float16)
        phase = 0
        for _ in ttgl.static_range(NUM_ITERS):
            ttgl.amd.gfx1250.mbarrier.wait(empty_bar, phase=phase)
            phase = phase ^ 1
            smem.index(0).store(val)
            ttgl.amd.gfx1250.mbarrier.arrive(ready_bar, count=1)

    @gluon.jit
    def ws_kernel(output, XBLOCK: ttgl.constexpr, NUM_ITERS: ttgl.constexpr):
        WARP_SIZE: ttgl.constexpr = 32
        smem_layout: ttgl.constexpr = ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0])
        blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32],
                                                            warps_per_cta=[4], order=[0])
        smem = ttgl.allocate_shared_memory(ttgl.float16, [2, XBLOCK], smem_layout)
        bar = ttgl.allocate_shared_memory(ttgl.int64, [3, 1], ttgl.amd.gfx1250.mbarrier.MBarrierLayout())
        for i in ttgl.static_range(3):
            # we have 4 default warps and 4 worker warps and arrive on barrier once per thread
            ttgl.amd.gfx1250.mbarrier.init(bar.index(i), count=4 * WARP_SIZE)
        ready_bar = bar.index(0)
        done_bar = bar.index(1)
        empty_bar = bar.index(2)

        ttgl.amd.gfx1250.mbarrier.arrive(empty_bar, count=1)
        # NOTE: We have 8 warps in total. worker_num_warps = [4] (num warps for ws_producer partition) and num_warps = 4 (num warps for consumer partition)
        ttgl.warp_specialize([
            (ws_consumer, (smem, ready_bar, done_bar, empty_bar, XBLOCK, NUM_ITERS, blocked_layout)),
            (ws_producer, (smem, ready_bar, empty_bar, XBLOCK, NUM_ITERS, blocked_layout)),
        ], [4])
        ttgl.amd.gfx1250.mbarrier.wait(done_bar, phase=0)
        val = smem.index(1).load(blocked_layout)
        output_ptrs = output + ttgl.arange(0, XBLOCK, blocked_layout)
        ttgl.store(output_ptrs, val)

    output = torch.empty((XBLOCK, ), dtype=torch.float16).cuda()
    ws_kernel[(1, )](output, XBLOCK=XBLOCK, NUM_ITERS=NUM_ITERS, num_warps=4)
    torch_output = NUM_ITERS * torch.arange(0, XBLOCK, dtype=torch.float16)
    output_ref = output.cpu()
    assert torch.equal(output_ref, torch_output)


@pytest.mark.parametrize("BLOCK_M,BLOCK_N", [(32, 32), (32, 64), (64, 64), (1, 512), (256, 2)])
@pytest.mark.parametrize("NUM_BUFFERS", [1, 2])
@pytest.mark.parametrize("NUM_TOTAL_WARPS", [8, 16])
@pytest.mark.parametrize("M,N", [(32, 32), (1024, 1024), (1008, 1008), (1000, 1000)])
def test_runtime_ws_tensor_async_load_store_mbarrier(M, N, BLOCK_M, BLOCK_N, NUM_BUFFERS, NUM_TOTAL_WARPS):
    """
    Tests warp specialization with tensor descriptor async load/store operations coordinated by mbarriers on GFX1250.

    This test validates the producer-consumer pattern using TDM async operations
    with multiple buffers, where each buffer has its own dedicated mbarrier for synchronization:
    - ws_producer (worker) partition: Asynchronously loads data from global memory to shared memory buffers
      using TDM async_load, with each load operation automatically signaling its corresponding mbarrier
    - ws_consumer (default) partition: Waits on each buffer's mbarrier, then asynchronously stores data
      from shared memory to global memory using TDM async_store

    The synchronization pattern uses one mbarrier per buffer (bars.index(i)), ensuring that the consumer
    only accesses a buffer after the producer has completed loading into it.

    The test verifies that the output matches the input, confirming that async load/store operations are correctly coordinated by mbarriers.
    """

    @gluon.jit
    def ws_producer(a_desc, a_buffer, bars, pid_n, idx_m, BLOCK_N: ttgl.constexpr, NUM_BUFFERS: ttgl.constexpr):
        for i in ttgl.static_range(0, NUM_BUFFERS):
            idx_n = pid_n * (BLOCK_N * NUM_BUFFERS) + i * BLOCK_N
            ttgl.amd.gfx1250.tdm.async_load(a_desc, [idx_m, idx_n], a_buffer.index(i), mbarrier=bars.index(i))

    @gluon.jit
    def ws_consumer(b_desc, a_buffer, bars, pid_n, idx_m, BLOCK_N: ttgl.constexpr, NUM_BUFFERS: ttgl.constexpr):
        for i in ttgl.static_range(0, NUM_BUFFERS):
            ttgl.amd.gfx1250.mbarrier.wait(bars.index(i), 0)
            idx_n = pid_n * (BLOCK_N * NUM_BUFFERS) + i * BLOCK_N
            ttgl.amd.gfx1250.tdm.async_store(b_desc, [idx_m, idx_n], a_buffer.index(i))

        ttgl.amd.gfx1250.tdm.async_wait(0)

    @gluon.jit
    def ws_tensor_async_load_store_mbarrier_kernel(a_ptr, b_ptr, M, N,  #
                                                   BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr,
                                                   NUM_BUFFERS: ttgl.constexpr, NUM_WARPS: ttgl.constexpr):
        SHARED_LAYOUT: ttgl.constexpr = ttgl.SwizzledSharedLayout(1, 1, 1, [1, 0])
        PRODUCER_WARPS: ttgl.constexpr = NUM_WARPS // 2
        pid = ttgl.program_id(axis=0)
        num_pid_m = ttgl.cdiv(M, BLOCK_M)
        pid_m = pid % num_pid_m
        pid_n = pid // num_pid_m

        a_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(base=a_ptr, shape=(M, N), strides=(N, 1),
                                                             block_shape=(BLOCK_M, BLOCK_N), layout=SHARED_LAYOUT)
        b_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(base=b_ptr, shape=(M, N), strides=(N, 1),
                                                             block_shape=(BLOCK_M, BLOCK_N), layout=SHARED_LAYOUT)
        bars = ttgl.allocate_shared_memory(ttgl.int64, [NUM_BUFFERS, 1], ttgl.amd.gfx1250.mbarrier.MBarrierLayout())
        a_buffer = ttgl.allocate_shared_memory(a_desc.dtype, [NUM_BUFFERS] + a_desc.block_shape, a_desc.layout)

        for i in ttgl.static_range(0, NUM_BUFFERS):
            ttgl.amd.gfx1250.mbarrier.init(bars.index(i), count=PRODUCER_WARPS)

        idx_m = pid_m * BLOCK_M

        ttgl.warp_specialize([
            (ws_consumer, (b_desc, a_buffer, bars, pid_n, idx_m, BLOCK_N, NUM_BUFFERS)),
            (ws_producer, (a_desc, a_buffer, bars, pid_n, idx_m, BLOCK_N, NUM_BUFFERS)),
        ], [PRODUCER_WARPS])

    torch.manual_seed(42)
    a = torch.randint(0x0, 0xFFFF, (M, N), dtype=torch.uint16)
    b = torch.zeros_like(a)

    a_device = a.cuda()
    b_device = b.cuda()
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N * NUM_BUFFERS), 1)

    ws_tensor_async_load_store_mbarrier_kernel[grid](a_device, b_device, M, N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                                                     NUM_BUFFERS=NUM_BUFFERS, NUM_WARPS=NUM_TOTAL_WARPS,
                                                     num_warps=NUM_TOTAL_WARPS // 2)

    b_triton = b_device.cpu()
    assert torch.equal(b_triton, a)


@pytest.mark.parametrize("BLOCK_M,BLOCK_N", [(32, 32), (32, 64), (64, 64), (1, 512), (256, 2)])
@pytest.mark.parametrize("NUM_BUFFERS", [1, 2])
@pytest.mark.parametrize("NUM_TOTAL_WARPS", [8, 16])
@pytest.mark.parametrize("M,N", [(32, 32), (1024, 1024), (1008, 1008), (1000, 1000)])
def test_runtime_ws_tensor_copy_mbarrier(M, N, BLOCK_M, BLOCK_N, NUM_BUFFERS, NUM_TOTAL_WARPS):
    """
    Tests warp specialization with mixed async/sync operations coordinated by mbarriers on GFX1250.

    This test validates the producer-consumer pattern using a combination of TDM async loads and
    synchronous stores with multiple buffers, where each buffer has its own dedicated mbarrier:
    - ws_producer (worker) partition: Asynchronously loads data from global memory to shared memory buffers
      using TDM async_load, with each load operation automatically signaling its corresponding mbarrier
    - ws_consumer (default) partition: Waits on each buffer's mbarrier, loads data from shared memory
      into registers using regular loads, then stores to global memory using regular synchronous stores

    The synchronization pattern uses one mbarrier per buffer (bars.index(i)), ensuring that the consumer
    only accesses a buffer after the producer has completed loading into it.

    NOTE: This test showcases that tensors (here: b_ptr) can be passed as arguments to the default partition
    (here: ws_consumer), which is not supported for worker partitions.

    The test verifies that the output matches the input, confirming correct synchronization.
    """

    @gluon.jit
    def ws_producer(a_desc, a_buffer, bars, pid_n, idx_m, BLOCK_N: ttgl.constexpr, NUM_BUFFERS: ttgl.constexpr):
        for i in ttgl.static_range(0, NUM_BUFFERS):
            idx_n = pid_n * (BLOCK_N * NUM_BUFFERS) + i * BLOCK_N
            ttgl.amd.gfx1250.tdm.async_load(a_desc, [idx_m, idx_n], a_buffer.index(i), mbarrier=bars.index(i))

    @gluon.jit
    def ws_consumer(a_buffer, b_ptr, bars, pid_n, idx_m, M, N, BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr,
                    NUM_BUFFERS: ttgl.constexpr, BLOCKED_LAYOUT: ttgl.constexpr):
        for i in ttgl.static_range(0, NUM_BUFFERS):
            ttgl.amd.gfx1250.mbarrier.wait(bars.index(i), 0)
            idx_n = pid_n * (BLOCK_N * NUM_BUFFERS) + i * BLOCK_N
            a = a_buffer.index(i).load(layout=BLOCKED_LAYOUT)
            offs_bm = idx_m + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, BLOCKED_LAYOUT))
            offs_bn = idx_n + ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, BLOCKED_LAYOUT))
            offs_b = (offs_bm[:, None] * N) + offs_bn[None, :]
            mask_b = (offs_bm[:, None] < M) & (offs_bn[None, :] < N)
            ttgl.store(b_ptr + offs_b, a, mask=mask_b)

    @gluon.jit
    def ws_tensor_async_copy_mbarrier_kernel(a_ptr, b_ptr, M, N,  #
                                             BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr,
                                             NUM_BUFFERS: ttgl.constexpr, BLOCKED_LAYOUT: ttgl.constexpr,
                                             NUM_WARPS: ttgl.constexpr):
        SHARED_LAYOUT: ttgl.constexpr = ttgl.PaddedSharedLayout.with_identity_for([[32, 4]], [BLOCK_M, BLOCK_N], [1, 0])
        PRODUCER_WARPS: ttgl.constexpr = NUM_WARPS // 2
        pid = ttgl.program_id(axis=0)
        num_pid_m = ttgl.cdiv(M, BLOCK_M)
        pid_m = pid % num_pid_m
        pid_n = pid // num_pid_m

        a_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(base=a_ptr, shape=(M, N), strides=(N, 1),
                                                             block_shape=(BLOCK_M, BLOCK_N), layout=SHARED_LAYOUT)
        bars = ttgl.allocate_shared_memory(ttgl.int64, [NUM_BUFFERS, 1], ttgl.amd.gfx1250.mbarrier.MBarrierLayout())
        a_buffer = ttgl.allocate_shared_memory(a_desc.dtype, [NUM_BUFFERS] + a_desc.block_shape, a_desc.layout)

        for i in ttgl.static_range(0, NUM_BUFFERS):
            # TDM arrives on barrier once per warp, so use producer warp count
            ttgl.amd.gfx1250.mbarrier.init(bars.index(i), count=PRODUCER_WARPS)

        idx_m = pid_m * BLOCK_M

        ttgl.warp_specialize([
            (ws_consumer, (a_buffer, b_ptr, bars, pid_n, idx_m, M, N, BLOCK_M, BLOCK_N, NUM_BUFFERS, BLOCKED_LAYOUT)),
            (ws_producer, (a_desc, a_buffer, bars, pid_n, idx_m, BLOCK_N, NUM_BUFFERS)),
        ], [PRODUCER_WARPS])

    torch.manual_seed(42)
    a = torch.randint(0x0, 0xFFFF, (M, N), dtype=torch.uint16)
    b = torch.zeros_like(a)

    a_device = a.cuda()
    b_device = b.cuda()
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N * NUM_BUFFERS), 1)

    blocked_layout = ttgl.BlockedLayout([1, 8], [4, 8], [NUM_TOTAL_WARPS // 2, 1], [1, 0])

    ws_tensor_async_copy_mbarrier_kernel[grid](a_device, b_device, M, N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                                               NUM_BUFFERS=NUM_BUFFERS, BLOCKED_LAYOUT=blocked_layout,
                                               NUM_WARPS=NUM_TOTAL_WARPS, num_warps=NUM_TOTAL_WARPS // 2)

    b_triton = b_device.cpu()
    assert torch.equal(b_triton, a)


@pytest.mark.parametrize("M,N", [(128, 128), (1024, 1024)])
@pytest.mark.parametrize("shared_layout", [
    ttgl.SwizzledSharedLayout(1, 1, 1, [1, 0]),
    ttgl.SwizzledSharedLayout(16, 2, 4, [1, 0]),
    ttgl.PaddedSharedLayout.with_identity_for([[4, 1]], [128, 128], [1, 0]),
])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.float8_e4m3fn])
@pytest.mark.parametrize("NUM_TOTAL_WARPS", [8])
def test_runtime_ws_async_copy_mbarrier(M, N, shared_layout, dtype, NUM_TOTAL_WARPS):
    """
    Tests warp specialization with async_copy operations and mbarrier synchronization on GFX1250.

    This test validates the producer-consumer pattern using async_copy with two mbarriers:
    - ready_bar: Signals that ws_producer has completed copying data to the input buffer
    - done_bar: Signals that ws_consumer has completed processing and writing to the output buffer

    - ws_producer (default) partition: Copies data from global memory to shared memory
      then signals completion via mbarrier_arrive on ready_bar.
    - ws_consumer (worker) partition: Waits on ready_bar, loads data from the input shared memory buffer,
      stores it to an output shared memory buffer, then signals done_bar.

    The main kernel (executed by default warps) waits on done_bar, then loads data
    from the output buffer and stores it to global memory.

    NOTE: This test showcases that tensors (here: a_ptrs) can be passed as arguments to
    the default partition (here: ws_producer), which is not supported for worker partitions.

    The test verifies that the output matches the input, confirming correct synchronization.
    """

    @gluon.jit
    def ws_producer(a_ptrs, buffer, ready_bar):
        ttgl.amd.gfx1250.async_copy.global_to_shared(buffer, a_ptrs)
        ttgl.amd.gfx1250.async_copy.mbarrier_arrive(ready_bar)

    @gluon.jit
    def ws_consumer(in_buffer, out_buffer, ready_bar, done_bar, BLOCKED_LAYOUT: ttgl.constexpr):
        ttgl.amd.gfx1250.mbarrier.wait(ready_bar, 0)
        val = in_buffer.load(BLOCKED_LAYOUT)
        out_buffer.store(val)
        ttgl.amd.gfx1250.mbarrier.arrive(done_bar, count=1)

    @gluon.jit
    def ws_async_copy_mbarrier_kernel(a_ptr, out_ptr, M, N, BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr,
                                      shared_layout: ttgl.constexpr, NUM_WARPS: ttgl.constexpr):
        PARTITION_WARPS: ttgl.constexpr = NUM_WARPS // 2
        ASYNC_LOAD_BLOCKED_LAYOUT: ttgl.constexpr = ttgl.BlockedLayout([1, 8], [4, 8], [PARTITION_WARPS, 1], [1, 0])
        BLOCKED_LAYOUT: ttgl.constexpr = ttgl.BlockedLayout(
            [1, 8], [4, 8], [PARTITION_WARPS // 2, PARTITION_WARPS // (PARTITION_WARPS // 2)], [1, 0])
        WARP_SIZE: ttgl.constexpr = 32

        pid = ttgl.program_id(axis=0)
        num_pid_m = ttgl.cdiv(M, BLOCK_M)
        pid_m = pid % num_pid_m
        pid_n = pid // num_pid_m

        offs_m = pid_m * BLOCK_M + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, ASYNC_LOAD_BLOCKED_LAYOUT))
        offs_n = pid_n * BLOCK_N + ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, ASYNC_LOAD_BLOCKED_LAYOUT))

        a_ptrs = a_ptr + offs_m[:, None] * N + offs_n[None, :]

        mbar = ttgl.allocate_shared_memory(ttgl.int64, [2, 1], ttgl.amd.gfx1250.mbarrier.MBarrierLayout())
        buffer = ttgl.allocate_shared_memory(a_ptr.type.element_ty, [BLOCK_M, BLOCK_N], shared_layout)
        out_buffer = ttgl.allocate_shared_memory(out_ptr.type.element_ty, [BLOCK_M, BLOCK_N], shared_layout)

        ready_bar = mbar.index(0)
        done_bar = mbar.index(1)

        # TDM arrives on barrier once per warp, so use partition warp count
        ttgl.amd.gfx1250.mbarrier.init(ready_bar, count=PARTITION_WARPS * WARP_SIZE)
        ttgl.amd.gfx1250.mbarrier.init(done_bar, count=PARTITION_WARPS * WARP_SIZE)

        ttgl.warp_specialize([
            (ws_producer, (a_ptrs, buffer, ready_bar)),
            (ws_consumer, (buffer, out_buffer, ready_bar, done_bar, BLOCKED_LAYOUT)),
        ], [PARTITION_WARPS])

        out_offs_m = pid_m * BLOCK_M + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, BLOCKED_LAYOUT))
        out_offs_n = pid_n * BLOCK_N + ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, BLOCKED_LAYOUT))
        mask = (out_offs_m[:, None] < M) & (out_offs_n[None, :] < N)
        ttgl.amd.gfx1250.mbarrier.wait(done_bar, 0)
        res = out_buffer.load(BLOCKED_LAYOUT)
        out_ptrs = out_ptr + out_offs_m[:, None] * N + out_offs_n[None, :]
        ttgl.store(out_ptrs, res, mask)

    BLOCK_M = 128
    BLOCK_N = 128

    if dtype == torch.float8_e4m3fn:
        # range from min normal (0 00001 00) to max normal (0 11110 11)
        a = torch.randint(0x04, 0x7B, (M, N), dtype=torch.uint8).view(dtype)
    else:
        a = torch.rand((M, N), dtype=dtype)
    out = torch.empty_like(a)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1)
    out_handle = out.cuda()

    ws_async_copy_mbarrier_kernel[grid](a.cuda(), out_handle, M, N, BLOCK_M, BLOCK_N, shared_layout,
                                        NUM_WARPS=NUM_TOTAL_WARPS, num_warps=NUM_TOTAL_WARPS // 2)
    out_tri = out_handle.cpu()
    out_ref = a.cpu()
    assert torch.equal(out_tri, out_ref)
