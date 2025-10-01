import torch
import pytest
import re
from itertools import product

import triton
import triton.language as tl

from triton._internal_testing import (
    is_ampere_or_newer,
    is_blackwell,
    is_hip_gfx11,
    is_hip_gfx12,
    is_hip_cdna3,
    is_hip_cdna4,
    is_hopper_or_newer,
    is_hopper,
)
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl
from triton.experimental.gluon.language.nvidia.ampere import async_copy, mma_v2
from triton.experimental.gluon.language.nvidia.hopper import tma, mbarrier, fence_async_shared
from triton.experimental.gluon.language.nvidia import hopper
from triton.experimental.gluon.language.amd.cdna4 import async_copy as cdna4_async_copy
from triton.experimental.gluon.language.extra import libdevice
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    TensorMemoryScalesLayout,
    allocate_tensor_memory,
    get_tmem_32x32b_reg_layout,
    tcgen05_mma,
    tcgen05_commit,
    tcgen05_copy,
    float2,
)

THREADS_PER_WARP = triton.runtime.driver.active.get_current_target().warp_size


@gluon.jit
def copy_kernel(Out, In, numel, XBLOCK: ttgl.constexpr, layout: ttgl.constexpr):
    xbase = ttgl.program_id(0) * XBLOCK
    xoffset = xbase + ttgl.arange(0, XBLOCK, layout=layout)
    xmask = xoffset < numel
    data = ttgl.load(In + xoffset, xmask)
    ttgl.store(Out + xoffset, data, xmask)


@pytest.mark.parametrize("layout", [
    ttgl.BlockedLayout(size_per_thread=[1], threads_per_warp=[THREADS_PER_WARP], warps_per_cta=[4], order=[0]),
    ttgl.BlockedLayout(size_per_thread=[2], threads_per_warp=[THREADS_PER_WARP], warps_per_cta=[4], order=[0]),
    ttgl.BlockedLayout(size_per_thread=[4], threads_per_warp=[THREADS_PER_WARP], warps_per_cta=[4], order=[0]),
    ttgl.BlockedLayout(size_per_thread=[8], threads_per_warp=[THREADS_PER_WARP], warps_per_cta=[4], order=[0]),
    ttgl.BlockedLayout(size_per_thread=[1], threads_per_warp=[THREADS_PER_WARP], warps_per_cta=[8], order=[0]),
    ttgl.BlockedLayout(size_per_thread=[2], threads_per_warp=[THREADS_PER_WARP], warps_per_cta=[8], order=[0]),
    ttgl.BlockedLayout(size_per_thread=[4], threads_per_warp=[THREADS_PER_WARP], warps_per_cta=[8], order=[0]),
    ttgl.BlockedLayout(size_per_thread=[8], threads_per_warp=[THREADS_PER_WARP], warps_per_cta=[8], order=[0]),
])
@pytest.mark.parametrize("XBLOCK", [128, 256, 512, 1024, 2048])
def test_copy_kernel(layout, XBLOCK):
    inp = torch.randn(XBLOCK * 4 - 7, device="cuda")
    out = torch.empty_like(inp)

    copy_kernel[(4, )](out, inp, inp.numel(), XBLOCK, layout, num_warps=layout.warps_per_cta[0])
    torch.testing.assert_close(out, inp)


@gluon.jit
def tma_kernel(desc):
    layout: ttgl.constexpr = ttgl.BlockedLayout([1, 2], [4, 8], [4, 1], [1, 0])
    value = ttgl.full(desc.block_shape, 0, desc.dtype, layout)
    alloc = ttgl.allocate_shared_memory(desc.dtype, desc.block_shape, desc.layout, value)
    tma.async_copy_shared_to_global(desc, [0, 0], alloc)
    tma.store_wait(0)
    alloc._keep_alive()


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Requires Hopper")
def test_tma():
    out = torch.ones((16, 16), dtype=torch.float16, device="cuda")
    layout = ttgl.NVMMASharedLayout(
        swizzle_byte_width=32,
        element_bitwidth=16,
        rank=2,
        transposed=False,
        fp4_padded=False,
    )

    desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(out, [16, 16], layout)
    tma_kernel[(1, )](desc)
    torch.testing.assert_close(out, torch.zeros_like(out))


@gluon.jit
def async_copy_mbarrier_kernel(out, inp, xnumel, XBLOCK: ttgl.constexpr, YBLOCK: ttgl.constexpr):
    smem = ttgl.allocate_shared_memory(inp.dtype.element_ty, [XBLOCK, YBLOCK],
                                       ttgl.SwizzledSharedLayout(1, 1, 1, order=[1, 0]))
    block_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 4], [1, 32], [4, 1], [1, 0])
    xindex = ttgl.arange(0, XBLOCK, ttgl.SliceLayout(1, block_layout))[:, None]
    yindex = ttgl.arange(0, YBLOCK, ttgl.SliceLayout(0, block_layout))[None, :]
    mask = xindex < xnumel
    async_copy.async_copy_global_to_shared(
        smem,
        inp + xindex * YBLOCK + yindex,
        mask,
    )
    mbar = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(mbar, count=1)
    async_copy.mbarrier_arrive(mbar)
    mbarrier.arrive(mbar)
    mbarrier.wait(mbar, 0)

    val = smem.load(block_layout)
    ttgl.store(out + xindex * YBLOCK + yindex, val)


@pytest.mark.skipif(not is_ampere_or_newer(), reason="Requires Ampere")
def test_async_copy_mbarrier():
    tensor_opts = dict(dtype=torch.float, device="cuda")
    out = torch.empty((32, 32), **tensor_opts)
    inp = torch.randn((20, 32), **tensor_opts)
    async_copy_mbarrier_kernel[(1, )](out, inp, inp.shape[0], XBLOCK=32, YBLOCK=32)
    torch.testing.assert_close(out[:20], inp)
    torch.testing.assert_close(out[20:], torch.zeros((12, 32), **tensor_opts))


@gluon.jit
def warpgroup_mma_kernel(a, b, out, M: ttgl.constexpr, N: ttgl.constexpr, K: ttgl.constexpr,
                         block_layout: ttgl.constexpr, mma_layout: ttgl.constexpr, shared_layout_a: ttgl.constexpr,
                         shared_layout_b: ttgl.constexpr, acc_dtype: ttgl.constexpr, ASYNC: ttgl.constexpr):
    a_offs_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, block_layout))[:, None]
    a_offs_k = ttgl.arange(0, K, layout=ttgl.SliceLayout(0, block_layout))[None, :]
    b_offs_k = ttgl.arange(0, K, layout=ttgl.SliceLayout(1, block_layout))[:, None]
    b_offs_n = ttgl.arange(0, N, layout=ttgl.SliceLayout(0, block_layout))[None, :]

    out_offs_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, mma_layout))[:, None]
    out_offs_n = ttgl.arange(0, N, layout=ttgl.SliceLayout(0, mma_layout))[None, :]

    operand_dtype = a.dtype.element_ty
    a_tile = ttgl.load(a + a_offs_m * K + a_offs_k)
    b_tile = ttgl.load(b + b_offs_k * N + b_offs_n)

    smem_a = ttgl.allocate_shared_memory(operand_dtype, [M, K], shared_layout_a, a_tile)
    smem_b = ttgl.allocate_shared_memory(operand_dtype, [K, N], shared_layout_b, b_tile)

    fence_async_shared()

    acc = ttgl.zeros([M, N], dtype=acc_dtype, layout=mma_layout)
    acc = hopper.warpgroup_mma(smem_a, smem_b, acc, is_async=ASYNC)

    if ASYNC:
        acc = hopper.warpgroup_mma_wait(num_outstanding=0, deps=[acc])

    ttgl.store(out + out_offs_m * N + out_offs_n, acc)


@pytest.mark.skipif(not is_hopper(), reason="Requires Hopper")
@pytest.mark.parametrize("ASYNC", [True, False])
def test_warpgroup_mma(ASYNC):
    torch.manual_seed(0)
    M, N, K = 64, 32, 32
    warps = [4, 1]
    block_layout = ttgl.BlockedLayout([1, 1], [1, THREADS_PER_WARP], warps_per_cta=warps, order=[1, 0])
    mma_layout = ttgl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=warps, instr_shape=[16, 32, 16])
    shared_layout_a = ttgl.NVMMASharedLayout.get_default_for([M, K], ttgl.float16)
    shared_layout_b = ttgl.NVMMASharedLayout.get_default_for([K, N], ttgl.float16)
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    out = torch.zeros((M, N), device="cuda", dtype=torch.float16)
    warpgroup_mma_kernel[(1, )](
        a,
        b,
        out,
        M,
        N,
        K,
        block_layout,
        mma_layout,
        shared_layout_a,
        shared_layout_b,
        ttgl.float16,
        ASYNC,
        num_warps=warps[0] * warps[1],
    )

    ref = torch.matmul(a, b)

    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-1)


@pytest.mark.skipif(not is_hopper(), reason="Requires Hopper")
@pytest.mark.parametrize("bitwidth, transpose_a, transpose_b, acc_dtype",
                         [(bitwidth, transpose_a, transpose_b, acc_dtype)
                          for bitwidth in [8, 16, 32]
                          for (transpose_a, transpose_b) in product([False, True], repeat=2)
                          for acc_dtype in [torch.float16, torch.float32]
                          if bitwidth == 16 or (acc_dtype == torch.float32 and not transpose_a and transpose_b)])
@pytest.mark.parametrize("warps", ([8, 1], [4, 2], [4, 1]))
# Swizzling 0 does not map to a valid memory descriptor lol
@pytest.mark.parametrize("swizzling_a, swizzling_b", product([32, 64, 128], repeat=2))
@pytest.mark.parametrize("shape_m, shape_n, shape_k", [(1, 1, 1), (2, 4, 1), (2, 2, 4)])
def test_warpgroup_mma_shared_inputs(bitwidth, transpose_a, transpose_b, acc_dtype, warps, swizzling_a, swizzling_b,
                                     shape_m, shape_n, shape_k):

    torch_dtype_map = {
        8: torch.float8_e4m3fn,
        16: torch.float16,
        32: torch.float32,
    }
    acc_dtype_map = {
        torch.float16: ttgl.float16,
        torch.float32: ttgl.float32,
    }

    # We'll choose a larger instr shape along N, but sure
    instr_shape_k_map = {8: 32, 16: 16, 32: 8}
    instr_shape = [16, 32, instr_shape_k_map[bitwidth]]
    M = instr_shape[0] * warps[0]
    N = instr_shape[1] * warps[1]
    K = instr_shape[2]

    def min_shape(swizzling, dim0, dim1, trans):
        tile_cols = (8 * max(16, swizzling)) // bitwidth
        outer_dim, contig_dim = (dim0, dim1)
        if trans:
            outer_dim, contig_dim = contig_dim, outer_dim
        contig_dim = max(contig_dim, tile_cols)
        outer_dim = max(outer_dim, 8)
        if trans:
            outer_dim, contig_dim = contig_dim, outer_dim
        return outer_dim, contig_dim

    # Get the minimum shape for the given swizzling / transpose
    M, K = min_shape(swizzling_a, M, K, transpose_a)
    K, N = min_shape(swizzling_b, K, N, transpose_b)
    M *= shape_m
    N *= shape_n
    K *= shape_k
    instr_shape[1] *= shape_n

    shared_mem_accum = M * K * bitwidth // 8 + K * N * bitwidth // 8
    if triton.runtime.driver.active.utils.get_device_properties(
            triton.runtime.driver.active.get_current_device())["max_shared_mem"] < shared_mem_accum:
        pytest.skip("Skipped due to insufficient shared memory on this GPU.")

    torch_dtype = torch_dtype_map[bitwidth]
    gl_acc_dtype = acc_dtype_map[acc_dtype]
    out_dtype = torch.float32

    block_layout = ttgl.BlockedLayout([1, 1], [1, THREADS_PER_WARP], warps_per_cta=warps, order=[1, 0])
    shared_layout_a = ttgl.NVMMASharedLayout(swizzle_byte_width=swizzling_a, element_bitwidth=bitwidth, rank=2,
                                             transposed=transpose_a)
    shared_layout_b = ttgl.NVMMASharedLayout(swizzle_byte_width=swizzling_b, element_bitwidth=bitwidth, rank=2,
                                             transposed=transpose_b)
    mma_layout = ttgl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=warps, instr_shape=instr_shape)

    torch.manual_seed(0)

    def cast(x, dtype):
        if dtype != torch.float32:
            return x.to(torch_dtype)
        else:
            # zero-out the lower 13 bits
            x = x.view(torch.int32)
            x = x & ~((1 << 13) - 1)
            return x.view(dtype)

    # Sample bf16 as tf32 does not use the full range
    a = cast(torch.randn((M, K), device="cuda", dtype=torch.float32), torch_dtype)
    b = cast(torch.randn((K, N), device="cuda", dtype=torch.float32), torch_dtype)
    out = torch.zeros((M, N), device="cuda", dtype=out_dtype)

    warpgroup_mma_kernel[(1, )](
        a,
        b,
        out,
        M,
        N,
        K,
        block_layout,
        mma_layout,
        shared_layout_a,
        shared_layout_b,
        gl_acc_dtype,
        False,
        num_warps=warps[0] * warps[1],
    )

    try:
        allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = True
        allow_fp16_red = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = acc_dtype == torch.float16
        ref = torch.matmul(a.to(acc_dtype), b.to(acc_dtype)).to(out_dtype)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = allow_fp16_red

    if bitwidth == 8:
        atol, rtol = 0.5, 0.5
    elif bitwidth == 16:
        atol, rtol = 3e-2, 1e-1
    else:
        atol, rtol = 5e-4, 5e-3
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires CDNA4")
@pytest.mark.parametrize("use_buffer_load", [True, False])
def test_amd_direct_load_to_shared(use_buffer_load):

    @gluon.jit
    def kernel(a_ptr, b_ptr, use_buffer_load: ttgl.constexpr):
        blocked: ttgl.constexpr = ttgl.BlockedLayout([1, 8], [32, 2], [4, 1], [1, 0])
        shared: ttgl.constexpr = ttgl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])

        smem = ttgl.allocate_shared_memory(a_ptr.dtype.element_ty, [128, 16], shared)
        offsets = ttgl.arange(0, 128, layout=ttgl.SliceLayout(1, blocked))[:, None] * 16 + \
                  ttgl.arange(0, 16, layout=ttgl.SliceLayout(0, blocked))[None, :]
        if use_buffer_load:
            cdna4_async_copy.buffer_load_to_shared(smem, a_ptr, offsets)
        else:
            cdna4_async_copy.global_load_to_shared(smem, a_ptr + offsets)

        cdna4_async_copy.async_wait(0)
        a = cdna4_async_copy.load_shared_relaxed(smem, blocked)

        ttgl.store(b_ptr + offsets, a)

    torch.manual_seed(0)
    a = torch.randn((128, 16), dtype=torch.float16, device='cuda')
    b = torch.empty_like(a)
    pgm = kernel[(1, )](a, b, use_buffer_load)

    torch.testing.assert_close(a, b)
    assert re.search(r'ttg\.local_load .* \{ttg\.amdgpu\.syncedViaAsyncWait = true\}', pgm.asm['ttgir'], re.MULTILINE)
    if use_buffer_load:
        assert re.search(r"buffer_load.*lds$", pgm.asm['amdgcn'], re.MULTILINE)
    else:
        assert re.search(r"global_load_lds", pgm.asm['amdgcn'], re.MULTILINE)
    assert 'vmcnt(0)' in pgm.asm['amdgcn']


@pytest.mark.skipif(not (is_hip_gfx11() or is_hip_gfx12()), reason="Requires RDNA3 or RDNA4")
@pytest.mark.parametrize("M, N, K", [(64, 64, 64)])
@pytest.mark.parametrize("in_dtype", ['float16', 'bfloat16'])
def test_amd_wmma(M, N, K, in_dtype):

    @gluon.jit
    def kernel(a_ptr, b_ptr, c_ptr,  #
               stride_am, stride_ak,  #
               stride_bk, stride_bn,  #
               stride_cm, stride_cn,  #
               BLOCK_SIZE_M: ttgl.constexpr,  #
               BLOCK_SIZE_N: ttgl.constexpr,  #
               BLOCK_SIZE_K: ttgl.constexpr,  #
               BLOCKED_LAYOUT: ttgl.constexpr,  #
               WMMA_LAYOUT: ttgl.constexpr,  #
               K_WIDTH: ttgl.constexpr):
        offs_am = ttgl.arange(0, BLOCK_SIZE_M, layout=ttgl.SliceLayout(1, BLOCKED_LAYOUT))
        offs_bn = ttgl.arange(0, BLOCK_SIZE_N, layout=ttgl.SliceLayout(0, BLOCKED_LAYOUT))

        offs_ak = ttgl.arange(0, BLOCK_SIZE_K, layout=ttgl.SliceLayout(0, BLOCKED_LAYOUT))
        offs_bk = ttgl.arange(0, BLOCK_SIZE_K, layout=ttgl.SliceLayout(1, BLOCKED_LAYOUT))

        offs_a = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
        offs_b = offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn

        a = ttgl.load(a_ptr + offs_a)
        b = ttgl.load(b_ptr + offs_b)

        a = ttgl.convert_layout(a, layout=ttgl.DotOperandLayout(0, WMMA_LAYOUT, K_WIDTH))
        b = ttgl.convert_layout(b, layout=ttgl.DotOperandLayout(1, WMMA_LAYOUT, K_WIDTH))

        acc = ttgl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], ttgl.float32, WMMA_LAYOUT)
        if WMMA_LAYOUT.version == 1:
            c = ttgl.amd.rdna3.wmma(a, b, acc)
        else:
            ttgl.static_assert(WMMA_LAYOUT.version == 2, "WMMA_LAYOUT.version must be 1 or 2")
            c = ttgl.amd.rdna4.wmma(a, b, acc)
        c = c.to(a_ptr.dtype.element_ty)

        offs_cm = ttgl.arange(0, BLOCK_SIZE_M, layout=ttgl.SliceLayout(1, WMMA_LAYOUT))
        offs_cn = ttgl.arange(0, BLOCK_SIZE_N, layout=ttgl.SliceLayout(0, WMMA_LAYOUT))
        offs_c = offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
        ttgl.store(c_ptr + offs_c, c)

    elem_type = torch.float16 if in_dtype == 'float16' else torch.bfloat16
    a = torch.randn((M, K), device='cuda', dtype=elem_type)
    b = torch.randn((K, N), device='cuda', dtype=elem_type)
    c = torch.empty((M, N), device=a.device, dtype=elem_type)

    blocked = ttgl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0])
    wmma_version = 1 if is_hip_gfx11() else 2
    k_width = 16 if is_hip_gfx11() else 8
    wmma = ttgl.amd.AMDWMMALayout(wmma_version, True, [2, 2])
    kernel[1, 1](a, b, c, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), BLOCK_SIZE_M=M,
                 BLOCK_SIZE_N=N, BLOCK_SIZE_K=K, BLOCKED_LAYOUT=blocked, WMMA_LAYOUT=wmma, K_WIDTH=k_width, num_warps=4)

    ref = torch.matmul(a, b)
    triton_output = c
    torch.testing.assert_close(ref, triton_output)


@pytest.mark.skipif(not (is_hip_cdna3() or is_hip_cdna4()), reason="Requires CDNA3 or CDNA4")
@pytest.mark.parametrize("M, N, K", [(32, 32, 16), (16, 16, 32)])
@pytest.mark.parametrize("in_dtype", ['float16', 'bfloat16'])
@pytest.mark.parametrize("num_warps", [4, 8])
@pytest.mark.parametrize("cdna_version", [3, 4])
def test_amd_mfma(M, N, K, in_dtype, num_warps, cdna_version):
    if is_hip_cdna3() and cdna_version != 3:
        pytest.skip("On CDNA3 target, skip if mfma version is not 3")

    if is_hip_cdna4() and cdna_version != 4:
        pytest.skip("On CDNA4 target, skip if mfma version is not 4")

    @gluon.jit
    def kernel(a_ptr, b_ptr, c_ptr,  #
               stride_am, stride_ak,  #
               stride_bk, stride_bn,  #
               stride_cm, stride_cn,  #
               BLOCK_SIZE_M: ttgl.constexpr, BLOCK_SIZE_N: ttgl.constexpr, BLOCK_SIZE_K: ttgl.constexpr,
               blocked: ttgl.constexpr, k_width: ttgl.constexpr, mfma_layout: ttgl.constexpr):
        dot_a_layout: ttgl.constexpr = ttgl.DotOperandLayout(operand_index=0, parent=mfma_layout, k_width=k_width)
        dot_b_layout: ttgl.constexpr = ttgl.DotOperandLayout(operand_index=1, parent=mfma_layout, k_width=k_width)

        offs_am = ttgl.arange(0, BLOCK_SIZE_M, layout=ttgl.SliceLayout(1, blocked))
        offs_bn = ttgl.arange(0, BLOCK_SIZE_N, layout=ttgl.SliceLayout(0, blocked))

        offs_ak = ttgl.arange(0, BLOCK_SIZE_K, layout=ttgl.SliceLayout(0, blocked))
        offs_bk = ttgl.arange(0, BLOCK_SIZE_K, layout=ttgl.SliceLayout(1, blocked))
        offs_a = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
        offs_b = offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn

        a = ttgl.amd.cdna3.buffer_load(ptr=a_ptr, offsets=offs_a)
        b = ttgl.amd.cdna3.buffer_load(ptr=b_ptr, offsets=offs_b)
        a1 = ttgl.convert_layout(a, layout=dot_a_layout)
        b1 = ttgl.convert_layout(b, layout=dot_b_layout)
        acc = ttgl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], ttgl.float32, mfma_layout)
        c = ttgl.amd.cdna3.mfma(a1, b1, acc)
        c = ttgl.convert_layout(c, layout=blocked)
        c = c.to(a_ptr.dtype.element_ty)

        offs_cm = ttgl.arange(0, BLOCK_SIZE_M, layout=ttgl.SliceLayout(1, blocked))
        offs_cn = ttgl.arange(0, BLOCK_SIZE_N, layout=ttgl.SliceLayout(0, blocked))
        offs_c = offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
        ttgl.amd.cdna3.buffer_store(stored_value=c, ptr=c_ptr, offsets=offs_c)

    elem_type = torch.float16 if in_dtype == 'float16' else torch.bfloat16
    a = torch.randn((M, K), device='cuda', dtype=elem_type) - 0.5
    b = torch.randn((K, N), device='cuda', dtype=elem_type) - 0.5
    c = torch.empty((M, N), device=a.device, dtype=elem_type)
    nonkdim: ttgl.constexpr = 32
    kdim: ttgl.constexpr = 8 if cdna_version == 3 else 16
    k_width: ttgl.constexpr = 4 if cdna_version == 3 else 8
    blocked: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[4, 4], threads_per_warp=[4, 16],
                                                 warps_per_cta=[num_warps, 1], order=[1, 0])
    mfma_layout: ttgl.constexpr = ttgl.amd.AMDMFMALayout(version=cdna_version, instr_shape=[nonkdim, nonkdim, kdim],
                                                         transposed=True, warps_per_cta=[num_warps, 1])

    kernel[1, 1](
        a, b, c,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        BLOCK_SIZE_M=M, BLOCK_SIZE_N=N, BLOCK_SIZE_K=K,  #
        blocked=blocked, k_width=k_width, mfma_layout=mfma_layout,  #
        num_warps=num_warps)

    ref = torch.matmul(a, b)
    triton_output = c
    torch.testing.assert_close(ref, triton_output)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires CDNA4")
@pytest.mark.parametrize("M, N, K, rhs_scale, mxfp_type, normal_type", [(32, 32, 128, rhs_scale, mxfp_type, normal_type)
                                                                        for rhs_scale in [True, False]
                                                                        for mxfp_type in ["e2m1"]
                                                                        for normal_type in ["e4m3", "e5m2"]])
def test_amd_mfma_scaled(M, N, K, rhs_scale, mxfp_type, normal_type):
    device = 'cuda'

    @triton.jit
    def triton_kernel(a_base, stride_am, stride_ak, a_scale,  #
                      b_base, stride_bk, stride_bn, b_scale,  #
                      out,  #
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,  #
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
        tl.store(out_ptr, c.to(tl.bfloat16))

    @gluon.jit
    def gluon_kernel(a_base, stride_am, stride_ak, a_scale,  #
                     b_base, stride_bk, stride_bn, b_scale,  #
                     out,  #
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,  #
                     type_a: tl.constexpr, type_b: tl.constexpr):
        DIV_FACTOR_A: tl.constexpr = 2 if type_a == "e2m1" else 1
        DIV_FACTOR_B: tl.constexpr = 2 if type_b == "e2m1" else 1
        PACKED_BLOCK_K_A: tl.constexpr = BLOCK_K // DIV_FACTOR_A
        PACKED_BLOCK_K_B: tl.constexpr = BLOCK_K // DIV_FACTOR_B
        SCALE_BLOCK_K: tl.constexpr = BLOCK_K // 32

        a_unpacked_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 16], [8, 8], [4, 1], [1, 0])
        a_packed_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 8], [8, 8], [4, 1], [1, 0])
        a_layout: ttgl.constexpr = a_packed_layout if type_a == "e2m1" else a_unpacked_layout

        a_scale_layout: ttgl.constexpr = ttgl.DistributedLinearLayout(
            reg_bases=[], lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2]], warp_bases=[[0, 0], [16, 0]],
            block_bases=[], shape=[32, 4])

        b_unpacked_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 16], [32, 2], [4, 1], [1, 0])
        b_packed_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 8], [16, 4], [4, 1], [1, 0])
        b_layout: ttgl.constexpr = b_packed_layout if type_b == "e2m1" else b_unpacked_layout

        b_scale_layout: ttgl.constexpr = ttgl.DistributedLinearLayout(
            reg_bases=[], lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2]], warp_bases=[[16, 0], [0, 0]],
            block_bases=[], shape=[32, 4])

        mfma_layout: ttgl.constexpr = ttgl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 128], transposed=True,
                                                             warps_per_cta=[2, 2])

        zero = ttgl.zeros([BLOCK_M, BLOCK_N], dtype=ttgl.float32, layout=mfma_layout)

        a_offsets = ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, a_layout))[:, None] * stride_am + \
                    ttgl.arange(0, PACKED_BLOCK_K_A, layout=ttgl.SliceLayout(0, a_layout))[None, :] * stride_ak
        a = ttgl.amd.cdna4.buffer_load(a_base, a_offsets)
        a = ttgl.convert_layout(a, ttgl.DotOperandLayout(operand_index=0, parent=mfma_layout, k_width=16))

        b_offsets = ttgl.arange(0, PACKED_BLOCK_K_B, layout=ttgl.SliceLayout(1, b_layout))[:, None] * stride_bk + \
                    ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, b_layout))[None, :] * stride_bn
        b = ttgl.amd.cdna4.buffer_load(b_base, b_offsets)
        b = ttgl.convert_layout(b, ttgl.DotOperandLayout(operand_index=1, parent=mfma_layout, k_width=16))

        if a_scale is not None:
            a_scale_offsets = ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, a_scale_layout))[:, None] * SCALE_BLOCK_K + \
                              ttgl.arange(0, SCALE_BLOCK_K, layout=ttgl.SliceLayout(0, a_scale_layout))[None, :]
            a_scale = ttgl.amd.cdna4.buffer_load(a_scale, a_scale_offsets)
        else:
            a_scale = ttgl.full([BLOCK_M, SCALE_BLOCK_K], 127, dtype=ttgl.int8, layout=a_scale_layout)

        if b_scale is not None:
            b_scale_offsets = ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(1, b_scale_layout))[:, None] * SCALE_BLOCK_K + \
                              ttgl.arange(0, SCALE_BLOCK_K, layout=ttgl.SliceLayout(0, b_scale_layout))[None, :]
            b_scale = ttgl.amd.cdna4.buffer_load(b_scale, b_scale_offsets)
        else:
            b_scale = ttgl.full([BLOCK_M, SCALE_BLOCK_K], 127, dtype=ttgl.int8, layout=b_scale_layout)

        c = ttgl.amd.cdna4.mfma_scaled(a, a_scale, type_a, b, b_scale, type_b, zero)
        c = c.to(out.dtype.element_ty)

        out_offsets = ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, mfma_layout))[:, None] * BLOCK_N + \
                      ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, mfma_layout))[None, :]
        ttgl.amd.cdna4.buffer_store(c, out, out_offsets)

    torch.manual_seed(0)

    type_a = normal_type if rhs_scale else mxfp_type
    type_b = mxfp_type if rhs_scale else normal_type

    DIV_FACTOR_A = 2 if type_a == "e2m1" else 1
    DIV_FACTOR_B = 2 if type_b == "e2m1" else 1
    x = torch.randint(20, 40, (M, K // DIV_FACTOR_A), dtype=torch.uint8, device=device)
    y = torch.randint(20, 40, (K // DIV_FACTOR_B, N), dtype=torch.uint8, device=device)

    min_scale, max_scale = (0, 142)
    scale_x = torch.randint(min_scale, max_scale + 1, (M, K // 32), dtype=torch.uint8, device=device)
    scale_y = torch.randint(min_scale, max_scale + 1, (N, K // 32), dtype=torch.uint8, device=device)
    if rhs_scale:
        scale_x = None
    else:
        scale_y = None

    def make_finite(x, dtype):
        if dtype not in ("e5m2", "e4m3"):
            return x
        mask = 0x7C if dtype == "e5m2" else 0x7F
        finite = torch.arange(x.numel(), device=device, dtype=torch.uint8).reshape_as(x) % mask
        x_finite = torch.where(x & mask == mask, finite | (0x80 & x), x)
        x.copy_(x_finite)
        return x

    x = make_finite(x, type_a)
    y = make_finite(y, type_b)

    z = torch.zeros((M, N), dtype=torch.bfloat16, device=device)
    pgm = gluon_kernel[(1, )](x, *x.stride(), scale_x, y, *y.stride(), scale_y, z, M, N, K, type_a, type_b)
    assert "v_mfma_scale_f32_16x16x128_f8f6f4" in pgm.asm["amdgcn"]

    z_ref = torch.zeros((M, N), dtype=torch.bfloat16, device=device)
    triton_kernel[(1, )](x, *x.stride(), scale_x, y, *y.stride(), scale_y, z_ref, M, N, K, type_a, type_b)

    torch.testing.assert_close(z, z_ref, rtol=1e-5, atol=1e-5)


def test_math_fast_expf():

    @gluon.jit
    def fast_expf_kernel(x_ptr, y_ptr, warp_size: ttgl.constexpr, num_warps: ttgl.constexpr):
        blocked: ttgl.constexpr = ttgl.BlockedLayout([1], [warp_size], [num_warps], [0])

        offs = ttgl.arange(0, warp_size * num_warps, layout=blocked)
        x = ttgl.load(x_ptr + offs)
        y = libdevice.fast_expf(x)
        ttgl.store(y_ptr + offs, y)

    num_warps = 4

    torch.manual_seed(0)
    x = torch.randn(THREADS_PER_WARP * num_warps, device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)
    fast_expf_kernel[(1, )](x, y, THREADS_PER_WARP, num_warps)
    torch.testing.assert_close(y, torch.exp(x), atol=1e-5, rtol=1e-4)


def test_math_fast_dividef():

    @gluon.jit
    def fast_dividef_kernel(x_ptr, y_ptr, z_ptr, warp_size: ttgl.constexpr, num_warps: ttgl.constexpr):
        blocked: ttgl.constexpr = ttgl.BlockedLayout([1], [warp_size], [num_warps], [0])

        offs = ttgl.arange(0, warp_size * num_warps, layout=blocked)
        x = ttgl.load(x_ptr + offs)
        y = ttgl.load(y_ptr + offs)
        z = libdevice.fast_dividef(x, y)
        ttgl.store(z_ptr + offs, z)

    num_warps = 4

    torch.manual_seed(0)
    x = torch.randn(THREADS_PER_WARP * num_warps, device="cuda", dtype=torch.float32)
    y = torch.randn_like(x)
    z = torch.empty_like(x)
    y[y == 0] = 1.0
    fast_dividef_kernel[(1, )](x, y, z, THREADS_PER_WARP, num_warps)
    torch.testing.assert_close(z, torch.div(x, y), atol=1e-5, rtol=1e-4)


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_tmem_copy_2d():
    device = "cuda"

    smem_h = 64
    smem_w = 16
    num_rows = 128
    num_cols = smem_h * smem_w // 32

    @gluon.jit
    def kernel(in_ptr, out_ptr, smem_h: ttgl.constexpr, smem_w: ttgl.constexpr, num_rows: ttgl.constexpr,
               num_cols: ttgl.constexpr):
        in_ptrs = in_ptr + ttgl.arange(0, smem_h)[:, None] * smem_w + ttgl.arange(0, smem_w)[None, :]
        out_ptrs = out_ptr + ttgl.arange(0, num_rows)[:, None] * num_cols + ttgl.arange(0, num_cols)[None, :]

        blocked: ttgl.constexpr = ttgl.BlockedLayout([1, 4], [32, 1], [4, 1], [1, 0])
        value = ttgl.load(ttgl.set_auto_layout(in_ptrs, blocked))

        smem_layout: ttgl.constexpr = ttgl.SharedLinearLayout(
            offset_bases=[[0, 1], [0, 2], [32, 0], [0, 4], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 8]])
        tmem_layout: ttgl.constexpr = TensorMemoryScalesLayout()
        smem = ttgl.allocate_shared_memory(ttgl.int8, (smem_h, smem_w), layout=smem_layout)
        tmem = allocate_tensor_memory(ttgl.int8, (smem_h, smem_w), layout=tmem_layout)

        barrier = ttgl.allocate_shared_memory(ttgl.int64, [1], ttgl.constexpr(mbarrier.MBarrierLayout()))
        mbarrier.init(barrier, count=1)

        smem.store(value)
        fence_async_shared()
        tcgen05_copy(smem, tmem)
        tcgen05_commit(barrier)
        mbarrier.wait(barrier, phase=0)
        tmem_alias: ttgl.constexpr = TensorMemoryLayout((num_rows, num_cols), col_stride=1)
        tmem = tmem._reinterpret(ttgl.int8, (num_rows, num_cols), tmem_alias)
        value = tmem.load(blocked)
        ttgl.static_print(ttgl.to_linear_layout(blocked, (smem_h, smem_w)))
        ttgl.static_print(ttgl.to_linear_layout(blocked, (num_rows, num_cols)))
        ttgl.store(ttgl.set_auto_layout(out_ptrs, blocked), value)

    torch.manual_seed(0)
    x = torch.randint(size=(smem_h, smem_w), low=-100, high=100, dtype=torch.int8).to(device)
    #x = torch.arange(smem_h * smem_w, dtype=torch.int8, device=device).reshape(smem_h, smem_w)
    z_tri = torch.zeros(size=(num_rows, num_cols), dtype=torch.int8).to(device)
    kernel[(1, )](x, z_tri, smem_h, smem_w, num_rows, num_cols)

    # offset_bases=[[0, 1], [0, 2], [32, 0], [0, 4], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 8]],
    # Split into contiguous shmem chunks
    x_res = x.reshape(2, 32, 2, 2, 4)
    # Put tmem cols first then rows
    x_res = x_res.permute(1, 2, 3, 0, 4)
    # Reshape as 32xnum_cols
    x_res = x_res.reshape(num_rows // 4, num_cols)

    warps = torch.chunk(z_tri, chunks=4, dim=0)
    for warp in warps:
        torch.testing.assert_close(x_res, warp)


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_tmem_subslice_block_m_64():

    @gluon.jit
    def kernel(s_ptr, out_ptr):
        BLOCK_M: ttgl.constexpr = 64
        N: ttgl.constexpr = 128
        BLOCK_N: ttgl.constexpr = 64

        tmem_layout: ttgl.constexpr = TensorMemoryLayout((BLOCK_M, BLOCK_N), col_stride=1)
        s_tmem = allocate_tensor_memory(ttgl.float32, (BLOCK_M, N), layout=tmem_layout)
        o_tmem = allocate_tensor_memory(ttgl.float32, (BLOCK_M, N), layout=tmem_layout)

        layout: ttgl.constexpr = get_tmem_32x32b_reg_layout(BLOCK_M, BLOCK_N, (BLOCK_M, N), num_warps=4)

        offsets = ttgl.arange(0, BLOCK_M)[:, None] * N + ttgl.arange(0, N)[None, :]
        offsets = ttgl.set_auto_layout(offsets, layout)
        s = ttgl.load(s_ptr + offsets)

        s_tmem.store(s)
        o_tmem.store(s)

        p_tmem_layout: ttgl.constexpr = TensorMemoryLayout((BLOCK_M, BLOCK_N), col_stride=1)
        p_tmem = s_tmem.slice(0, N // 2)._reinterpret(ttgl.float16, [BLOCK_M, N], p_tmem_layout)
        p_tmem.store(ttgl.full((BLOCK_M, N), 0.0, dtype=ttgl.float16, layout=layout))

        d1_tmem_layout: ttgl.constexpr = TensorMemoryLayout((BLOCK_M, 2), col_stride=1)
        d1_layout: ttgl.constexpr = get_tmem_32x32b_reg_layout(BLOCK_M, 2, (BLOCK_M, 2), num_warps=4)

        m_tmem = s_tmem.slice(N // 4, 2)._reinterpret(ttgl.float32, [BLOCK_M, 2], d1_tmem_layout)
        m_tmem.store(ttgl.full((BLOCK_M, 2), 2.0, dtype=ttgl.float32, layout=d1_layout))
        l_tmem = s_tmem.slice(N // 4 + 2, 2)._reinterpret(ttgl.float32, [BLOCK_M, 2], d1_tmem_layout)
        l_tmem.store(ttgl.full((BLOCK_M, 2), 3.0, dtype=ttgl.float32, layout=d1_layout))
        a_tmem = s_tmem.slice(N // 4 + 4, 2)._reinterpret(ttgl.float32, [BLOCK_M, 2], d1_tmem_layout)
        a_tmem.store(ttgl.full((BLOCK_M, 2), 4.0, dtype=ttgl.float32, layout=d1_layout))

        s = s_tmem.load(layout)

        ttgl.store(out_ptr + offsets, s)

    torch.manual_seed(0)
    s = torch.randn((64, 128), dtype=torch.float32, device="cuda")

    out_tri = torch.empty_like(s)
    compiled = kernel[(1, )](s, out_tri)

    ttgir = compiled.asm["ttgir"]
    # Check that we have two 64x128xf32 allocations.
    assert ttgir.count("ttng.tmem_alloc") == 2
    assert ttgir.count("ttng.tmem_alloc : () -> !ttg.memdesc<64x128xf32") == 2

    # Check that we allocated only 128 columns of TMEM.
    llir = compiled.asm["llir"]
    assert llir.count("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [$1], 128")

    # Given TMEM[0:32] is the slice of TMEM for warpgroup 0, the expected layout
    # of S is
    #
    #   TMEM[0:16]  = S[0:16, 0:64]
    #   TMEM[16:32] = S[0:16, 64:128]
    #
    # When slicing S to obtain P, we expect it to overlap with the left half,
    # i.e. S[0:16, 0:32] and S[0:16, 64:96].
    out_ref = s
    out_ref[:, 0:32] = 0.0
    out_ref[:, 64:96] = 0.0

    # Given S = [s0, s1, s2, s3], they are arranged like
    #
    #   TMEM[0:16]  = [s0, s1]
    #   TMEM[16:32] = [s2, s3]
    #
    # Thus slicing S at  N//4 will obtain an offset to the beginning of s1.
    out_ref[:, 32:34] = 2.0
    out_ref[:, 34:36] = 3.0
    out_ref[:, 36:38] = 4.0

    torch.testing.assert_close(out_ref, out_tri, atol=0, rtol=0)


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_block_m_64_mma():

    @gluon.jit
    def kernel(a_ptr, b_ptr, c_ptr, d_ptr):
        BLOCK_M: ttgl.constexpr = 64
        N: ttgl.constexpr = 128
        BLOCK_N: ttgl.constexpr = 64

        a_offsets = ttgl.arange(0, BLOCK_M)[:, None] * N + ttgl.arange(0, N)[None, :]
        b_offsets = ttgl.arange(0, N)[:, None] * N + ttgl.arange(0, N)[None, :]

        a_layout: ttgl.constexpr = get_tmem_32x32b_reg_layout(BLOCK_M, BLOCK_N, (BLOCK_M, N), num_warps=4)
        b_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [1, 32], [4, 1], [1, 0])
        a_offsets = ttgl.set_auto_layout(a_offsets, a_layout)
        b_offsets = ttgl.set_auto_layout(b_offsets, b_layout)

        a = ttgl.load(a_ptr + a_offsets)
        b = ttgl.load(b_ptr + b_offsets)
        c = ttgl.load(c_ptr + a_offsets)

        a_tmem_layout: ttgl.constexpr = TensorMemoryLayout((BLOCK_M, BLOCK_N), col_stride=1)
        acc_tmem_layout: ttgl.constexpr = TensorMemoryLayout((BLOCK_M, BLOCK_N), col_stride=1)
        al_tmem = allocate_tensor_memory(ttgl.float16, (BLOCK_M, N), layout=a_tmem_layout)
        ar_tmem = allocate_tensor_memory(ttgl.float16, (BLOCK_M, N), layout=a_tmem_layout)
        acc_tmem = allocate_tensor_memory(ttgl.float32, (BLOCK_M, N), layout=acc_tmem_layout)

        a0, a1 = a.reshape((BLOCK_M, 2, N // 2)).permute(0, 2, 1).split()

        al = ttgl.join(a0, a1).permute(0, 2, 1).reshape((BLOCK_M, N))
        ar = ttgl.join(a1, a0).permute(0, 2, 1).reshape((BLOCK_M, N))

        al_tmem.store(ttgl.convert_layout(al, a_layout, assert_trivial=True))
        ar_tmem.store(ttgl.convert_layout(ar, a_layout, assert_trivial=True))

        b_shared_layout: ttgl.constexpr = ttgl.NVMMASharedLayout(swizzle_byte_width=32, element_bitwidth=16, rank=2)
        b_shared = ttgl.allocate_shared_memory(ttgl.float16, [N, N], layout=b_shared_layout)
        b_shared.store(b)

        acc_tmem.store(c)

        bar = ttgl.allocate_shared_memory(ttgl.int64, [1], ttgl.constexpr(mbarrier.MBarrierLayout()))
        mbarrier.init(bar, count=1)

        # This is a manually tiled MMA where LHS is in TMEM with blockM=64,
        # where we circumvent the limitation that LHS and accumulator need to
        # share the same TMEM rows by storing the LHS twice.
        #
        # TMEM      al   ar   c
        # [0, 16)   a0   a1   c0
        # [16, 32)  a1   a0   c1
        #
        # d0 = a0 @ b00 + a1 @ b10 + c0
        # d1 = a0 @ b10 + a1 @ b11 + c1

        N2: ttgl.constexpr = N // 2
        c0 = acc_tmem.slice(0, N2)
        c1 = acc_tmem.slice(N2, N2)

        tcgen05_mma(al_tmem.slice(0, N2), b_shared.slice(0, N2, dim=0).slice(0, N2, dim=1), c0)
        tcgen05_mma(ar_tmem.slice(0, N2), b_shared.slice(N2, N2, dim=0).slice(0, N2, dim=1), c0)
        tcgen05_mma(ar_tmem.slice(N2, N2), b_shared.slice(0, N2, dim=0).slice(N2, N2, dim=1), c1)
        tcgen05_mma(al_tmem.slice(N2, N2), b_shared.slice(N2, N2, dim=0).slice(N2, N2, dim=1), c1)

        tcgen05_commit(bar)
        mbarrier.wait(bar, 0)
        mbarrier.invalidate(bar)

        d = acc_tmem.load(a_layout)
        ttgl.store(d_ptr + a_offsets, d)

    torch.manual_seed(0)
    a = torch.randn((64, 128), dtype=torch.float16, device="cuda")
    b = torch.randn((128, 128), dtype=torch.float16, device="cuda")
    c = torch.randn((64, 128), dtype=torch.float32, device="cuda")

    d_tri = torch.empty_like(c)
    compiled = kernel[(1, )](a, b, c, d_tri)

    ttgir = compiled.asm["ttgir"]
    assert ttgir.count("ttng.tmem_alloc") == 3
    assert ttgir.count("ttng.tmem_alloc : () -> !ttg.memdesc<64x128xf32") == 1
    assert ttgir.count("ttng.tmem_alloc : () -> !ttg.memdesc<64x128xf16") == 2

    llir = compiled.asm["llir"]
    assert llir.count("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [$1], 128")

    d_ref = a @ b + c
    torch.testing.assert_close(d_ref, d_tri, rtol=0.08, atol=0)


def test_slice_reinterpret():
    BLOCK = ttgl.constexpr(2048)
    SPLIT_BLOCK = ttgl.constexpr(BLOCK // 2)
    XBLOCK = ttgl.constexpr(32)
    YBLOCK = ttgl.constexpr(SPLIT_BLOCK // 4 // XBLOCK)
    NUM_THREADS = ttgl.constexpr(THREADS_PER_WARP)

    @gluon.jit
    def kernel(in_ptr, out_ptr):
        smem_layout_1d: ttgl.constexpr = ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0])
        smem_layout_2d: ttgl.constexpr = ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
        smem = ttgl.allocate_shared_memory(ttgl.int8, [BLOCK], smem_layout_1d)
        smem_slice0 = smem.slice(0, SPLIT_BLOCK)
        smem_slice1 = smem.slice(SPLIT_BLOCK, SPLIT_BLOCK)._reinterpret(ttgl.int32, [XBLOCK, YBLOCK], smem_layout_2d)

        offs = ttgl.arange(0, XBLOCK)[:, None] * YBLOCK + ttgl.arange(0, YBLOCK)[None, :]
        blocked: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [1, NUM_THREADS], [1, 4], [1, 0])
        value = ttgl.load(ttgl.set_auto_layout(in_ptr + offs, blocked))

        blocked_1d: ttgl.constexpr = ttgl.BlockedLayout([1], [NUM_THREADS], [4], [0])
        smem_slice1.store(value)
        smem_slice0.store(ttgl.zeros((SPLIT_BLOCK, ), dtype=ttgl.int8, layout=blocked_1d))
        value = smem_slice1.load(blocked)
        ttgl.store(ttgl.set_auto_layout(out_ptr + offs, blocked), value)

    input = torch.randint(0, 100, (XBLOCK, YBLOCK), dtype=torch.int32, device="cuda")
    output = torch.empty_like(input)
    kernel[(1, )](input, output)
    torch.testing.assert_close(input, output, atol=0, rtol=0)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Requires Hopper")
def test_tma_slice():
    XBLOCK = YBLOCK = ttgl.constexpr(128)

    @gluon.jit
    def kernel(in_desc, out_desc):
        smem = ttgl.allocate_shared_memory(in_desc.dtype, [2 * XBLOCK, YBLOCK], in_desc.layout)
        smem_slice0 = smem.slice(0, XBLOCK)
        smem_slice1 = smem.slice(XBLOCK, XBLOCK)

        bar = ttgl.allocate_shared_memory(ttgl.int64, [1], ttgl.constexpr(mbarrier.MBarrierLayout()))
        mbarrier.init(bar, count=1)

        mbarrier.expect(bar, in_desc.block_type.nbytes)
        tma.async_copy_global_to_shared(in_desc, [0, 0], bar, smem_slice1)
        mbarrier.wait(bar, phase=0)

        blocked: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [1, 32], [1, 4], [1, 0])
        smem_slice0.store(ttgl.zeros((XBLOCK, YBLOCK), dtype=ttgl.float32, layout=blocked))

        tma.async_copy_shared_to_global(out_desc, [0, 0], smem_slice1)
        tma.store_wait(0)

    input = torch.rand((XBLOCK, YBLOCK), dtype=torch.float32, device="cuda")
    output = torch.empty_like(input)

    block_shape = [XBLOCK.value, YBLOCK.value]
    layout = ttgl.NVMMASharedLayout.get_default_for(block_shape, ttgl.float32)
    in_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(input, block_shape, layout)
    out_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(output, block_shape, layout)
    kernel[(1, )](in_desc, out_desc)
    torch.testing.assert_close(input, output, atol=0, rtol=0)


@pytest.mark.parametrize("swizzle", [32, 64, 128])
@pytest.mark.parametrize("num_warps", [4, 8])
@pytest.mark.parametrize("M, N, BLOCK_N", [(128, 128, 128), (256, 128, 64), (128, 128, 16)])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_tmem_copy_no_scales(M, N, BLOCK_N, num_warps, swizzle):

    @gluon.jit
    def tmem_copy_no_scales(in_ptr, out_ptr, M: ttgl.constexpr, N: ttgl.constexpr, BLOCK_N: ttgl.constexpr,
                            swizzle: ttgl.constexpr, num_warps: ttgl.constexpr):
        tmem_reg_layout: ttgl.constexpr = get_tmem_32x32b_reg_layout(
            M=128,
            N=BLOCK_N,
            shape=[M, N],
            num_warps=num_warps,
        )
        offs_m = ttgl.arange(0, M, ttgl.SliceLayout(1, tmem_reg_layout))
        offs_n = ttgl.arange(0, N, ttgl.SliceLayout(0, tmem_reg_layout))
        offs = offs_m[:, None] * N + offs_n[None, :]

        input = ttgl.load(in_ptr + offs)
        tmem_layout: ttgl.constexpr = TensorMemoryLayout(
            block=(128, BLOCK_N),
            col_stride=32 // in_ptr.dtype.element_ty.primitive_bitwidth,
        )

        smem_layout: ttgl.constexpr = ttgl.NVMMASharedLayout(swizzle_byte_width=swizzle, element_bitwidth=32, rank=2)
        smem = ttgl.allocate_shared_memory(in_ptr.dtype.element_ty, [M, N], layout=smem_layout)

        smem.store(input)
        tmem = allocate_tensor_memory(
            element_ty=in_ptr.dtype.element_ty,
            shape=[M, N],
            layout=tmem_layout,
        )
        bar = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
        mbarrier.init(bar, count=1)
        tcgen05_copy(smem, tmem)
        tcgen05_commit(bar)
        mbarrier.wait(bar, phase=0)
        output = tmem.load(tmem_reg_layout)
        ttgl.store(out_ptr + offs, output)

    input = torch.arange(M * N, device="cuda").reshape(M, N).to(torch.int32)
    output = torch.empty_like(input)

    tmem_copy_no_scales[(1, )](input, output, M, N, BLOCK_N, swizzle, num_warps=num_warps)
    assert (output == input).all()


@gluon.jit
def early_return_kernel(x):
    if x.sum(0).sum(0):
        return x
    x = x + x
    return x


def test_2d_tensor_early_return():
    warp_size = ttgl.constexpr(THREADS_PER_WARP)

    @gluon.jit
    def kernel(N, out):
        layout: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [1, warp_size], [1, 4], [1, 0])
        BLOCK: ttgl.constexpr = 32

        x0 = ttgl.arange(0, BLOCK, layout=ttgl.SliceLayout(1, layout))
        x1 = ttgl.arange(0, BLOCK, layout=ttgl.SliceLayout(0, layout))
        x = x0[:, None] * x1[None, :]
        for i in range(N):
            x += early_return_kernel(x)
        ttgl.store(out, x.sum(0).sum(0))

    out = torch.empty(1, dtype=torch.int32, device="cuda")
    compiled_kernel = kernel.warmup(N=100, out=out, grid=(1, ))
    assert compiled_kernel.asm["llir"].count("define") == 1


@pytest.mark.skipif(not is_hip_cdna3() and not is_hip_cdna4(), reason="Requires CDNA3 or CDNA4")
def test_inline_with_amdgpu_dialect():

    @gluon.jit
    def buffer_load(x, offsets):
        return ttgl.amd.cdna3.buffer_load(ptr=x, offsets=offsets)

    @gluon.jit
    def kernel(x, y):
        layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1], threads_per_warp=[64], warps_per_cta=[4],
                                                    order=[0])
        offsets = ttgl.arange(0, 64, layout=layout)

        a = buffer_load(x, offsets)
        ttgl.amd.cdna3.buffer_store(stored_value=a, ptr=y, offsets=offsets)

    input = torch.arange(64, device="cuda").to(torch.int32)
    output = torch.empty_like(input)

    compiled_kernel = kernel.warmup(input, output, grid=(1, ))
    assert compiled_kernel.asm["ttgir"].count("tt.func private") == 0


@pytest.mark.parametrize("interval_pairs", [[[32, 4]], [[16, 4]], [[16, 4], [64, 8]]])
@pytest.mark.parametrize(
    "shared_layout",
    [{"order": [0, 1]}, {"order": [1, 0]},
     {"offsets": [[0, 1], [0, 2], [0, 8], [0, 4], [0, 16], [0, 32], [2, 0], [1, 0], [4, 0], [8, 0], [16, 0], [32, 0]]}])
@pytest.mark.parametrize("slice_m_offset, slice_n_offset, slice_m, slice_n", [(48, 16, 16, 16), (32, 48, 32, 16),
                                                                              (48, 32, 16, 32)])
def test_padded_shared_layout_subslice(interval_pairs, shared_layout, slice_m_offset, slice_n_offset, slice_m, slice_n):
    m = 64
    n = 64
    num_warps = 1
    num_warps_cst = ttgl.constexpr(num_warps)
    warp_size_cst = ttgl.constexpr(THREADS_PER_WARP)

    shape = [m, n]
    if "order" in shared_layout:
        order = shared_layout["order"]
        smem_layout = ttgl.constexpr(ttgl.PaddedSharedLayout.with_identity_for(interval_pairs, shape, order))
    elif "offsets" in shared_layout:
        offsets = shared_layout["offsets"]
        blocks = []
        smem_layout = ttgl.constexpr(ttgl.PaddedSharedLayout(interval_pairs, offsets, blocks, shape))

    @gluon.jit
    def kernel(in_ptr, out_ptr, M: ttgl.constexpr, N: ttgl.constexpr, SLICE_M_OFFSET: ttgl.constexpr,
               SLICE_N_OFFSET: ttgl.constexpr, SLICE_M: ttgl.constexpr, SLICE_N: ttgl.constexpr):
        blocked: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [warp_size_cst, 1], [1, num_warps_cst], [1, 0])
        offs_m_load = ttgl.arange(0, M, ttgl.SliceLayout(1, blocked))
        offs_n_load = ttgl.arange(0, N, ttgl.SliceLayout(0, blocked))
        in_offs = offs_m_load[:, None] * N + offs_n_load[None, :]

        in_data = ttgl.load(in_ptr + in_offs)

        smem = ttgl.allocate_shared_memory(ttgl.int32, [M, N], smem_layout)
        smem_slice0 = smem.slice(SLICE_M_OFFSET, SLICE_M, dim=0)
        smem_slice1 = smem_slice0.slice(SLICE_N_OFFSET, SLICE_N, dim=1)

        smem.store(in_data)

        out_data = smem_slice1.load(blocked)

        offs_m_store = ttgl.arange(0, SLICE_M, ttgl.SliceLayout(1, blocked))
        offs_n_store = ttgl.arange(0, SLICE_N, ttgl.SliceLayout(0, blocked))
        out_offs = offs_m_store[:, None] * SLICE_N + offs_n_store[None, :]
        ttgl.store(out_ptr + out_offs, out_data)

    input = torch.arange(m * n, device="cuda").reshape(m, n).to(torch.int32)
    output = torch.zeros((slice_m, slice_n), dtype=torch.int32, device="cuda")
    ref_output = input[slice_m_offset:slice_m_offset + slice_m, slice_n_offset:slice_n_offset + slice_n]

    kernel[(1, )](input, output, m, n, slice_m_offset, slice_n_offset, slice_m, slice_n, num_warps=num_warps)

    assert (output == ref_output).all()


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("op, tol", [("add", 0), ("sub", 0), ("mul", 0), ("fma", 1e-6)])
def test_float2(op, tol):
    BLOCK_M = ttgl.constexpr(128)
    BLOCK_N = ttgl.constexpr(128)
    threads_per_warp = ttgl.constexpr(THREADS_PER_WARP)
    op = ttgl.constexpr(op)

    @gluon.jit
    def kernel(a_ptr, b_ptr, c_ptr, out_ptr):
        layout: ttgl.constexpr = ttgl.BlockedLayout(
            size_per_thread=[1, BLOCK_N],
            threads_per_warp=[threads_per_warp, 1],
            warps_per_cta=[ttgl.num_warps(), 1],
            order=[0, 1],
        )
        offs_m = ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, layout))[:, None]
        offs_n = ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, layout))[None, :]
        a = ttgl.load(a_ptr + offs_m * BLOCK_N + offs_n)
        b = ttgl.load(b_ptr + offs_m * BLOCK_N + offs_n)
        c = ttgl.load(c_ptr + offs_m * BLOCK_N + offs_n)
        a = float2.pack(a, axis=1)
        b = float2.pack(b, axis=1)
        c = float2.pack(c, axis=1)

        if op == "add":
            out = a + b
        elif op == "sub":
            out = a - b
        elif op == "mul":
            out = a * b
        elif op == "fma":
            out = float2.fma(a, b, c)

        out = float2.unpack(out, axis=1)
        ttgl.store(out_ptr + offs_m * BLOCK_N + offs_n, out)

    torch.manual_seed(0)
    shape = [BLOCK_M.value, BLOCK_N.value]
    a = torch.rand(shape, dtype=torch.float32, device="cuda")
    b = torch.rand(shape, dtype=torch.float32, device="cuda")
    c = torch.rand(shape, dtype=torch.float32, device="cuda")
    out = torch.empty(shape, dtype=torch.float32, device="cuda")

    kernel[(1, )](a, b, c, out)
    if op == "add":
        ref = a + b
    elif op == "sub":
        ref = a - b
    elif op == "mul":
        ref = a * b
    elif op == "fma":
        ref = a * b + c
    torch.testing.assert_close(ref, out, atol=tol, rtol=tol)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires CDNA4")
def test_buffer_atomic_rmw_add_bf16():
    BLOCK = 128
    elem_type = torch.bfloat16
    SIZE_PER_THREAD = 8

    @gluon.jit
    def kernel(a, BLOCK: ttgl.constexpr, SIZE_PER_THREAD: ttgl.constexpr):
        blocked: ttgl.constexpr = ttgl.BlockedLayout([SIZE_PER_THREAD], [64], [4], [0])
        offsets = ttgl.arange(0, BLOCK, layout=blocked)
        val = ttgl.full([BLOCK], 1.0, ttgl.bfloat16, layout=blocked)
        ttgl.amd.cdna4.buffer_atomic_add(a, offsets, val, mask=1, scope="cta", sem="relaxed")

    a = torch.randn((BLOCK), dtype=elem_type, device="cuda")
    origin_a = a.clone()
    compiled = kernel[(1, )](a, BLOCK, SIZE_PER_THREAD)

    torch_ref = origin_a + torch.ones((BLOCK, ), device='cuda', dtype=torch.bfloat16)
    torch.testing.assert_close(a, torch_ref)

    ttgir = compiled.asm["ttgir"]
    assert ttgir.count("amdgpu.buffer_atomic_rmw fadd, relaxed, cta") == 1

    llir = compiled.asm["llir"]
    assert llir.count("tail call <2 x bfloat> @llvm.amdgcn.raw.ptr.buffer.atomic.fadd.v2bf16") == SIZE_PER_THREAD // 2


@pytest.mark.skipif(not is_ampere_or_newer(), reason="Requires Ampere or newer")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_mma_v2(dtype):
    torch.manual_seed(42)
    B = ttgl.constexpr(128)
    threads_per_warp = ttgl.constexpr(THREADS_PER_WARP)

    @gluon.jit
    def kernel(a_ptr, b_ptr, c_ptr, out_ptr):
        layout: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [threads_per_warp, 1], [ttgl.num_warps(), 1], [1, 0])
        acc_layout: ttgl.constexpr = ttgl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[ttgl.num_warps(), 1],
                                                                 instr_shape=[16, 8])
        lhs_layout: ttgl.constexpr = ttgl.DotOperandLayout(parent=acc_layout, operand_index=0, k_width=8)
        rhs_layout: ttgl.constexpr = ttgl.DotOperandLayout(parent=acc_layout, operand_index=1, k_width=8)

        offs_m = ttgl.arange(0, B, layout=ttgl.SliceLayout(1, layout))[:, None]
        offs_n = ttgl.arange(0, B, layout=ttgl.SliceLayout(0, layout))[None, :]
        offs = offs_m * B + offs_n
        a = ttgl.convert_layout(ttgl.load(a_ptr + offs), lhs_layout)
        b = ttgl.convert_layout(ttgl.load(b_ptr + offs), rhs_layout)
        c = ttgl.convert_layout(ttgl.load(c_ptr + offs), acc_layout)
        if c.dtype == ttgl.bfloat16:
            out = mma_v2(a, b, c.to(ttgl.float32), input_precision="tf32").to(ttgl.bfloat16)
        else:
            out = mma_v2(a, b, c, input_precision="tf32")
        ttgl.store(out_ptr + offs, ttgl.convert_layout(out, layout))

    a = torch.randn((B, B), dtype=dtype, device="cuda")
    b = torch.randn((B, B), dtype=dtype, device="cuda")
    c = torch.randn((B, B), dtype=dtype, device="cuda")
    out = torch.empty((B, B), dtype=dtype, device="cuda")
    kernel[(1, )](a, b, c, out)
    torch.testing.assert_close(out, torch.addmm(c, a, b), atol=0.05, rtol=1e-2)


def test_dot_fma():
    torch.manual_seed(42)
    B = ttgl.constexpr(32)
    threads_per_warp = ttgl.constexpr(THREADS_PER_WARP)

    @gluon.jit
    def kernel(a_ptr, b_ptr, c_ptr, out_ptr):
        layout: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [threads_per_warp, 1], [ttgl.num_warps(), 1], [1, 0])
        lhs_layout: ttgl.constexpr = ttgl.DotOperandLayout(parent=layout, operand_index=0, k_width=0)
        rhs_layout: ttgl.constexpr = ttgl.DotOperandLayout(parent=layout, operand_index=1, k_width=0)

        offs_m = ttgl.arange(0, B, layout=ttgl.SliceLayout(1, layout))[:, None]
        offs_n = ttgl.arange(0, B, layout=ttgl.SliceLayout(0, layout))[None, :]
        offs = offs_m * B + offs_n
        a = ttgl.convert_layout(ttgl.load(a_ptr + offs), lhs_layout)
        b = ttgl.convert_layout(ttgl.load(b_ptr + offs), rhs_layout)
        c = ttgl.load(c_ptr + offs)
        out = ttgl.dot_fma(a, b, c)
        ttgl.store(out_ptr + offs, out)

    a = torch.rand((B, B), dtype=torch.float32, device="cuda")
    b = torch.ones((B, B), dtype=torch.float32, device="cuda")
    c = torch.rand((B, B), dtype=torch.float32, device="cuda")
    out = torch.empty((B, B), dtype=torch.float32, device="cuda")
    kernel[(1, )](a, b, c, out)
    torch.testing.assert_close(out, torch.addmm(c, a, b), atol=1e-2, rtol=1e-2)


@gluon.jit
def kernel_auto_layout_constant(threads_per_warp: ttgl.constexpr):
    BLOCK: ttgl.constexpr = 16
    SIZE: ttgl.constexpr = 10

    mask = ttgl.full(
        (BLOCK, BLOCK),
        True,
        ttgl.int1,
        ttgl.BlockedLayout(
            size_per_thread=[1, 1],
            threads_per_warp=[1, threads_per_warp],
            warps_per_cta=[1, 4],
            order=[1, 0],
        ),
    )

    mask &= (ttgl.arange(0, BLOCK, ttgl.AutoLayout()) < SIZE).expand_dims(0)
    mask &= (ttgl.arange(0, BLOCK, ttgl.AutoLayout()) < SIZE).expand_dims(1)


def test_auto_layout_constant():
    kernel_auto_layout_constant.warmup(THREADS_PER_WARP, grid=(1, ))
