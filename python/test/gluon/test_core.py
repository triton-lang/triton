import torch
import pytest
import re

import triton
import triton.language as tl

from triton._internal_testing import (
    is_ampere_or_newer,
    is_blackwell,
    is_hip_cdna3,
    is_hip_cdna4,
    is_hopper_or_newer,
    is_hopper,
)
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl
from triton.experimental.gluon.language.nvidia.ampere import async_copy
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
def warpgroup_mma_kernel(a, b, out, M: ttgl.constexpr, N: ttgl.constexpr, K: ttgl.constexpr, ASYNC: ttgl.constexpr):
    block_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [1, 32], [4, 1], [1, 0])
    mma_layout: ttgl.constexpr = ttgl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1],
                                                             instr_shape=[16, 32, 16])
    nvmma_layout: ttgl.constexpr = ttgl.NVMMASharedLayout(swizzle_byte_width=64, element_bitwidth=16, rank=2)

    a_offs_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, block_layout))[:, None]
    a_offs_n = ttgl.arange(0, K, layout=ttgl.SliceLayout(0, block_layout))[None, :]
    b_offs_m = ttgl.arange(0, K, layout=ttgl.SliceLayout(1, block_layout))[:, None]
    b_offs_n = ttgl.arange(0, N, layout=ttgl.SliceLayout(0, block_layout))[None, :]

    out_offs_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, mma_layout))[:, None]
    out_offs_n = ttgl.arange(0, N, layout=ttgl.SliceLayout(0, mma_layout))[None, :]

    acc = ttgl.zeros([M, N], dtype=a.dtype.element_ty, layout=mma_layout)
    A = ttgl.load(a + a_offs_m * K + a_offs_n)
    B = ttgl.load(b + b_offs_m * N + b_offs_n)

    a_shmem = ttgl.allocate_shared_memory(ttgl.float16, [M, K], nvmma_layout, A)
    b_shmem = ttgl.allocate_shared_memory(ttgl.float16, [K, N], nvmma_layout, B)

    fence_async_shared()
    acc = hopper.warpgroup_mma(a_shmem, b_shmem, acc, is_async=ASYNC)

    if ASYNC:
        acc = hopper.warpgroup_mma_wait(num_outstanding=0, deps=[acc])

    ttgl.store(out + out_offs_m * N + out_offs_n, acc)


@pytest.mark.skipif(not is_hopper(), reason="Requires Hopper")
@pytest.mark.parametrize("ASYNC", [True, False])
def test_warpgroup_mma(ASYNC):
    torch.manual_seed(0)
    M, N, K = 64, 32, 32
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    out = torch.zeros((M, N), device="cuda", dtype=torch.float16)
    warpgroup_mma_kernel[(1, )](a, b, out, M, N, K, ASYNC)

    ref = torch.matmul(a, b)

    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-1)


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


@pytest.mark.parametrize("M, N, K", [(32, 32, 16), (16, 16, 32)])
@pytest.mark.parametrize("in_dtype", ['float16', 'bfloat16'])
@pytest.mark.parametrize("num_warps", [4, 8])
@pytest.mark.parametrize("cdna_version", [3, 4])
def test_amd_mfma(M, N, K, in_dtype, num_warps, cdna_version):

    @gluon.jit
    def kernel(a_ptr, b_ptr, c_ptr, stride_am, stride_ak,  #
               stride_bk, stride_bn,  #
               stride_cm, stride_cn, BLOCK_SIZE_M: ttgl.constexpr, BLOCK_SIZE_N: ttgl.constexpr,
               BLOCK_SIZE_K: ttgl.constexpr, blocked: ttgl.constexpr, mfma_layout: ttgl.constexpr):
        dot_a_layout: ttgl.constexpr = ttgl.DotOperandLayout(operand_index=0, parent=mfma_layout, k_width=8)
        dot_b_layout: ttgl.constexpr = ttgl.DotOperandLayout(operand_index=1, parent=mfma_layout, k_width=8)

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

    if not is_hip_cdna4() and not is_hip_cdna3():
        pytest.skip("mfma quires target to be CDNA3 or CDNA4")

    if is_hip_cdna3() and cdna_version != 3:
        pytest.skip("On CDNA3 target, skip if mfma version is not 3")

    if is_hip_cdna4() and cdna_version != 4:
        pytest.skip("On CDNA4 target, skip if mfma version is not 4")

    elem_type = torch.float16 if in_dtype == 'float16' else torch.bfloat16
    a = torch.randn((M, K), device='cuda', dtype=elem_type) - 0.5
    b = torch.randn((K, N), device='cuda', dtype=elem_type) - 0.5
    c = torch.empty((M, N), device=a.device, dtype=elem_type)
    nonkdim: ttgl.constexpr = 32
    blocked: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[4, 4], threads_per_warp=[4, 16],
                                                 warps_per_cta=[num_warps, 1], order=[1, 0])
    mfma_layout: ttgl.constexpr = ttgl.amd.AMDMFMALayout(version=cdna_version, instr_shape=[nonkdim, nonkdim],
                                                         transposed=True, warps_per_cta=[num_warps, 1])

    kernel[1, 1](a, b, c, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), BLOCK_SIZE_M=M,
                 BLOCK_SIZE_N=N, BLOCK_SIZE_K=K, blocked=blocked, mfma_layout=mfma_layout, num_warps=num_warps)

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

        mfma_layout: ttgl.constexpr = ttgl.amd.AMDMFMALayout(version=4, warps_per_cta=[2, 2], tiles_per_warp=[1, 1],
                                                             instr_shape=[16, 16], transposed=True)

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


@pytest.mark.xfail(reason="copy to tmem with scale layout is currently broken in Gluon.")
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_tmem_copy_2d():
    device = "cuda"

    smem_h = 256
    smem_w = 4
    num_rows = 128
    num_cols = smem_h * smem_w // 32

    @gluon.jit
    def kernel(in_ptr, out_ptr, smem_h: ttgl.constexpr, smem_w: ttgl.constexpr, num_rows: ttgl.constexpr,
               num_cols: ttgl.constexpr):
        in_ptrs = in_ptr + ttgl.arange(0, smem_h)[:, None] * smem_w + ttgl.arange(0, smem_w)[None, :]
        out_ptrs = out_ptr + ttgl.arange(0, num_rows)[:, None] * num_cols + ttgl.arange(0, num_cols)[None, :]

        blocked: ttgl.constexpr = ttgl.BlockedLayout([1, 4], [32, 1], [4, 1], [0, 1])
        value = ttgl.load(ttgl.set_auto_layout(in_ptrs, blocked))

        smem_layout: ttgl.constexpr = ttgl.NVMMASharedLayout(swizzle_byte_width=0, element_bitwidth=8, rank=2)
        tmem_layout: ttgl.constexpr = TensorMemoryScalesLayout()
        smem = ttgl.allocate_shared_memory(ttgl.int8, (smem_h, smem_w), layout=smem_layout)
        tmem = allocate_tensor_memory(ttgl.int8, (num_rows, num_cols), layout=tmem_layout)

        barrier = ttgl.allocate_shared_memory(ttgl.int64, [1], ttgl.constexpr(mbarrier.MBarrierLayout()))
        mbarrier.init(barrier, count=1)

        smem.store(value)
        fence_async_shared()
        tcgen05_copy(smem, tmem)
        tcgen05_commit(barrier)
        mbarrier.wait(barrier, phase=0)
        tmem_alias: ttgl.constexpr = TensorMemoryLayout((128, 32), unpacked=False)
        tmem = tmem._reinterpret(ttgl.int8, (num_rows, num_cols), tmem_alias)
        value = tmem.load(blocked)
        ttgl.store(ttgl.set_auto_layout(out_ptrs, blocked), value)

    x = torch.randint(size=(smem_h, smem_w), low=-100, high=100, dtype=torch.int8).to(device)
    z_tri = torch.zeros(size=(num_rows, num_cols), dtype=torch.int8).to(device)
    kernel[(1, )](x, z_tri, smem_h, smem_w, num_rows, num_cols)

    num_rep_m = smem_h // 32

    for m in range(num_rep_m):
        col_offset = m * 4
        for i in range(4):
            # Copied values are duplicated across warps
            assert torch.equal(x[m * 32:(m + 1) * 32], z_tri[32 * i:32 * (i + 1), col_offset:(col_offset + 4)])


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_tmem_subslice_block_m_64():

    @gluon.jit
    def kernel(s_ptr, out_ptr):
        BLOCK_M: ttgl.constexpr = 64
        N: ttgl.constexpr = 128
        BLOCK_N: ttgl.constexpr = 64

        tmem_layout: ttgl.constexpr = TensorMemoryLayout((BLOCK_M, BLOCK_N), unpacked=True)
        s_tmem = allocate_tensor_memory(ttgl.float32, (BLOCK_M, N), layout=tmem_layout)
        o_tmem = allocate_tensor_memory(ttgl.float32, (BLOCK_M, N), layout=tmem_layout)

        layout: ttgl.constexpr = get_tmem_32x32b_reg_layout(BLOCK_M, BLOCK_N, (BLOCK_M, N), num_warps=4)

        offsets = ttgl.arange(0, BLOCK_M)[:, None] * N + ttgl.arange(0, N)[None, :]
        offsets = ttgl.set_auto_layout(offsets, layout)
        s = ttgl.load(s_ptr + offsets)

        s_tmem.store(s)
        o_tmem.store(s)

        p_tmem_layout: ttgl.constexpr = TensorMemoryLayout((BLOCK_M, BLOCK_N), unpacked=False)
        p_tmem = s_tmem.slice(0, N // 2)._reinterpret(ttgl.float16, [BLOCK_M, N], p_tmem_layout)
        p_tmem.store(ttgl.full((BLOCK_M, N), 0.0, dtype=ttgl.float16, layout=layout))

        d1_tmem_layout: ttgl.constexpr = TensorMemoryLayout((BLOCK_M, 1), unpacked=True)
        d1_layout: ttgl.constexpr = get_tmem_32x32b_reg_layout(BLOCK_M, 1, (BLOCK_M, 1), num_warps=4)

        m_tmem = s_tmem.slice(N // 4, 1)._reinterpret(ttgl.float32, [BLOCK_M, 1], d1_tmem_layout)
        m_tmem.store(ttgl.full((BLOCK_M, 1), 2.0, dtype=ttgl.float32, layout=d1_layout))
        l_tmem = s_tmem.slice(N // 4 + 1, 1)._reinterpret(ttgl.float32, [BLOCK_M, 1], d1_tmem_layout)
        l_tmem.store(ttgl.full((BLOCK_M, 1), 3.0, dtype=ttgl.float32, layout=d1_layout))
        a_tmem = s_tmem.slice(N // 4 + 2, 1)._reinterpret(ttgl.float32, [BLOCK_M, 1], d1_tmem_layout)
        a_tmem.store(ttgl.full((BLOCK_M, 1), 4.0, dtype=ttgl.float32, layout=d1_layout))

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
    out_ref[:, 32] = 2.0
    out_ref[:, 33] = 3.0
    out_ref[:, 34] = 4.0

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

        a_tmem_layout: ttgl.constexpr = TensorMemoryLayout((BLOCK_M, BLOCK_N), unpacked=False)
        acc_tmem_layout: ttgl.constexpr = TensorMemoryLayout((BLOCK_M, BLOCK_N), unpacked=True)
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
            unpacked=True,
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
