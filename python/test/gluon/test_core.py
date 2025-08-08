import torch
import pytest

import triton
import triton.language as tl

from triton._internal_testing import is_cuda, is_ampere_or_newer, is_hopper_or_newer, is_hopper, is_hip, is_hip_cdna, is_hip_gfx12
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl
from triton.experimental.gluon.language.nvidia.ampere import async_copy, mbarrier
from triton.experimental.gluon.language.nvidia.hopper import tma
from triton.experimental.gluon.language.nvidia import hopper


@gluon.jit
def copy_kernel(Out, In, numel, XBLOCK: ttgl.constexpr, layout: ttgl.constexpr):
    xbase = ttgl.program_id(0) * XBLOCK
    xoffset = xbase + ttgl.arange(0, XBLOCK, layout=layout)
    xmask = xoffset < numel
    data = ttgl.load(In + xoffset, xmask)
    ttgl.store(Out + xoffset, data, xmask)


copy_kernel_tpw = [32] if is_cuda() else [64]


@pytest.mark.parametrize("layout", [
    ttgl.BlockedLayout(size_per_thread=[1], threads_per_warp=copy_kernel_tpw, warps_per_cta=[4], order=[0]),
    ttgl.BlockedLayout(size_per_thread=[2], threads_per_warp=copy_kernel_tpw, warps_per_cta=[4], order=[0]),
    ttgl.BlockedLayout(size_per_thread=[4], threads_per_warp=copy_kernel_tpw, warps_per_cta=[4], order=[0]),
    ttgl.BlockedLayout(size_per_thread=[8], threads_per_warp=copy_kernel_tpw, warps_per_cta=[4], order=[0]),
    ttgl.BlockedLayout(size_per_thread=[1], threads_per_warp=copy_kernel_tpw, warps_per_cta=[8], order=[0]),
    ttgl.BlockedLayout(size_per_thread=[2], threads_per_warp=copy_kernel_tpw, warps_per_cta=[8], order=[0]),
    ttgl.BlockedLayout(size_per_thread=[4], threads_per_warp=copy_kernel_tpw, warps_per_cta=[8], order=[0]),
    ttgl.BlockedLayout(size_per_thread=[8], threads_per_warp=copy_kernel_tpw, warps_per_cta=[8], order=[0]),
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


@pytest.mark.parametrize("M, N, K, rhs_scale, mxfp_type, normal_type, num_warps, mma, kpack",
                         [(32, 64, 128, rhs_scale, mxfp_type, normal_type, 4, 16, 1)
                          for rhs_scale in [True, False]
                          for mxfp_type in ["e2m1", "e4m3", "e5m2"]
                          for normal_type in ["e4m3", "e5m2"]])
def test_amd_mfma_scaled(M, N, K, rhs_scale, mxfp_type, normal_type, num_warps, mma, kpack):
    if is_cuda():
        pytest.skip()
    if is_hip():
        if not (is_hip_cdna() or is_hip_gfx12()):
            pytest.skip("scaled_dot only implemented for HIP CDNA and gfx12")

    device = 'cuda'

    @triton.jit
    def triton_kernel(a_base, stride_a0, stride_a1, a_scale, b_base, stride_b0, stride_b1, b_scale, out,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, type_a: tl.constexpr,
                      type_b: tl.constexpr):
        DIV_FACTOR_A: tl.constexpr = 2 if type_a == "e2m1" else 1
        DIV_FACTOR_B: tl.constexpr = 2 if type_b == "e2m1" else 1
        PACKED_BLOCK_K_A: tl.constexpr = BLOCK_K // DIV_FACTOR_A
        PACKED_BLOCK_K_B: tl.constexpr = BLOCK_K // DIV_FACTOR_B
        a_ptr = a_base + tl.arange(0, BLOCK_M)[:, None] * stride_a0 + \
                tl.arange(0, PACKED_BLOCK_K_A)[None, :] * stride_a1
        b_ptr = b_base + tl.arange(0, PACKED_BLOCK_K_B)[:, None] * stride_b0 + \
                tl.arange(0, BLOCK_N)[None, :] * stride_b1

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
    def gluon_kernel(a_base, stride_a0, stride_a1, a_scale, b_base, stride_b0, stride_b1, b_scale, out,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, type_a: tl.constexpr,
                     type_b: tl.constexpr):
        DIV_FACTOR_A: tl.constexpr = 2 if type_a == "e2m1" else 1
        DIV_FACTOR_B: tl.constexpr = 2 if type_b == "e2m1" else 1
        PACKED_BLOCK_K_A: tl.constexpr = BLOCK_K // DIV_FACTOR_A
        PACKED_BLOCK_K_B: tl.constexpr = BLOCK_K // DIV_FACTOR_B

        a_layout: ttgl.constexpr = ttgl.DistributedLinearLayout(
            reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8]], lane_bases=[[0, 16], [0, 32], [0, 64], [1, 0], [2, 0], [4, 0]],
            warp_bases=[[8, 0], [16, 0]], block_bases=[], shape=[32, 128])
        a_scale_layout: ttgl.constexpr = ttgl.DistributedLinearLayout(
            reg_bases=[[16, 0]], lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2]],
            warp_bases=[[0, 0], [0, 0]], block_bases=[], shape=[32, 4])
        b_layout: ttgl.constexpr = ttgl.DistributedLinearLayout(
            reg_bases=[[1, 0], [2, 0], [4, 0], [8, 0]], lane_bases=[[16, 0], [32, 0], [0, 1], [0, 2], [0, 4], [0, 8]],
            warp_bases=[[0, 16], [0, 32]], block_bases=[], shape=[64, 64])
        b_scale_layout: ttgl.constexpr = ttgl.DistributedLinearLayout(
            reg_bases=[], lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2]], warp_bases=[[16, 0], [32, 0]],
            block_bases=[], shape=[64, 4])
        mma: ttgl.constexpr = ttgl.amd.AMDMFMALayout(version=4, warps_per_cta=[1, 4], tiles_per_warp=[1, 1],
                                                     instr_shape=[16, 16], transposed=True)

        zero = ttgl.zeros([BLOCK_M, BLOCK_N], dtype=ttgl.float32, layout=mma)

        a_offsets = ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, a_layout))[:, None] * stride_a0 + \
                    ttgl.arange(0, PACKED_BLOCK_K_A, layout=ttgl.SliceLayout(0, a_layout))[None, :] * stride_a1
        a = ttgl.amd.cdna4.buffer_load(a_base, a_offsets)
        a = ttgl.convert_layout(a, ttgl.DotOperandLayout(operand_index=0, parent=mma, k_width=16))

        b_offsets = ttgl.arange(0, PACKED_BLOCK_K_B, layout=ttgl.SliceLayout(1, b_layout))[:, None] * stride_b0 + \
                    ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, b_layout))[None, :] * stride_b1
        b = ttgl.amd.cdna4.buffer_load(b_base, b_offsets)
        b = ttgl.convert_layout(b, ttgl.DotOperandLayout(operand_index=1, parent=mma, k_width=16))

        SCALE_BLOCK_K: tl.constexpr = BLOCK_K // 32
        if a_scale is not None:
            a_offsets = ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, a_scale_layout))[:, None] * SCALE_BLOCK_K + \
                        ttgl.arange(0, SCALE_BLOCK_K, layout=ttgl.SliceLayout(0, a_scale_layout))[None, :]
            a_scale = ttgl.amd.cdna4.buffer_load(a_scale, a_offsets)

            b_scale = ttgl.full([BLOCK_M, SCALE_BLOCK_K], 127, dtype=ttgl.int8, layout=b_scale_layout)
        else:
            assert b_scale is not None
            a_scale = ttgl.full([BLOCK_M, SCALE_BLOCK_K], 127, dtype=ttgl.int8, layout=a_scale_layout)

            b_scale_offsets = ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(1, b_scale_layout))[:, None] * SCALE_BLOCK_K + \
                              ttgl.arange(0, SCALE_BLOCK_K, layout=ttgl.SliceLayout(0, b_scale_layout))[None, :]
            b_scale = ttgl.amd.cdna4.buffer_load(b_scale, b_scale_offsets)

        c = ttgl.amd.cdna4.mfma_scaled(a, a_scale, type_a, b, b_scale, type_b, zero, layout=mma)
        c = c.to(out.dtype.element_ty)

        out_offsets = ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, mma))[:, None] * BLOCK_N + \
                      ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, mma))[None, :]
        ttgl.amd.cdna4.buffer_store(c, out, out_offsets)

    comp_dtype = torch.float16 if normal_type == "fp16" else torch.bfloat16
    # The max exponent we use to initialize data in the x/y and associated scale tensor to avoid
    # overflow when scaling.
    comp_dtype_max_exp = 6 if normal_type == "fp16" else 15

    torch.manual_seed(0)

    def make_arg(shape, ty, col_major=False):
        if col_major:
            shape = shape[:-2] + (shape[-1], shape[-2])
        if ty == "bf16" or ty == "fp16":
            ret = torch.randn(shape, dtype=comp_dtype, device=device)
            # Clamp to avoid relative error issues
            ret.clamp_(-2**comp_dtype_max_exp, 2**comp_dtype_max_exp - 1)
        else:
            # On other chips, the A/B operands are upcasted to fp16/bf16
            # before matmul, which has larger range to avoid overflow.
            # On CDNA4, we use the V_MFMA_*_F8F6F4 instructions to
            # directly calculate matmul on F8F6F4 data. So we need
            # to narrow down the range of input to avoid overflow.
            ret = torch.randint(20, 40, shape, dtype=torch.uint8, device=device)

        if col_major:
            ret = ret.mT
        return ret

    type_a = normal_type if rhs_scale else mxfp_type
    type_b = mxfp_type if rhs_scale else normal_type

    DIV_FACTOR_A = 2 if type_a == "e2m1" else 1
    DIV_FACTOR_B = 2 if type_b == "e2m1" else 1
    x = make_arg((M, K // DIV_FACTOR_A), type_a, col_major=False)
    y = make_arg((K // DIV_FACTOR_B, N), type_b, col_major=True)

    min_scale, max_scale = (0, 142) if comp_dtype == torch.bfloat16 else (124, 131)
    scale_x = torch.randint(min_scale, max_scale + 1, (M, K // 32), dtype=torch.uint8, device=device)
    scale_y = torch.randint(min_scale, max_scale + 1, (N, K // 32), dtype=torch.uint8, device=device)
    if rhs_scale:
        scale_x = None
    else:
        scale_y = None

    def make_finite(x, dtype):
        # e5m2 has too many non-finite values when sampled uniformly (1 / 32) and
        # Fp8E5M2_to_Bf16 doesn't preserve NaNs (fixme)
        if dtype not in ("e5m2", "e4m3"):
            return x
        if dtype == "e5m2" and comp_dtype == torch.float16:
            x = x & 0xB
        mask = 0x7C if dtype == "e5m2" else 0x7F
        finite = torch.arange(x.numel(), device=device, dtype=torch.uint8).reshape_as(x) % mask
        x_finite = torch.where(x & mask == mask, finite | (0x80 & x), x)
        x.copy_(x_finite)
        return x

    x = make_finite(x, type_a)
    y = make_finite(y, type_b)

    kernel_kwargs = {"num_warps": num_warps}
    kernel_kwargs["kpack"] = kpack
    kernel_kwargs["matrix_instr_nonkdim"] = mma

    z = torch.zeros((M, N), dtype=comp_dtype, device=device)
    gluon_kernel[(1, )](x, *x.stride(), scale_x, y, *y.stride(), scale_y, z, M, N, K, type_a, type_b, **kernel_kwargs)

    z_ref = torch.zeros((M, N), dtype=comp_dtype, device=device)
    triton_kernel[(1, )](x, *x.stride(), scale_x, y, *y.stride(), scale_y, z_ref, M, N, K, type_a, type_b,
                         **kernel_kwargs)

    torch.testing.assert_close(z, z_ref, rtol=1e-5, atol=1e-5)
