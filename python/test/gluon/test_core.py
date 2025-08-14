import torch
import pytest

from triton._internal_testing import is_cuda, is_ampere_or_newer, is_hip_cdna3, is_hip_cdna4, is_hopper_or_newer, is_hopper
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl
from triton.experimental.gluon.language.nvidia.ampere import async_copy, mbarrier
from triton.experimental.gluon.language.nvidia.hopper import tma, fence_async_shared
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
