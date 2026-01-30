import torch
import math
import pytest
import re
from itertools import product

import triton
import triton.language as tl

from triton._internal_testing import (
    is_ampere_or_newer,
    is_blackwell,
    is_blackwell_ultra,
    is_hip_rdna,
    is_hip_rdna3,
    is_hip_rdna4,
    is_hip_cdna,
    is_hip_cdna3,
    is_hip_cdna4,
    is_hopper_or_newer,
    is_hopper,
)
from triton.compiler import max_shared_mem
from triton.tools.mxfp import MXFP4Tensor, MXScaleTensor
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
    get_tmem_reg_layout,
    tcgen05_mma_barrier_count,
    tcgen05_mma,
    tcgen05_mma_scaled,
    tcgen05_commit,
    tcgen05_copy,
    float2,
)
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

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


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Requires Hopper")
def test_copy_kernel_multi_cta():
    XBLOCK = 2048
    layout = ttgl.BlockedLayout(size_per_thread=[8], threads_per_warp=[THREADS_PER_WARP], warps_per_cta=[8], order=[0],
                                cga_layout=[[1]])

    inp = torch.randn(XBLOCK * 4 - 7, device="cuda")
    out = torch.empty_like(inp)
    copy_kernel[(4, )](out, inp, inp.numel(), XBLOCK, layout, num_warps=layout.warps_per_cta[0], num_ctas=2)
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
def tma_round_f32_to_tf32_kernel(in_desc, out_desc):
    smem = ttgl.allocate_shared_memory(in_desc.dtype, in_desc.block_shape, in_desc.layout)
    bar = mbarrier.allocate_mbarrier()
    mbarrier.init(bar, count=1)
    mbarrier.expect(bar, in_desc.nbytes_per_cta)
    tma.async_copy_global_to_shared(in_desc, [0, 0], bar, smem)
    mbarrier.wait(bar, phase=0, deps=[smem])
    mbarrier.invalidate(bar)
    tma.async_copy_shared_to_global(out_desc, [0, 0], smem)
    tma.store_wait(0)
    smem._keep_alive()


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Requires Hopper")
def test_gluon_tma_round_f32_to_tf32():

    def round_to_tf32(x: torch.Tensor) -> torch.Tensor:
        bits = x.view(torch.int32)
        bits_i64 = bits.to(torch.int64) & 0xFFFFFFFF
        exp_mask = 0x7F800000
        is_special = (bits_i64 & exp_mask) == exp_mask
        round_bias = ((bits_i64 >> 13) & 1) + 0x00000FFF
        rounded = (bits_i64 + round_bias) & 0xFFFFE000
        out_bits = torch.where(is_special, bits_i64, rounded)
        return (out_bits & 0xFFFFFFFF).to(torch.int32).view(torch.float32)

    torch.manual_seed(17)
    inp = torch.randn((16, 16), device="cuda", dtype=torch.float32)
    out = torch.empty_like(inp)

    layout = ttgl.NVMMASharedLayout(
        swizzle_byte_width=32,
        element_bitwidth=32,
        rank=2,
        transposed=False,
        fp4_padded=False,
    )
    in_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(inp, [16, 16], layout, round_f32_to_tf32=True)
    out_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(out, [16, 16], layout)

    tma_round_f32_to_tf32_kernel[(1, )](in_desc, out_desc)

    expected = round_to_tf32(inp)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@gluon.jit
def tma_multicast_copy_kernel(in_desc, out_desc):
    smem = ttgl.allocate_shared_memory(in_desc.dtype, in_desc.block_shape, in_desc.layout)

    bar = mbarrier.allocate_mbarrier()
    mbarrier.init(bar, count=1)
    # Need to synchronise all the CTAs after the mbarrier initialisation
    # so that they all see it before tma.async_copy_global_to_shared(multicast=True)
    mbarrier.sync_cluster_init()

    mbarrier.expect(bar, in_desc.nbytes_per_cta)
    tma.async_copy_global_to_shared(in_desc, [0, 0], bar, smem, multicast=True)
    mbarrier.wait(bar, phase=0, deps=[smem])

    tma.async_copy_shared_to_global(out_desc, [0, 0], smem)
    tma.store_wait(0)

    mbarrier.invalidate(bar)
    smem._keep_alive()


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Requires Hopper")
@pytest.mark.parametrize("ctas_per_cga", [[2, 1], [1, 4], [4, 4]])
def test_tma_multicast_copy(ctas_per_cga):
    from triton._C.libtriton.gluon_ir import make_cga_layout
    cga_split_num = [min(ctas_per_cga[0], 2), min(ctas_per_cga[1], 2)]
    cga_layout = make_cga_layout(ctas_per_cga, cga_split_num, [1, 0])

    BLOCK_M, BLOCK_N = 16, 16
    BLOCK_M *= cga_split_num[0]
    BLOCK_N *= cga_split_num[1]

    inp = torch.randn((BLOCK_M, BLOCK_N), dtype=torch.float16, device="cuda")
    out = torch.empty_like(inp)

    layout = ttgl.NVMMASharedLayout.get_default_for(
        [BLOCK_M, BLOCK_N],
        ttgl.float16,
        cga_layout=cga_layout,
    )

    in_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(inp, [BLOCK_M, BLOCK_N], layout)
    out_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(out, [BLOCK_M, BLOCK_N], layout)
    num_ctas = ctas_per_cga[0] * ctas_per_cga[1]
    compiled = tma_multicast_copy_kernel[(1, )](
        in_desc,
        out_desc,
        num_warps=4,
        num_ctas=num_ctas,
    )
    expect_multicast = any(ctas_per_cga[i] > cga_split_num[i] for i in range(len(ctas_per_cga)))
    assert (".multicast::cluster" in compiled.asm["ptx"]) == expect_multicast
    torch.testing.assert_close(out, inp, atol=0, rtol=0)


@gluon.jit
def tcgen05_mma_multicast_commit_kernel(a_desc, b_desc, out_ptrs, BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr,
                                        acc_tmem_layout: ttgl.constexpr, blocked_c: ttgl.constexpr):
    smem_a = ttgl.allocate_shared_memory(a_desc.dtype, a_desc.block_shape, a_desc.layout)
    smem_b = ttgl.allocate_shared_memory(b_desc.dtype, b_desc.block_shape, b_desc.layout)

    tma_bar = mbarrier.allocate_mbarrier(two_ctas=acc_tmem_layout.two_ctas)
    mbarrier.init(tma_bar, count=1)
    mma_bar = mbarrier.allocate_mbarrier()
    mbarrier.init(mma_bar, count=tcgen05_mma_barrier_count([smem_a, smem_b], True))

    # Need to synchronise all the CTAs after the mbarrier initialisation
    # so that they all see it before tma.async_copy_global_to_shared(multicast=True)
    mbarrier.sync_cluster_init()

    mbarrier.expect(tma_bar, a_desc.nbytes_per_cta + b_desc.nbytes_per_cta)
    tma.async_copy_global_to_shared(a_desc, [0, 0], tma_bar, smem_a, multicast=True)
    tma.async_copy_global_to_shared(b_desc, [0, 0], tma_bar, smem_b, multicast=True)
    mbarrier.wait(tma_bar, phase=0, deps=[smem_a, smem_b])
    mbarrier.invalidate(tma_bar)

    acc_tmem = allocate_tensor_memory(ttgl.float32, [BLOCK_M, BLOCK_N], acc_tmem_layout)
    # If it's not in a loop we don't striclty need multicast=True, but we add it to exercise the path in the test
    tcgen05_mma(smem_a, smem_b, acc_tmem, use_acc=False, multicast=True, mbarriers=[mma_bar])
    mbarrier.wait(mma_bar, phase=0, deps=[smem_a, smem_b])
    mbarrier.invalidate(mma_bar)

    tmem_reg_layout: ttgl.constexpr = get_tmem_reg_layout(
        ttgl.float32,
        (BLOCK_M, BLOCK_N),
        acc_tmem_layout,
        num_warps=ttgl.num_warps(),
        cga_layout=blocked_c.cga_layout,
    )
    out = acc_tmem.load(tmem_reg_layout)
    out = ttgl.convert_layout(out, blocked_c)

    out_offs_m = ttgl.arange(0, BLOCK_M)[:, None]
    out_offs_n = ttgl.arange(0, BLOCK_N)[None, :]
    out_ptrs = out_ptrs + out_offs_m * BLOCK_N + out_offs_n
    ttgl.store(out_ptrs, out)


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("ctas_per_cga", [[2, 1], [2, 4], [4, 4]])
@pytest.mark.parametrize("two_ctas", [True, False] if is_blackwell() else [False])
def test_tcgen05_mma_multicast_commit(ctas_per_cga, two_ctas):

    if two_ctas:
        ctas_per_cga_b = [ctas_per_cga[0] // 2, 2 * ctas_per_cga[1]]
    else:
        ctas_per_cga_b = ctas_per_cga
    BLOCK_M = 128 * ctas_per_cga[0]
    BLOCK_N = 64 * ctas_per_cga_b[1]
    BLOCK_K = 32

    # multicast into tcgen05_mma
    cta_split_a = [ctas_per_cga[0], 1]
    cta_split_b = [1, ctas_per_cga_b[1]]
    cta_order = [1, 0]

    from triton._C.libtriton.gluon_ir import make_cga_layout
    if two_ctas:

        def make_2cta_cga_layout(ctas_per_cga, cta_split, cta_order, two_cta_dim):
            ctas_per_cga = list(ctas_per_cga)
            cta_split = list(cta_split)
            assert cta_split[two_cta_dim] > 1
            cta_split[two_cta_dim] //= 2
            ctas_per_cga[two_cta_dim] //= 2
            aux_cga_layout = make_cga_layout(ctas_per_cga, cta_split, cta_order)
            assert two_cta_dim in (0, 1)
            basis = [0, 0]
            basis[two_cta_dim] = 1
            for b in aux_cga_layout:
                b[two_cta_dim] *= 2
            cga_layout = [basis] + aux_cga_layout
            return cga_layout

        cga_layout_a = make_2cta_cga_layout(ctas_per_cga, cta_split_a, cta_order, 0)
        cga_layout_b = make_2cta_cga_layout(ctas_per_cga_b, cta_split_b, cta_order, 1)
        cga_layout_c = make_2cta_cga_layout(ctas_per_cga, ctas_per_cga, cta_order, 0)
    else:
        cga_layout_a = make_cga_layout(ctas_per_cga, cta_split_a, cta_order)
        cga_layout_b = make_cga_layout(ctas_per_cga_b, cta_split_b, cta_order)
        cga_layout_c = make_cga_layout(ctas_per_cga, ctas_per_cga, cta_order)

    shared_layout_a = ttgl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_K], ttgl.float16, cga_layout=cga_layout_a)
    shared_layout_b = ttgl.NVMMASharedLayout.get_default_for([BLOCK_K, BLOCK_N], ttgl.float16, cga_layout=cga_layout_b)

    a = torch.randn((BLOCK_M, BLOCK_K), dtype=torch.float16, device="cuda")
    b = torch.randn((BLOCK_K, BLOCK_N), dtype=torch.float16, device="cuda")
    out = torch.empty((BLOCK_M, BLOCK_N), dtype=torch.float32, device="cuda")

    a_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(a, [BLOCK_M, BLOCK_K], shared_layout_a)
    b_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(b, [BLOCK_K, BLOCK_N], shared_layout_b)

    tmem_shape = (128, BLOCK_N // ctas_per_cga[1])
    acc_tmem_layout = TensorMemoryLayout(block=tmem_shape, col_stride=1, two_ctas=two_ctas,
                                         cta_split_num=tuple(ctas_per_cga))
    blocked_c = ttgl.BlockedLayout([1, 2], [ctas_per_cga[1], 32 // ctas_per_cga[1]], [4, 1], [1, 0],
                                   cga_layout=cga_layout_c)

    compiled = tcgen05_mma_multicast_commit_kernel[(1, )](
        a_desc,
        b_desc,
        out,
        BLOCK_M,
        BLOCK_N,
        acc_tmem_layout,
        blocked_c,
        num_warps=4,
        num_ctas=ctas_per_cga[0] * ctas_per_cga[1],
    )

    assert "tcgen05.commit.cta_group::" + ("2" if two_ctas else "1") in compiled.asm["ptx"]
    # For [2, 1] and two_ctas we don't multicast as there are not enough tiles
    # but we do a commit.multicast::cluster so let's grep that one instead
    assert ("multicast::cluster" in compiled.asm["ptx"])
    torch.testing.assert_close(out, torch.matmul(a.to(out.dtype), b.to(out.dtype)))


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


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Requires Hopper")
def test_device_tma_load():

    @gluon.jit
    def tma_device_load_kernel(input_ptr, output_ptr, XBLOCK: ttgl.constexpr, smem_layout: ttgl.constexpr):
        input_desc = tma.make_tensor_descriptor(
            input_ptr,
            shape=[XBLOCK, XBLOCK],
            strides=[XBLOCK, 1],
            block_shape=[XBLOCK, XBLOCK],
            layout=smem_layout,
        )

        smem = ttgl.allocate_shared_memory(ttgl.float16, [XBLOCK, XBLOCK], smem_layout)
        bar = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
        mbarrier.init(bar, count=1)

        mbarrier.expect(bar, input_desc.nbytes_per_cta)
        tma.async_copy_global_to_shared(input_desc, [0, 0], bar, smem)
        mbarrier.wait(bar, 0)
        mbarrier.invalidate(bar)

        block_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 2], [4, 8], [4, 1], [1, 0])
        xindex = ttgl.arange(0, XBLOCK, ttgl.SliceLayout(1, block_layout))[:, None]
        yindex = ttgl.arange(0, XBLOCK, ttgl.SliceLayout(0, block_layout))[None, :]
        val = smem.load(block_layout)
        ttgl.store(output_ptr + yindex + xindex * XBLOCK, val)

    XBLOCK = 16
    input = torch.zeros((XBLOCK, XBLOCK), device="cuda", dtype=torch.float16)
    output = torch.ones_like(input)
    smem_layout = ttgl.NVMMASharedLayout(
        swizzle_byte_width=32,
        element_bitwidth=16,
        rank=2,
        transposed=False,
        fp4_padded=False,
    )

    def alloc_fn(size: int, alignment: int, stream: int):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    tma_device_load_kernel[(1, )](input, output, XBLOCK, smem_layout)
    torch.testing.assert_close(input, output)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Requires Hopper")
def test_device_tma_store():

    @gluon.jit
    def tma_device_store_kernel(out_ptr, XBLOCK: ttgl.constexpr, smem_layout: ttgl.constexpr):
        layout: ttgl.constexpr = ttgl.BlockedLayout([1, 2], [4, 8], [4, 1], [1, 0])
        value = ttgl.full([XBLOCK, XBLOCK], 0, ttgl.float16, layout)
        alloc = ttgl.allocate_shared_memory(ttgl.float16, [XBLOCK, XBLOCK], smem_layout, value)
        out_desc = tma.make_tensor_descriptor(
            out_ptr,
            shape=[XBLOCK, XBLOCK],
            strides=[XBLOCK, 1],
            block_shape=[XBLOCK, XBLOCK],
            layout=smem_layout,
        )
        tma.async_copy_shared_to_global(out_desc, [0, 0], alloc)
        tma.store_wait(0)
        alloc._keep_alive()

    XBLOCK = 16
    out = torch.ones((XBLOCK, XBLOCK), dtype=torch.float16, device="cuda")
    smem_layout = ttgl.NVMMASharedLayout(
        swizzle_byte_width=32,
        element_bitwidth=16,
        rank=2,
        transposed=False,
        fp4_padded=False,
    )

    def alloc_fn(size: int, alignment: int, stream: int):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    tma_device_store_kernel[(1, )](out, XBLOCK, smem_layout)
    torch.testing.assert_close(out, torch.zeros_like(out))


@gluon.jit
def mma_kernel(a, b, out, M: ttgl.constexpr, N: ttgl.constexpr, K: ttgl.constexpr, block_layout_a: ttgl.constexpr,
               block_layout_b: ttgl.constexpr, cga_layout_c: ttgl.constexpr, acc_layout: ttgl.constexpr,
               shared_layout_a: ttgl.constexpr, shared_layout_b: ttgl.constexpr, acc_dtype: ttgl.constexpr,
               ASYNC: ttgl.constexpr, USE_TCGEN05: ttgl.constexpr):
    a_offs_m = ttgl.arange(0, M)[:, None]
    a_offs_k = ttgl.arange(0, K)[None, :]
    b_offs_k = ttgl.arange(0, K)[:, None]
    b_offs_n = ttgl.arange(0, N)[None, :]

    operand_dtype = a.dtype.element_ty
    a_ptrs = a + a_offs_m * K + a_offs_k
    b_ptrs = b + b_offs_k * N + b_offs_n
    a_tile = ttgl.load(ttgl.set_auto_layout(a_ptrs, block_layout_a))
    b_tile = ttgl.load(ttgl.set_auto_layout(b_ptrs, block_layout_b))

    smem_a = ttgl.allocate_shared_memory(operand_dtype, [M, K], shared_layout_a, a_tile)
    smem_b = ttgl.allocate_shared_memory(operand_dtype, [K, N], shared_layout_b, b_tile)

    if USE_TCGEN05:
        two_ctas: ttgl.constexpr = acc_layout.two_ctas
        fence_async_shared(cluster=two_ctas)
        mma_barrier = mbarrier.allocate_mbarrier()
        mbarrier.init(mma_barrier, count=1)
        # Need to synchronise all the CTAs after the mbarrier initialisation
        # so that they all see it
        if two_ctas:
            mbarrier.sync_cluster_init()

        acc_tmem = allocate_tensor_memory(acc_dtype, [M, N], acc_layout)

        tcgen05_mma(smem_a, smem_b, acc_tmem, use_acc=False, mbarriers=[mma_barrier])
        mbarrier.wait(mma_barrier, phase=0, deps=[smem_a, smem_b])
        mbarrier.invalidate(mma_barrier)

        tmem_reg_layout: ttgl.constexpr = get_tmem_reg_layout(
            acc_dtype,
            (M, N),
            acc_layout,
            num_warps=ttgl.num_warps(),
            cga_layout=cga_layout_c,
        )
        acc = acc_tmem.load(tmem_reg_layout)
    else:
        fence_async_shared()
        acc = ttgl.zeros([M, N], dtype=acc_dtype, layout=acc_layout)
        acc = hopper.warpgroup_mma(smem_a, smem_b, acc, is_async=ASYNC)

        if ASYNC:
            acc = hopper.warpgroup_mma_wait(num_outstanding=0, deps=[acc])

    out_offs_m = ttgl.arange(0, M)[:, None]
    out_offs_n = ttgl.arange(0, N)[None, :]
    out_ptrs = out + out_offs_m * N + out_offs_n
    ttgl.store(out_ptrs, acc)


@pytest.mark.skipif(not is_hopper(), reason="Requires Hopper")
@pytest.mark.parametrize("ASYNC", [True, False])
def test_warpgroup_mma(ASYNC):
    torch.manual_seed(0)
    M, N, K = 64, 32, 32
    warps = [4, 1]
    block_layout = ttgl.BlockedLayout([1, 1], [1, THREADS_PER_WARP], warps_per_cta=warps, order=[1, 0])
    acc_layout = ttgl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=warps, instr_shape=[16, 32, 16])
    shared_layout_a = ttgl.NVMMASharedLayout.get_default_for([M, K], ttgl.float16)
    shared_layout_b = ttgl.NVMMASharedLayout.get_default_for([K, N], ttgl.float16)
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    out = torch.zeros((M, N), device="cuda", dtype=torch.float16)
    mma_kernel[(1, )](
        a,
        b,
        out,
        M,
        N,
        K,
        block_layout,
        block_layout,
        block_layout,
        acc_layout,
        shared_layout_a,
        shared_layout_b,
        ttgl.float16,
        ASYNC,
        False,
        num_warps=warps[0] * warps[1],
    )

    ref = torch.matmul(a, b)

    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-1)


@gluon.jit
def tma_mma_shared_inputs_kernel(a_desc, b_desc, out_ptr, BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr,
                                 BLOCK_K: ttgl.constexpr, NUM_K_TILES: ttgl.constexpr, block_layout_c: ttgl.constexpr,
                                 acc_layout: ttgl.constexpr, acc_tmem_layout: ttgl.constexpr,
                                 use_tcgen05: ttgl.constexpr, multicast: ttgl.constexpr):
    smem_a = ttgl.allocate_shared_memory(a_desc.dtype, a_desc.block_shape, a_desc.layout)
    smem_b = ttgl.allocate_shared_memory(b_desc.dtype, b_desc.block_shape, b_desc.layout)

    two_ctas: ttgl.constexpr = isinstance(acc_tmem_layout, TensorMemoryLayout) and acc_tmem_layout.two_ctas

    tma_bar = mbarrier.allocate_mbarrier(two_ctas=two_ctas)
    mbarrier.init(tma_bar, count=1)
    phase_tma = 0

    if use_tcgen05:
        mma_bar = mbarrier.allocate_mbarrier()
        phase_mma = 0
        mbarrier.init(mma_bar, count=tcgen05_mma_barrier_count([smem_a, smem_b], multicast))
        acc_tmem = allocate_tensor_memory(
            element_ty=ttgl.float32,
            shape=[BLOCK_M, BLOCK_N],
            layout=acc_tmem_layout,
        )
    else:
        acc = ttgl.zeros([BLOCK_M, BLOCK_N], dtype=ttgl.float32, layout=acc_layout)

    # Need to synchronise all the CTAs after the mbarrier initialisation before we do
    # cross-CTA ops
    if (multicast and ttgl.num_ctas() > 1) or two_ctas:
        mbarrier.sync_cluster_init()

    for k in range(NUM_K_TILES):
        mbarrier.expect(tma_bar, a_desc.nbytes_per_cta + b_desc.nbytes_per_cta)
        tma.async_copy_global_to_shared(a_desc, [0, k * BLOCK_K], tma_bar, smem_a, multicast=multicast)
        tma.async_copy_global_to_shared(b_desc, [k * BLOCK_K, 0], tma_bar, smem_b, multicast=multicast)
        mbarrier.wait(tma_bar, phase=phase_tma, deps=[smem_a, smem_b])
        phase_tma ^= 1

        if use_tcgen05:
            tcgen05_mma(smem_a, smem_b, acc_tmem, use_acc=(k != 0), multicast=multicast, mbarriers=[mma_bar])
            mbarrier.wait(mma_bar, phase=phase_mma, deps=[smem_a, smem_b])
            phase_mma ^= 1
        else:
            acc = hopper.warpgroup_mma(smem_a, smem_b, acc, is_async=False)
            if multicast:
                # multicast into wgmma doesn't make much sense as you need to synchronise all
                # CTAs after the wgmma, as it doesn't provide a finer synchronization mechanism.
                ttgl.barrier(cluster=True)

    mbarrier.invalidate(tma_bar)

    if use_tcgen05:
        mbarrier.invalidate(mma_bar)
        reg_layout: ttgl.constexpr = get_tmem_reg_layout(
            ttgl.float32,
            (BLOCK_M, BLOCK_N),
            acc_tmem_layout,
            num_warps=ttgl.num_warps(),
            cga_layout=block_layout_c.cga_layout,
        )
        acc = acc_tmem.load(reg_layout)

    acc = ttgl.convert_layout(acc, block_layout_c)
    offs_m = ttgl.arange(0, BLOCK_M)[:, None]
    offs_n = ttgl.arange(0, BLOCK_N)[None, :]
    ttgl.store(out_ptr + offs_m * BLOCK_N + offs_n, acc)


@pytest.mark.skipif(not (is_hopper() or is_blackwell()), reason="Requires Hopper or Blackwell")
@pytest.mark.parametrize("warps", ([8, 1], [4, 2], [4, 1]))
@pytest.mark.parametrize("reps", ([1, 1, 1], [2, 2, 2], [1, 4, 2]))
@pytest.mark.parametrize("ctas_per_cga", [[1, 1], [2, 1], [4, 4]])
@pytest.mark.parametrize("two_ctas", [False, True] if is_blackwell() else [False])
@pytest.mark.parametrize("multicast", [False, True])
def test_tma_mma_shared_inputs(warps, reps, ctas_per_cga, two_ctas, multicast):
    bitwidth = 16
    acc_dtype = torch.float32

    if ctas_per_cga[0] == 1 and two_ctas:
        pytest.skip("Need at least 2 CTAs along M for 2CTA mode")

    cta_order = [1, 0]

    if two_ctas:
        assert ctas_per_cga[0] >= 2, "Need at least 2 CTAs along M for 2CTA mode"
        ctas_per_cga_b = [ctas_per_cga[0] // 2, 2 * ctas_per_cga[1]]
    else:
        ctas_per_cga_b = ctas_per_cga
    cta_split_a = [ctas_per_cga[0], 1]
    cta_split_b = [1, ctas_per_cga_b[1]]

    # M = 128 for blackkwell
    instr_shape = [32 if is_blackwell() else 16, 32, 256 // bitwidth]
    NUM_K_TILES = 4
    BLOCK_M = instr_shape[0] * warps[0] * ctas_per_cga[0] * reps[0]
    BLOCK_N = instr_shape[1] * warps[1] * ctas_per_cga_b[1] * reps[1]
    if is_blackwell() and BLOCK_N >= 256 * ctas_per_cga[1]:
        # tcgen05 doesn't support reps along N
        BLOCK_N = 256 * ctas_per_cga[1]
    BLOCK_K = instr_shape[2] * reps[2]
    K = (256 // bitwidth) * NUM_K_TILES

    from triton._C.libtriton.gluon_ir import make_cga_layout
    if two_ctas:

        def make_2cta_cga_layout(ctas_per_cga, cta_split, cta_order, two_cta_dim):
            ctas_per_cga = list(ctas_per_cga)
            cta_split = list(cta_split)
            assert cta_split[two_cta_dim] > 1
            cta_split[two_cta_dim] //= 2
            ctas_per_cga[two_cta_dim] //= 2
            aux_cga_layout = make_cga_layout(ctas_per_cga, cta_split, cta_order)
            assert two_cta_dim in (0, 1)
            basis = [0, 0]
            basis[two_cta_dim] = 1
            for b in aux_cga_layout:
                b[two_cta_dim] *= 2
            cga_layout = [basis] + aux_cga_layout
            return cga_layout

        cga_layout_a = make_2cta_cga_layout(ctas_per_cga, cta_split_a, cta_order, 0)
        cga_layout_b = make_2cta_cga_layout(ctas_per_cga_b, cta_split_b, cta_order, 1)
        cga_layout_c = make_2cta_cga_layout(ctas_per_cga, ctas_per_cga, cta_order, 0)
    else:
        cga_layout_a = make_cga_layout(ctas_per_cga, cta_split_a, cta_order)
        cga_layout_b = make_cga_layout(ctas_per_cga_b, cta_split_b, cta_order)
        cga_layout_c = make_cga_layout(ctas_per_cga, ctas_per_cga, cta_order)

    block_layout_c = ttgl.BlockedLayout([1, 8], [1, THREADS_PER_WARP], warps_per_cta=warps, order=[1, 0],
                                        cga_layout=cga_layout_c)

    acc_layout = ttgl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=warps, instr_shape=instr_shape,
                                             cga_layout=cga_layout_c)

    tmem_shape = (min(BLOCK_M // ctas_per_cga[0], 128), BLOCK_N // ctas_per_cga[1])
    acc_tmem_layout = TensorMemoryLayout(
        block=tmem_shape,
        col_stride=1,
        cta_split_num=tuple(ctas_per_cga),
        two_ctas=two_ctas,
    )

    def cast(x, dtype):
        if dtype != torch.float32:
            return x.to(dtype)
        # For b16 and fp32 (in both hopper and blackwell it seems)
        # Element-wise multiplication of matrix A and B is performed with specified precision.
        # wgmma.mma_async operation involving type .tf32 will truncate lower 13 bits of the 32-bit
        # input data before multiplication is issued
        x = x.view(torch.int32)
        x = x & ~((1 << 13) - 1)
        return x.view(dtype)

    torch_dtype = torch.float16
    device = triton.runtime.driver.active.get_current_device()
    a = cast(torch.randn((BLOCK_M, K), device=device, dtype=torch.float32), torch_dtype)
    # We transpose b in the kernel
    b = cast(torch.randn((K, BLOCK_N), device=device, dtype=torch.float32), torch_dtype)
    out = torch.empty((BLOCK_M, BLOCK_N), device=device, dtype=acc_dtype)

    gluon_dtype = ttgl.float16
    shared_layout_a = ttgl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_K], gluon_dtype, cga_layout=cga_layout_a)
    shared_layout_b = ttgl.NVMMASharedLayout.get_default_for([BLOCK_K, BLOCK_N], gluon_dtype, cga_layout=cga_layout_b)
    assert shared_layout_a.swizzle_byte_width != 0
    assert shared_layout_b.swizzle_byte_width != 0
    a_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(a, [BLOCK_M, BLOCK_K], shared_layout_a)
    b_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(b, [BLOCK_K, BLOCK_N], shared_layout_b)

    num_warps = warps[0] * warps[1]
    num_ctas = ctas_per_cga[0] * ctas_per_cga[1]

    try:
        tma_mma_shared_inputs_kernel[(1, )](
            a_desc,
            b_desc,
            out,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            NUM_K_TILES,
            block_layout_c,
            acc_layout,
            acc_tmem_layout,
            is_blackwell(),
            multicast=multicast,
            num_warps=num_warps,
            num_ctas=num_ctas,
        )
    except triton.runtime.errors.OutOfResources:
        pytest.skip("Too much shared memory required")

    try:
        allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = True
        ref = torch.matmul(a.to(torch.float32), b.to(torch.float32))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32

    if bitwidth == 8:
        atol, rtol = 8e-2, 8e-1
    elif bitwidth == 16:
        atol, rtol = 5e-2, 5e-1
    else:
        atol, rtol = 8e-4, 8e-3
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


@pytest.mark.skipif(not (is_hopper() or is_blackwell()), reason="Requires Hopper or Blackwell")
@pytest.mark.parametrize("bitwidth, transpose_a, transpose_b, acc_dtype",
                         [(bitwidth, transpose_a, transpose_b, acc_dtype)
                          for bitwidth in [8, 16, 32]
                          for (transpose_a, transpose_b) in product([False, True], repeat=2)
                          for acc_dtype in [torch.float16, torch.float32]
                          if bitwidth == 16 or (acc_dtype == torch.float32 and not transpose_a and transpose_b)])
@pytest.mark.parametrize("warps", ([8, 1], [4, 2], [4, 1]))
@pytest.mark.parametrize("swizzling_a, swizzling_b", product([0, 32, 64, 128], repeat=2))
@pytest.mark.parametrize("instr_m", [64, 128] if is_blackwell() else [64])
@pytest.mark.parametrize("shape_m, shape_n, shape_k", [(1, 1, 1), (2, 4, 1), (2, 2, 4)])
@pytest.mark.parametrize("ctas_per_cga", [[1, 1], [2, 1], [4, 4]])
@pytest.mark.parametrize("two_ctas", [False, True] if is_blackwell() else [False])
def test_mma_shared_inputs(bitwidth, transpose_a, transpose_b, acc_dtype, warps, swizzling_a, swizzling_b, instr_m,
                           shape_m, shape_n, shape_k, ctas_per_cga, two_ctas):
    # FIXME: Workaround for a bug in PTXAS when the shared layout is transposed and the swizzling is 0
    # This is fixed in PTXAS 13.0.88. Remove once we upgrade
    if bitwidth == 16 and ((transpose_a and swizzling_a == 0 and shape_m > 1) or
                           (not transpose_b and swizzling_b == 0 and shape_n > 1)):
        pytest.skip("Skipped due to a bug in PTXAS when the shared layout is transposed and the swizzling is 0")
    if ctas_per_cga[0] == 1 and two_ctas:
        pytest.skip("Need at least 2 CTAs for 2CTA mode")

    use_tcgen05 = is_blackwell()

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
    # instr_m is the instruction per warp group so we divide by 4
    instr_shape = [instr_m // 4, 32, 256 // bitwidth]
    M = instr_shape[0] * warps[0]
    N = instr_shape[1] * warps[1]
    K = instr_shape[2]

    if two_ctas:
        assert ctas_per_cga[0] >= 2, "Need at least 2 CTAs along M for 2CTA mode"
        ctas_per_cga_b = [ctas_per_cga[0] // 2, 2 * ctas_per_cga[1]]
    else:
        ctas_per_cga_b = ctas_per_cga
    cta_split_a = [ctas_per_cga[0], 1]
    cta_split_b = [1, ctas_per_cga_b[1]]

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
    M *= shape_m * ctas_per_cga[0]
    N *= shape_n * ctas_per_cga_b[1]
    K *= shape_k
    instr_shape[1] *= shape_n

    num_warps = warps[0] * warps[1]
    num_ctas = ctas_per_cga[0] * ctas_per_cga[1]

    if is_blackwell():
        # Avoid too many rows in TMEM
        MAX_ROWS = 512
        if M * N // 128 // num_ctas > MAX_ROWS:
            N //= (M * N // 128 // num_ctas // MAX_ROWS)

    total_shmem = (M + N) * K * bitwidth // 8
    device = triton.runtime.driver.active.get_current_device()
    MAX_SHMEM = max_shared_mem(device)
    if total_shmem > MAX_SHMEM:
        pytest.skip(f"Total shared memory {total_shmem} bytes exceeds the maximum allowed {MAX_SHMEM} bytes")

    if two_ctas and N // ctas_per_cga[1] == 512:
        # grep for [Note: numRepN > 1 and two_ctas]
        pytest.skip("grep for [Note: numRepN > 1 and two_ctas]")

    assert M >= 64, "M must be at least 64 for mmav3 and mmav5"

    def log2_int(x):
        return x.bit_length() - 1

    def get_shared_swizzling_zero(M, K, transpose, cga_layout):
        if cga_layout:
            dim_cga = [1, 1]
            for b in cga_layout:
                for i, bi in enumerate(b):
                    if bi != 0:
                        dim_cga[i] *= 2
            cta_shape = (M // dim_cga[0], K // dim_cga[1])
            cta_layout = get_shared_swizzling_zero(cta_shape[0], cta_shape[1], transpose, None)
            cga_bases = list(cga_layout)
            for b in cga_bases:
                for i in range(len(b)):
                    b[i] *= cta_shape[i]
            return ttgl.SharedLinearLayout(cta_layout.offset_bases, cga_bases)
        if transpose:
            assert not cga_layout
            shared = get_shared_swizzling_zero(K, M, False, cga_layout)
            # Transpose the bases
            bases = list(shared.offset_bases)
            for i in range(len(bases)):
                bases[i] = [bases[i][1], bases[i][0]]
            return ttgl.SharedLinearLayout(bases)
        bases = []
        for i in range(log2_int(128 // bitwidth)):
            bases.append([0, 1 << i])
        for i in range(log2_int(M)):
            bases.append([1 << i, 0])
        for i in range(log2_int(K // (128 // bitwidth))):
            offset = int(math.log2(128 // bitwidth)) + i
            bases.append([0, 1 << offset])
        return ttgl.SharedLinearLayout(bases)

    torch_dtype = torch_dtype_map[bitwidth]
    gl_acc_dtype = acc_dtype_map[acc_dtype]
    out_dtype = torch.float32
    cta_order = [1, 0]

    # TODO Remove this function altogether

    from triton._C.libtriton.gluon_ir import make_cga_layout
    if two_ctas:

        def make_2cta_cga_layout(ctas_per_cga, cta_split, cta_order, two_cta_dim):
            ctas_per_cga = list(ctas_per_cga)
            cta_split = list(cta_split)
            assert cta_split[two_cta_dim] > 1
            cta_split[two_cta_dim] //= 2
            ctas_per_cga[two_cta_dim] //= 2
            aux_cga_layout = make_cga_layout(ctas_per_cga, cta_split, cta_order)
            assert two_cta_dim in (0, 1)
            basis = [0, 0]
            basis[two_cta_dim] = 1
            for b in aux_cga_layout:
                b[two_cta_dim] *= 2
            cga_layout = [basis] + aux_cga_layout
            return cga_layout

        cga_layout_a = make_2cta_cga_layout(ctas_per_cga, cta_split_a, cta_order, 0)
        cga_layout_b = make_2cta_cga_layout(ctas_per_cga_b, cta_split_b, cta_order, 1)
        # The TMEM layout for instr_m == 128 splits along M, the one for instr_m == 64 splits along N
        cga_layout_c = make_2cta_cga_layout(ctas_per_cga, ctas_per_cga, cta_order, 0)
    else:
        cga_layout_a = make_cga_layout(ctas_per_cga, cta_split_a, cta_order)
        cga_layout_b = make_cga_layout(ctas_per_cga_b, cta_split_b, cta_order)
        cga_layout_c = make_cga_layout(ctas_per_cga, ctas_per_cga, cta_order)
    cga_layout_c = tuple(tuple(basis) for basis in cga_layout_c)

    block_layout_a = ttgl.BlockedLayout([1, 8], [1, THREADS_PER_WARP], warps_per_cta=warps, order=[0, 1],
                                        cga_layout=cga_layout_a)
    block_layout_b = ttgl.BlockedLayout([1, 8], [1, THREADS_PER_WARP], warps_per_cta=warps, order=[1, 0],
                                        cga_layout=cga_layout_b)
    if swizzling_a == 0:
        shared_layout_a = get_shared_swizzling_zero(M, K, transpose_a, cga_layout_a)
    else:
        shared_layout_a = ttgl.NVMMASharedLayout(swizzle_byte_width=swizzling_a, element_bitwidth=bitwidth, rank=2,
                                                 transposed=transpose_a, cga_layout=cga_layout_a)
    if swizzling_b == 0:
        shared_layout_b = get_shared_swizzling_zero(K, N, transpose_b, cga_layout_b)
    else:
        shared_layout_b = ttgl.NVMMASharedLayout(swizzle_byte_width=swizzling_b, element_bitwidth=bitwidth, rank=2,
                                                 transposed=transpose_b, cga_layout=cga_layout_b)
    if use_tcgen05:
        tmem_shape = (instr_m, min(N // ctas_per_cga[1], 256))
        acc_layout = TensorMemoryLayout(tmem_shape, col_stride=32 // torch.finfo(acc_dtype).bits,
                                        cta_split_num=tuple(ctas_per_cga), two_ctas=two_ctas)
    else:
        acc_layout = ttgl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=warps, instr_shape=instr_shape,
                                                 cga_layout=cga_layout_c)

    torch.manual_seed(0)

    def cast(x, dtype):
        if dtype != torch.float32:
            return x.to(dtype)
        # For b16 and fp32 (in both hopper and blackwell it seems)
        # Element-wise multiplication of matrix A and B is performed with specified precision.
        # wgmma.mma_async operation involving type .tf32 will truncate lower 13 bits of the 32-bit
        # input data before multiplication is issued
        x = x.view(torch.int32)
        x = x & ~((1 << 13) - 1)
        return x.view(dtype)

    # Sample bf16 as tf32 does not use the full range
    a = cast(torch.randn((M, K), device=device, dtype=torch.float32), torch_dtype)
    b = cast(torch.randn((K, N), device=device, dtype=torch.float32), torch_dtype)
    out = torch.zeros((M, N), device=device, dtype=out_dtype)

    compiled = mma_kernel[(1, )](
        a,
        b,
        out,
        M,
        N,
        K,
        block_layout_a,
        block_layout_b,
        cga_layout_c,
        acc_layout,
        shared_layout_a,
        shared_layout_b,
        gl_acc_dtype,
        False,
        use_tcgen05,
        num_warps=num_warps,
        num_ctas=num_ctas,
    )

    assert two_ctas == ("two_ctas" in compiled.asm["ttgir"])

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
        atol, rtol = 5e-2, 5e-1
    elif bitwidth == 16:
        atol, rtol = 5e-2, 5e-1
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
        cdna4_async_copy.commit_group()

        cdna4_async_copy.wait_group(0)
        a = cdna4_async_copy.load_shared_relaxed(smem, blocked)

        ttgl.store(b_ptr + offsets, a)

    torch.manual_seed(0)
    a = torch.randn((128, 16), dtype=torch.float16, device='cuda')
    b = torch.empty_like(a)
    pgm = kernel[(1, )](a, b, use_buffer_load)

    torch.testing.assert_close(a, b)
    assert re.search(r'ttg\.local_load .* \{ttg\.amdg\.syncedViaAsyncWait = true\}', pgm.asm['ttgir'], re.MULTILINE)
    if use_buffer_load:
        assert re.search(r"buffer_load.*lds$", pgm.asm['amdgcn'], re.MULTILINE)
    else:
        assert re.search(r"global_load_lds", pgm.asm['amdgcn'], re.MULTILINE)
    assert 'vmcnt(0)' in pgm.asm['amdgcn']


@pytest.mark.skipif(not (is_hip_rdna3() or is_hip_rdna4()), reason="Requires RDNA3 or RDNA4")
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
    wmma_version = 1 if is_hip_rdna3() else 2
    k_width = 16 if is_hip_rdna3() else 8
    wmma = ttgl.amd.AMDWMMALayout(wmma_version, True, [[0, 1], [1, 0]])
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
@pytest.mark.parametrize("M, N, K", [(32, 32, 128)])
@pytest.mark.parametrize("a_type, b_type", [(a_type, b_type)
                                            for a_type in ["e2m1", "e4m3", "e5m2"]
                                            for b_type in ["e2m1", "e4m3", "e5m2"]])
@pytest.mark.parametrize("has_scale", [True, False])
def test_amd_mfma_scaled(M, N, K, a_type, b_type, has_scale, device='cuda'):

    @gluon.jit
    def kernel(out_ptr, a_ptr, b_ptr, a_scale_ptr, b_scale_ptr,  #
               M: ttgl.constexpr, N: ttgl.constexpr, K: ttgl.constexpr,  #
               a_type: tl.constexpr, b_type: tl.constexpr):
        DIV_FACTOR_A: tl.constexpr = 2 if a_type == "e2m1" else 1
        DIV_FACTOR_B: tl.constexpr = 2 if b_type == "e2m1" else 1
        K_A: tl.constexpr = K // DIV_FACTOR_A
        K_B: tl.constexpr = K // DIV_FACTOR_B

        mfma_layout: ttgl.constexpr = ttgl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 128], transposed=True,
                                                             warps_per_cta=[2, 2])

        a_unpacked_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 16], [8, 8], [4, 1], [1, 0])
        a_packed_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 8], [8, 8], [4, 1], [1, 0])
        a_load_layout: ttgl.constexpr = a_packed_layout if a_type == "e2m1" else a_unpacked_layout
        a_layout: ttgl.constexpr = ttgl.DotOperandLayout(operand_index=0, parent=mfma_layout, k_width=16)
        a_scale_layout: ttgl.constexpr = ttgl.amd.cdna4.get_mfma_scale_layout(a_layout, [M, K // 32])

        b_unpacked_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 16], [32, 2], [4, 1], [1, 0])
        b_packed_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 8], [16, 4], [4, 1], [1, 0])
        b_load_layout: ttgl.constexpr = b_packed_layout if b_type == "e2m1" else b_unpacked_layout
        b_layout: ttgl.constexpr = ttgl.DotOperandLayout(operand_index=1, parent=mfma_layout, k_width=16)
        b_scale_layout: ttgl.constexpr = ttgl.amd.cdna4.get_mfma_scale_layout(b_layout, [N, K // 32])

        a_offs_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, a_load_layout))[:, None]
        a_offs_k = ttgl.arange(0, K_A, layout=ttgl.SliceLayout(0, a_load_layout))[None, :]
        a = ttgl.amd.cdna4.buffer_load(a_ptr, a_offs_m * K_A + a_offs_k)
        a = ttgl.convert_layout(a, a_layout)

        b_offs_k = ttgl.arange(0, K_B, layout=ttgl.SliceLayout(1, b_load_layout))[:, None]
        b_offs_n = ttgl.arange(0, N, layout=ttgl.SliceLayout(0, b_load_layout))[None, :]
        b = ttgl.amd.cdna4.buffer_load(b_ptr, b_offs_k * N + b_offs_n)
        b = ttgl.convert_layout(b, b_layout)

        a_scale = None
        if a_scale_ptr is not None:
            a_scale_offs_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, a_scale_layout))[:, None]
            a_scale_offs_k = ttgl.arange(0, K // 32, layout=ttgl.SliceLayout(0, a_scale_layout))[None, :]
            a_scale = ttgl.amd.cdna4.buffer_load(a_scale_ptr, a_scale_offs_m * (K // 32) + a_scale_offs_k)

        b_scale = None
        if b_scale_ptr is not None:
            b_scale_offs_n = ttgl.arange(0, N, layout=ttgl.SliceLayout(1, b_scale_layout))[:, None]
            b_scale_offs_k = ttgl.arange(0, K // 32, layout=ttgl.SliceLayout(0, b_scale_layout))[None, :]
            b_scale = ttgl.amd.cdna4.buffer_load(b_scale_ptr, b_scale_offs_n * (K // 32) + b_scale_offs_k)

        zero = ttgl.zeros([M, N], dtype=ttgl.float32, layout=mfma_layout)
        c = ttgl.amd.cdna4.mfma_scaled(a, a_scale, a_type, b, b_scale, b_type, zero)
        c = c.to(out_ptr.dtype.element_ty)

        out_offs_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, mfma_layout))[:, None]
        out_offs_n = ttgl.arange(0, N, layout=ttgl.SliceLayout(0, mfma_layout))[None, :]
        ttgl.amd.cdna4.buffer_store(c, out_ptr, out_offs_m * N + out_offs_n)

    def _create_mxfp_operand(operand: int, m: int, n: int, dtype: str):
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
        return v.to(device), v_ref.to(device)

    def _create_mxfp_scale(operand: int, m: int, n: int):
        size = (m, n // 32)
        scale = MXScaleTensor(size=tuple(size)).random(1 / 32, 32)
        scale_ref = scale.to(torch.float32).repeat_interleave(32, dim=1)
        scale_ref = scale_ref.T.contiguous() if operand == 1 else scale_ref
        return scale.data.to(device), scale_ref.to(device)

    torch.manual_seed(0)
    a, a_ref = _create_mxfp_operand(0, M, K, a_type)
    b, b_ref = _create_mxfp_operand(1, K, N, b_type)

    if has_scale:
        a_scale, a_scale_ref = _create_mxfp_scale(0, M, K)
        b_scale, b_scale_ref = _create_mxfp_scale(1, N, K)
        out = torch.empty((M, N), dtype=torch.float32, device=device)
        compiled = kernel[(1, )](out, a, b, a_scale, b_scale, M, N, K, a_type, b_type, num_warps=4)
        out_ref = torch.matmul(a_ref * a_scale_ref, b_ref * b_scale_ref)
        torch.testing.assert_close(out, out_ref)
    else:
        out = torch.empty((M, N), dtype=torch.float32, device=device)
        compiled = kernel[(1, )](out, a, b, None, None, M, N, K, a_type, b_type, num_warps=4)
        out_ref = torch.matmul(a_ref, b_ref)
        torch.testing.assert_close(out, out_ref)

    assert 'v_mfma_scale_f32_16x16x128_f8f6f4' in compiled.asm['amdgcn']


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

        layout: ttgl.constexpr = get_tmem_reg_layout(ttgl.float32, (BLOCK_M, N), tmem_layout, num_warps=4)

        offsets = ttgl.arange(0, BLOCK_M)[:, None] * N + ttgl.arange(0, N)[None, :]
        offsets = ttgl.set_auto_layout(offsets, layout)
        s = ttgl.load(s_ptr + offsets)

        s_tmem.store(s)
        o_tmem.store(s)

        p_tmem_layout: ttgl.constexpr = TensorMemoryLayout((BLOCK_M, BLOCK_N), col_stride=1)
        p_tmem = s_tmem.slice(0, N // 2)._reinterpret(ttgl.float16, [BLOCK_M, N], p_tmem_layout)
        p_tmem.store(ttgl.full((BLOCK_M, N), 0.0, dtype=ttgl.float16, layout=layout))

        d1_tmem_layout: ttgl.constexpr = TensorMemoryLayout((BLOCK_M, 2), col_stride=1)
        d1_layout: ttgl.constexpr = get_tmem_reg_layout(ttgl.float32, (BLOCK_M, 2), d1_tmem_layout, num_warps=4)

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

        a_tmem_layout: ttgl.constexpr = TensorMemoryLayout((BLOCK_M, BLOCK_N), col_stride=1)
        acc_tmem_layout: ttgl.constexpr = TensorMemoryLayout((BLOCK_M, BLOCK_N), col_stride=1)
        a_layout: ttgl.constexpr = get_tmem_reg_layout(ttgl.float16, (BLOCK_M, N), a_tmem_layout, num_warps=4,
                                                       instr_variant="32x32b_splitn")
        b_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [1, 32], [4, 1], [1, 0])
        a_offsets = ttgl.set_auto_layout(a_offsets, a_layout)
        b_offsets = ttgl.set_auto_layout(b_offsets, b_layout)

        a = ttgl.load(a_ptr + a_offsets)
        b = ttgl.load(b_ptr + b_offsets)
        c = ttgl.load(c_ptr + a_offsets)

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

        mbarrier.expect(bar, in_desc.nbytes_per_cta)
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
        tmem_layout: ttgl.constexpr = TensorMemoryLayout(
            block=(128, BLOCK_N),
            col_stride=32 // in_ptr.dtype.element_ty.primitive_bitwidth,
        )
        tmem_reg_layout: ttgl.constexpr = get_tmem_reg_layout(
            in_ptr.dtype.element_ty,
            (M, N),
            tmem_layout,
            num_warps=num_warps,
        )
        offs_m = ttgl.arange(0, M, ttgl.SliceLayout(1, tmem_reg_layout))
        offs_n = ttgl.arange(0, N, ttgl.SliceLayout(0, tmem_reg_layout))
        offs = offs_m[:, None] * N + offs_n[None, :]

        input = ttgl.load(in_ptr + offs)

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
    assert ttgir.count("amdg.buffer_atomic_rmw fadd, relaxed, cta") == 1

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


def fp8e8m0_to_float32(scale):
    scale = scale.view(torch.uint8)
    scale = scale.to(torch.int32)
    scale = scale << 23
    scale = scale.view(torch.float32)
    return scale


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_tcgen05_mma_scaled_minimal():
    M = 128
    N = 128
    K = 128
    threads_per_warp = ttgl.constexpr(THREADS_PER_WARP)

    @gluon.jit
    def kernel(out_ptr, M: ttgl.constexpr, N: ttgl.constexpr, K: ttgl.constexpr, a, b, a_scale, b_scale):
        # Simple register layout for creating constants and storing results
        reg_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [threads_per_warp, 1], [ttgl.num_warps(), 1], [1, 0])

        # Shared-memory layouts for MMA operands
        nvmma_layout: ttgl.constexpr = ttgl.NVMMASharedLayout(swizzle_byte_width=128, transposed=False,
                                                              element_bitwidth=8, rank=2)
        # Allocate zero operands in shared memory (values don't matter since scales are zero)
        block_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [1, 32], warps_per_cta=[ttgl.num_warps(), 1],
                                                          order=[1, 0])
        a_offs_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, block_layout))[:, None]
        a_offs_k = ttgl.arange(0, K, layout=ttgl.SliceLayout(0, block_layout))[None, :]
        b_offs_k = ttgl.arange(0, K, layout=ttgl.SliceLayout(1, block_layout))[:, None]
        b_offs_n = ttgl.arange(0, N, layout=ttgl.SliceLayout(0, block_layout))[None, :]

        a_tile = ttgl.load(a + a_offs_m * K + a_offs_k)
        b_tile = ttgl.load(b + b_offs_k * N + b_offs_n)
        a_smem = ttgl.allocate_shared_memory(ttgl.float8e5, [M, K], nvmma_layout, a_tile)
        b_smem = ttgl.allocate_shared_memory(ttgl.float8e5, [K, N], nvmma_layout, b_tile)

        # Accumulator in TMEM initialized to ones
        acc_tmem_layout: ttgl.constexpr = TensorMemoryLayout([M, N], col_stride=1)
        tmem_reg_layout: ttgl.constexpr = get_tmem_reg_layout(ttgl.float32, (M, N), acc_tmem_layout, ttgl.num_warps())
        acc_init = ttgl.zeros([M, N], ttgl.float32, layout=tmem_reg_layout)
        acc_tmem = allocate_tensor_memory(ttgl.float32, [M, N], acc_tmem_layout, acc_init)

        # Zero scales in TMEM
        scale_layout: ttgl.constexpr = TensorMemoryScalesLayout()
        scale_reg_layout_m: ttgl.constexpr = get_tmem_reg_layout(ttgl.int8, (M, K // 32), scale_layout,
                                                                 ttgl.num_warps())
        scale_reg_layout_n: ttgl.constexpr = get_tmem_reg_layout(ttgl.int8, (N, K // 32), scale_layout,
                                                                 ttgl.num_warps())
        scale_offs_k = ttgl.arange(0, (K // 32), layout=ttgl.SliceLayout(0, scale_reg_layout_m))[None, :]
        scale_offs_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, scale_reg_layout_m))[:, None]
        scale_offs_n = ttgl.arange(0, N, layout=ttgl.SliceLayout(1, scale_reg_layout_n))[:, None]
        a_scale_init = ttgl.load(a_scale + scale_offs_m * (K // 32) + scale_offs_k)
        b_scale_init = ttgl.load(b_scale + scale_offs_n * (K // 32) + scale_offs_k)
        a_scale_tmem = allocate_tensor_memory(ttgl.int8, [M, K // 32], scale_layout, a_scale_init)
        b_scale_tmem = allocate_tensor_memory(ttgl.int8, [M, K // 32], scale_layout, b_scale_init)

        # Issue a single scaled MMA and commit
        bar = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
        mbarrier.init(bar, count=1)
        tcgen05_mma_scaled(a_smem, b_smem, acc_tmem, a_scale_tmem, b_scale_tmem, "e5m2", "e5m2", use_acc=True)
        tcgen05_commit(bar)
        mbarrier.wait(bar, phase=0)

        # Load result from TMEM and store to global
        out_reg = acc_tmem.load(tmem_reg_layout)
        store_layout: ttgl.constexpr = reg_layout
        offs_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, store_layout))[:, None]
        offs_n = ttgl.arange(0, N, layout=ttgl.SliceLayout(0, store_layout))[None, :]
        offs = offs_m * N + offs_n
        ttgl.store(out_ptr + offs, ttgl.convert_layout(out_reg, store_layout))

    out = torch.empty((M, N), dtype=torch.float32, device="cuda")
    a = torch.randint(20, 40, (M, K), dtype=torch.uint8, device="cuda").view(torch.float8_e5m2)
    b = torch.randint(20, 40, (K, N), dtype=torch.uint8, device="cuda").view(torch.float8_e5m2)
    a_scale = torch.randint(64, 130, (M, K // 32), dtype=torch.uint8, device="cuda")
    b_scale = torch.randint(64, 130, (N, K // 32), dtype=torch.uint8, device="cuda")
    compiled = kernel[(1, )](out, M, N, K, a, b, a_scale, b_scale)
    A = a.to(torch.float32)
    B = b.to(torch.float32)
    a_scale_f32 = fp8e8m0_to_float32(a_scale)
    b_scale_f32 = fp8e8m0_to_float32(b_scale)
    a_scale_f32 = a_scale_f32.repeat_interleave(32, dim=1)
    b_scale_f32 = b_scale_f32.repeat_interleave(32, dim=1)
    b_scale_f32 = b_scale_f32.T.contiguous()
    A = A * a_scale_f32
    B = B * b_scale_f32
    ref = torch.matmul(A, B)
    torch.testing.assert_close(out, ref, atol=1e-6, rtol=1e-6)
    ttgir = compiled.asm["ttgir"]
    assert "ttng.tc_gen5_mma_scaled" in ttgir


@pytest.mark.skipif(not is_ampere_or_newer(), reason="Requires Ampere or newer")
def test_coalesced_layout():

    @gluon.jit
    def kernel(in_ptr, out_ptr,  #
               xnumel, ynumel, xstride_in, ystride_in, xstride_out, ystride_out,  #
               XBLOCK: ttgl.constexpr, YBLOCK: ttgl.constexpr):
        pid_x = ttgl.program_id(0)
        pid_y = ttgl.program_id(1)
        indices_x = pid_x * XBLOCK + ttgl.arange(0, XBLOCK, ttgl.CoalescedLayout())
        indices_y = pid_y * YBLOCK + ttgl.arange(0, YBLOCK, ttgl.CoalescedLayout())

        in_offsets = xstride_in * indices_x[:, None] + ystride_in * indices_y[None, :]
        out_offsets = xstride_out * indices_x[:, None] + ystride_out * indices_y[None, :]

        # MASK
        mask = (indices_x[:, None] < xnumel) & (indices_y[None, :] < ynumel)

        # IN PTR
        in_ptrs = in_ptr + in_offsets
        value = ttgl.load(in_ptrs, mask=mask)
        value = ttgl.sin(value)
        value = ttgl.maximum(value, 0.0)

        # OUT PTR
        out_ptrs = out_ptr + out_offsets
        ttgl.store(out_ptrs, value, mask=mask)

    XBLOCK = 128
    YBLOCK = 256
    xnumel = 1000
    ynumel = 2000
    input = torch.randn((xnumel, ynumel), device="cuda")
    output = torch.zeros_like(input)
    ref = torch.maximum(torch.sin(input), torch.tensor(0.0, device="cuda"))

    grid = (triton.cdiv(xnumel, XBLOCK), triton.cdiv(ynumel, YBLOCK))
    kernel[grid](  #
        input, output, xnumel, ynumel,  #
        *input.stride(), *output.stride(),  #
        XBLOCK, YBLOCK, num_warps=4)

    torch.testing.assert_close(output, ref)


@pytest.mark.skipif(not is_ampere_or_newer(), reason="Requires Ampere or newer")
def test_convert_auto_layout_to_coalesced_layout():

    @gluon.jit
    def kernel(in_ptr, out_ptr,  #
               xnumel, ynumel, xstride_in, ystride_in, xstride_out, ystride_out,  #
               XBLOCK: ttgl.constexpr, YBLOCK: ttgl.constexpr):
        pid_x = ttgl.program_id(0)
        pid_y = ttgl.program_id(1)
        indices_x = pid_x * XBLOCK + ttgl.arange(0, XBLOCK, ttgl.AutoLayout())
        indices_y = pid_y * YBLOCK + ttgl.arange(0, YBLOCK, ttgl.AutoLayout())

        in_offsets = xstride_in * indices_x[:, None] + ystride_in * indices_y[None, :]
        out_offsets = xstride_out * indices_x[:, None] + ystride_out * indices_y[None, :]

        # MASK
        mask = (indices_x[:, None] < xnumel) & (indices_y[None, :] < ynumel)  # auto layout

        # IN PTR
        in_ptrs = ttgl.set_auto_layout(in_ptr + in_offsets, ttgl.CoalescedLayout())
        value = ttgl.load(in_ptrs, mask=mask)

        # OUT PTR
        out_ptrs = ttgl.set_auto_layout(out_ptr + out_offsets, ttgl.CoalescedLayout())
        out_mask_layouted = ttgl.set_auto_layout(mask, ttgl.CoalescedLayout())
        ttgl.store(out_ptrs, value, mask=out_mask_layouted)

    XBLOCK = 128
    YBLOCK = 256
    xnumel = 1000
    ynumel = 2000
    input = torch.ones((xnumel, ynumel), device="cuda")
    output = torch.zeros_like(input)
    ref = torch.ones_like(input)

    grid = (triton.cdiv(xnumel, XBLOCK), triton.cdiv(ynumel, YBLOCK))
    kernel[grid](  #
        input, output, xnumel, ynumel,  #
        *input.stride(), *output.stride(),  #
        XBLOCK, YBLOCK, num_warps=4)

    torch.testing.assert_close(output, ref)


@gluon.jit
def in_thread_transpose_roundtrip_kernel(input, output, M: ttgl.constexpr, N: ttgl.constexpr,
                                         first_layout: ttgl.constexpr, second_layout: ttgl.constexpr,
                                         shared_layout: ttgl.constexpr):
    offs_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, first_layout))[:, None]
    offs_n = ttgl.arange(0, N, layout=ttgl.SliceLayout(0, first_layout))[None, :]

    load_data = ttgl.load(input + offs_m * N + offs_n)
    converted_data = ttgl.convert_layout(load_data, second_layout)
    smem = ttgl.allocate_shared_memory(input.dtype.element_ty, [M, N], shared_layout, converted_data)
    out_data = smem.load(first_layout)
    ttgl.store(output + offs_m * N + offs_n, out_data)


@pytest.mark.skipif(not (is_hip_cdna() or is_hip_rdna()),
                    reason="Correctness tests for special cases on AMD architectures")
@pytest.mark.parametrize("reg_bases", [
    [[1, 0], [2, 0], [4, 0], [0, 1], [0, 2], [0, 4]],
    [[0, 1], [0, 4], [0, 2], [1, 0], [2, 0], [4, 0]],
    [[0, 2], [0, 1], [0, 4], [1, 0], [4, 0], [2, 0]],
])
def test_in_thread_convert_layout_8bit(reg_bases):
    torch.manual_seed(0)
    dtype = torch.int8
    first_layout = ttgl.BlockedLayout([8, 8], [1, THREADS_PER_WARP], warps_per_cta=[1, 1], order=[1, 0])
    M = first_layout.size_per_thread[0] * first_layout.threads_per_warp[0] * first_layout.warps_per_cta[0]
    N = first_layout.size_per_thread[1] * first_layout.threads_per_warp[1] * first_layout.warps_per_cta[1]

    numLaneBases = int(math.log2(THREADS_PER_WARP))
    lane_bases = [[0, 8 * (2**baseNo)] for baseNo in range(numLaneBases)]
    warp_bases = []
    second_layout = ttgl.DistributedLinearLayout(reg_bases=reg_bases, lane_bases=lane_bases, warp_bases=warp_bases,
                                                 block_bases=[], shape=[M, N])

    shared_layout = ttgl.SwizzledSharedLayout(1, 1, 1, order=[0, 1])
    input_buffer = (torch.randn((M, N), device="cuda") * 100).to(dtype)
    output_buffer = torch.zeros((M, N), device="cuda", dtype=dtype)
    pgm = in_thread_transpose_roundtrip_kernel[(1, )](input_buffer, output_buffer, M, N, first_layout, second_layout,
                                                      shared_layout, num_warps=1)

    assert re.search(r"v_perm", pgm.asm['amdgcn'], re.MULTILINE)
    torch.testing.assert_close(input_buffer, output_buffer, atol=1e-3, rtol=1e-3)


@gluon.jit
def descriptor_shape_kernel(desc, expect_shape):
    for i in ttgl.static_range(len(expect_shape)):
        ttgl.device_assert(desc.shape[i] == expect_shape[i])


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_descriptor_shape():
    t = torch.randint(0, 256, (512, 512), dtype=torch.uint8)

    for fp4_padded in [True]:
        layout = ttgl.NVMMASharedLayout.get_default_for([128, 64], ttgl.uint8, fp4_padded=fp4_padded)
        desc = TensorDescriptor.from_tensor(t, [128, 64], layout)
        descriptor_shape_kernel[(1, )](desc, t.shape, num_warps=1, debug=True)
        torch.cuda.synchronize()


@gluon.jit
def shared_gather_kernel(
    matrix_ptr,
    indices_ptr,
    output_ptr,
    N: ttgl.constexpr,
    M: ttgl.constexpr,
    layout_2d: ttgl.constexpr,
    layout_1d: ttgl.constexpr,
    shared_layout: ttgl.constexpr,
):
    """Test shared memory gather using smem.gather() with axis-based API."""
    # Load the matrix from global memory into registers
    indices_x = ttgl.arange(0, N, layout=ttgl.SliceLayout(dim=1, parent=layout_2d))
    indices_y = ttgl.arange(0, M, layout=ttgl.SliceLayout(dim=0, parent=layout_2d))
    offsets_2d = indices_x[:, None] * M + indices_y[None, :]
    matrix_data = ttgl.load(matrix_ptr + offsets_2d)

    # Allocate 2D shared memory and store the matrix
    smem_2d = ttgl.allocate_shared_memory(ttgl.float32, [N, M], layout=shared_layout)
    smem_2d.store(matrix_data)

    # Reshape to 1D to test gather along axis 0
    smem_1d = smem_2d.reshape([N * M])

    # Load the gather indices (diagonal elements: 0, M+1, 2*(M+1), ...)
    offsets_1d = ttgl.arange(0, N, layout=layout_1d)
    indices = ttgl.load(indices_ptr + offsets_1d)

    # Gather using axis-based API: result[i] = smem_1d[indices[i]]
    gathered = smem_1d.gather(indices, axis=0)

    # Store result to global memory
    ttgl.store(output_ptr + offsets_1d, gathered)


@pytest.mark.parametrize("N,M", [(32, 32), (64, 64), (128, 128)])
def test_shared_gather(N, M):
    """Test gathering from 1D reshaped shared memory (diagonal of 2D matrix)."""
    device = torch.device("cuda")

    # Create a test matrix with known values
    matrix = torch.arange(N * M, dtype=torch.float32, device=device).reshape(N, M)

    # Create gather indices for diagonal elements: 0, M+1, 2*(M+1), ...
    indices = torch.arange(N, dtype=torch.int32, device=device) * (M + 1)

    output = torch.zeros(N, dtype=torch.float32, device=device)

    # Compute expected result: diagonal elements
    expected = matrix.flatten()[indices]

    # Create layouts dynamically based on THREADS_PER_WARP
    layout_2d = ttgl.BlockedLayout(size_per_thread=[1, 1], threads_per_warp=[THREADS_PER_WARP // 4, 4],
                                   warps_per_cta=[1, 1], order=[1, 0])
    layout_1d = ttgl.BlockedLayout(size_per_thread=[1], threads_per_warp=[THREADS_PER_WARP], warps_per_cta=[1],
                                   order=[0])
    shared_layout = ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])

    # Launch kernel
    shared_gather_kernel[(1, )](
        matrix,
        indices,
        output,
        N=N,
        M=M,
        layout_2d=layout_2d,
        layout_1d=layout_1d,
        shared_layout=shared_layout,
        num_warps=1,
    )

    torch.testing.assert_close(output, expected)


@gluon.jit
def shared_scatter_kernel(
    indices_ptr,
    values_ptr,
    output_ptr,
    N: ttgl.constexpr,
    M: ttgl.constexpr,
    layout_2d: ttgl.constexpr,
    layout_1d: ttgl.constexpr,
    shared_layout: ttgl.constexpr,
):
    """Test shared memory scatter using smem.scatter() with axis-based API."""
    # Allocate 2D shared memory initialized to zero
    smem = ttgl.allocate_shared_memory(ttgl.float32, [N, M], layout=shared_layout)

    # Initialize shared memory to zero
    indices_x = ttgl.arange(0, N, layout=ttgl.SliceLayout(dim=1, parent=layout_2d))
    indices_y = ttgl.arange(0, M, layout=ttgl.SliceLayout(dim=0, parent=layout_2d))
    offsets_2d = indices_x[:, None] * M + indices_y[None, :]
    zeros = ttgl.zeros([N, M], ttgl.float32, layout=layout_2d)
    smem.store(zeros)

    # Reshape to 1D to test scatter along axis 0
    smem_1d = smem.reshape([N * M])

    # Load the scatter indices and values (diagonal elements: 0, M+1, 2*(M+1), ...)
    offsets_1d = ttgl.arange(0, N, layout=layout_1d)
    indices = ttgl.load(indices_ptr + offsets_1d)
    values = ttgl.load(values_ptr + offsets_1d)

    # Scatter using axis-based API: smem_1d[indices[i]] = values[i]
    smem_1d.scatter(values, indices, axis=0)

    # Read back the full matrix from shared memory
    matrix_data = smem.load(layout=layout_2d)

    # Store result to global memory
    ttgl.store(output_ptr + offsets_2d, matrix_data)


@pytest.mark.parametrize("N,M", [(32, 32), (64, 64), (128, 128)])
def test_shared_scatter(N, M):
    """Test scattering to 1D reshaped shared memory (diagonal of 2D matrix)."""
    device = torch.device("cuda")

    # Create scatter indices for diagonal elements: 0, M+1, 2*(M+1), ...
    indices = torch.arange(N, dtype=torch.int32, device=device) * (M + 1)

    # Create values to scatter
    values = torch.arange(N, dtype=torch.float32, device=device) + 100.0

    output = torch.zeros((N, M), dtype=torch.float32, device=device)

    # Compute expected result: matrix starts at zero, then diagonal gets values
    expected = torch.zeros((N, M), dtype=torch.float32, device=device)
    for i in range(N):
        expected[i, i] = values[i]

    # Create layouts dynamically based on THREADS_PER_WARP
    layout_2d = ttgl.BlockedLayout(size_per_thread=[1, 1], threads_per_warp=[THREADS_PER_WARP // 4, 4],
                                   warps_per_cta=[1, 1], order=[1, 0])
    layout_1d = ttgl.BlockedLayout(size_per_thread=[1], threads_per_warp=[THREADS_PER_WARP], warps_per_cta=[1],
                                   order=[0])
    shared_layout = ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])

    # Launch kernel
    shared_scatter_kernel[(1, )](
        indices,
        values,
        output,
        N=N,
        M=M,
        layout_2d=layout_2d,
        layout_1d=layout_1d,
        shared_layout=shared_layout,
        num_warps=1,
    )

    torch.testing.assert_close(output, expected)


# ============================================================================
# Multi-warp Tests
# ============================================================================


@pytest.mark.parametrize("N,M,num_warps", [(64, 64, 2), (128, 128, 4)])
def test_scatter_gather_multiwarp(N, M, num_warps):
    """Test scatter and gather with multiple warps."""
    device = torch.device("cuda")

    # Create layouts with multiple warps (shared across both tests)
    layout_2d = ttgl.BlockedLayout(size_per_thread=[1, 1], threads_per_warp=[THREADS_PER_WARP // 4, 4],
                                   warps_per_cta=[num_warps, 1], order=[1, 0])
    layout_1d = ttgl.BlockedLayout(size_per_thread=[1], threads_per_warp=[THREADS_PER_WARP], warps_per_cta=[num_warps],
                                   order=[0])
    shared_layout = ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])

    # Test gather
    matrix = torch.arange(N * M, dtype=torch.float32, device=device).reshape(N, M)
    gather_indices = torch.arange(N, dtype=torch.int32, device=device) * (M + 1)
    gather_output = torch.zeros(N, dtype=torch.float32, device=device)
    gather_expected = matrix.flatten()[gather_indices]

    shared_gather_kernel[(1, )](
        matrix,
        gather_indices,
        gather_output,
        N=N,
        M=M,
        layout_2d=layout_2d,
        layout_1d=layout_1d,
        shared_layout=shared_layout,
        num_warps=num_warps,
    )

    torch.testing.assert_close(gather_output, gather_expected)

    # Test scatter
    scatter_indices = torch.arange(N, dtype=torch.int32, device=device) * (M + 1)
    scatter_values = torch.arange(N, dtype=torch.float32, device=device) + 100.0
    scatter_output = torch.zeros((N, M), dtype=torch.float32, device=device)
    scatter_expected = torch.zeros((N, M), dtype=torch.float32, device=device)
    for i in range(N):
        scatter_expected[i, i] = scatter_values[i]

    shared_scatter_kernel[(1, )](
        scatter_indices,
        scatter_values,
        scatter_output,
        N=N,
        M=M,
        layout_2d=layout_2d,
        layout_1d=layout_1d,
        shared_layout=shared_layout,
        num_warps=num_warps,
    )

    torch.testing.assert_close(scatter_output, scatter_expected)


# ============================================================================
# 2D Native Gather/Scatter Tests
# ============================================================================


@gluon.jit
def gather_2d_kernel(
    matrix_ptr,
    indices_ptr,
    output_ptr,
    N: ttgl.constexpr,
    M: ttgl.constexpr,
    axis: ttgl.constexpr,
    layout_2d: ttgl.constexpr,
    shared_layout: ttgl.constexpr,
):
    """Test 2D gather along specified axis."""
    # Load the matrix from global memory [N, M]
    indices_x = ttgl.arange(0, N, layout=ttgl.SliceLayout(dim=1, parent=layout_2d))
    indices_y = ttgl.arange(0, M, layout=ttgl.SliceLayout(dim=0, parent=layout_2d))
    offsets_2d = indices_x[:, None] * M + indices_y[None, :]
    matrix_data = ttgl.load(matrix_ptr + offsets_2d)

    # Store in shared memory
    smem = ttgl.allocate_shared_memory(ttgl.float32, [N, M], layout=shared_layout)
    smem.store(matrix_data)

    # Load indices [N, M] - same rank as source
    indices = ttgl.load(indices_ptr + offsets_2d)

    # Gather along specified axis
    gathered = smem.gather(indices, axis=axis)

    # Store result
    ttgl.store(output_ptr + offsets_2d, gathered)


@pytest.mark.parametrize("N,M,axis", [(32, 32, 0), (32, 32, 1), (64, 64, 0), (64, 64, 1)])
def test_gather_2d_native(N, M, axis):
    """Test 2D gather along different axes."""
    device = torch.device("cuda")

    # Create a test matrix [N, M]
    matrix = torch.arange(N * M, dtype=torch.float32, device=device).reshape(N, M)

    # Create indices [N, M] - each position specifies where to gather from along the axis
    if axis == 0:
        # Each column gathers from a shifted row pattern
        indices = torch.arange(M, dtype=torch.int32, device=device)[None, :].expand(N, M)
        indices = (indices + torch.arange(N, dtype=torch.int32, device=device)[:, None]) % N
        # Expected: result[i, j] = matrix[indices[i, j], j]
        expected = torch.gather(matrix, 0, indices.long())
    else:  # axis == 1
        # Each row gathers from a shifted column pattern
        indices = torch.arange(N, dtype=torch.int32, device=device)[:, None].expand(N, M)
        indices = (indices + torch.arange(M, dtype=torch.int32, device=device)[None, :]) % M
        # Expected: result[i, j] = matrix[i, indices[i, j]]
        expected = torch.gather(matrix, 1, indices.long())

    output = torch.zeros((N, M), dtype=torch.float32, device=device)

    # Create layouts dynamically based on THREADS_PER_WARP
    layout_2d = ttgl.BlockedLayout(size_per_thread=[1, 1], threads_per_warp=[THREADS_PER_WARP // 4, 4],
                                   warps_per_cta=[1, 1], order=[1, 0])
    shared_layout = ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])

    gather_2d_kernel[(1, )](
        matrix,
        indices,
        output,
        N=N,
        M=M,
        axis=axis,
        layout_2d=layout_2d,
        shared_layout=shared_layout,
        num_warps=1,
    )

    torch.testing.assert_close(output, expected)


@gluon.jit
def scatter_2d_kernel(
    indices_ptr,
    values_ptr,
    output_ptr,
    N: ttgl.constexpr,
    M: ttgl.constexpr,
    axis: ttgl.constexpr,
    layout_2d: ttgl.constexpr,
    shared_layout: ttgl.constexpr,
):
    """Test 2D scatter along specified axis."""
    # Initialize shared memory to zero
    smem = ttgl.allocate_shared_memory(ttgl.float32, [N, M], layout=shared_layout)

    indices_x = ttgl.arange(0, N, layout=ttgl.SliceLayout(dim=1, parent=layout_2d))
    indices_y = ttgl.arange(0, M, layout=ttgl.SliceLayout(dim=0, parent=layout_2d))
    offsets_2d = indices_x[:, None] * M + indices_y[None, :]
    zeros = ttgl.zeros([N, M], ttgl.float32, layout=layout_2d)
    smem.store(zeros)

    # Load indices [N, M] and values [N, M]
    indices = ttgl.load(indices_ptr + offsets_2d)
    values = ttgl.load(values_ptr + offsets_2d)

    # Scatter along specified axis
    smem.scatter(values, indices, axis=axis)

    # Read back the result
    result = smem.load(layout=layout_2d)
    ttgl.store(output_ptr + offsets_2d, result)


@pytest.mark.parametrize("N,M,axis", [(32, 32, 0), (32, 32, 1)])
def test_scatter_2d_native(N, M, axis):
    """Test 2D scatter along different axes."""
    device = torch.device("cuda")

    # Create indices [N, M] - reverse pattern for scatter
    if axis == 0:
        indices = torch.arange(M, dtype=torch.int32, device=device)[None, :].expand(N, M)
        indices = (N - 1 - indices - torch.arange(N, dtype=torch.int32, device=device)[:, None]) % N
    else:  # axis == 1
        indices = torch.arange(N, dtype=torch.int32, device=device)[:, None].expand(N, M)
        indices = (M - 1 - indices - torch.arange(M, dtype=torch.int32, device=device)[None, :]) % M

    # Create values to scatter
    values = torch.arange(N * M, dtype=torch.float32, device=device).reshape(N, M) + 100.0

    output = torch.zeros((N, M), dtype=torch.float32, device=device)

    # Expected: scatter values according to indices
    expected = torch.zeros((N, M), dtype=torch.float32, device=device)
    expected.scatter_(axis, indices.long(), values)

    # Create layouts dynamically based on THREADS_PER_WARP
    layout_2d = ttgl.BlockedLayout(size_per_thread=[1, 1], threads_per_warp=[THREADS_PER_WARP // 4, 4],
                                   warps_per_cta=[1, 1], order=[1, 0])
    shared_layout = ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])

    scatter_2d_kernel[(1, )](
        indices,
        values,
        output,
        N=N,
        M=M,
        axis=axis,
        layout_2d=layout_2d,
        shared_layout=shared_layout,
        num_warps=1,
    )

    torch.testing.assert_close(output, expected)


# ============================================================================
# 3D Gather/Scatter Tests
# ============================================================================


@gluon.jit
def gather_3d_kernel(
    tensor_ptr,
    indices_ptr,
    output_ptr,
    N: ttgl.constexpr,
    M: ttgl.constexpr,
    P: ttgl.constexpr,
    axis: ttgl.constexpr,
    layout_3d: ttgl.constexpr,
    shared_layout: ttgl.constexpr,
):
    """Test 3D gather along specified axis."""
    # Load the tensor from global memory [N, M, P]
    idx_n = ttgl.arange(0, N)[:, None, None]
    idx_m = ttgl.arange(0, M)[None, :, None]
    idx_p = ttgl.arange(0, P)[None, None, :]

    offsets_3d = idx_n * (M * P) + idx_m * P + idx_p
    offsets_3d = ttgl.set_auto_layout(offsets_3d, layout_3d)

    tensor_data = ttgl.load(tensor_ptr + offsets_3d)

    # Store in shared memory
    smem = ttgl.allocate_shared_memory(ttgl.float32, [N, M, P], layout=shared_layout)
    smem.store(tensor_data)

    # Load indices [N, M, P] - same rank as source
    indices_data = ttgl.load(indices_ptr + offsets_3d)

    # Gather along specified axis
    gathered = smem.gather(indices_data, axis=axis)

    # Store result
    ttgl.store(output_ptr + offsets_3d, gathered)


@pytest.mark.parametrize("N,M,P,axis", [(16, 8, 4, 0), (16, 8, 4, 1), (16, 8, 4, 2)])
def test_gather_3d_native(N, M, P, axis):
    """Test 3D gather along different axes."""
    device = torch.device("cuda")

    # Create a test tensor [N, M, P]
    tensor = torch.arange(N * M * P, dtype=torch.float32, device=device).reshape(N, M, P)

    # Create indices [N, M, P] - each position specifies where to gather from along the axis
    if axis == 0:
        # Pattern for gathering along first dimension
        base = torch.arange(M * P, dtype=torch.int32, device=device).reshape(1, M, P)
        offset = torch.arange(N, dtype=torch.int32, device=device).reshape(N, 1, 1)
        indices = (base + offset) % N
    elif axis == 1:
        # Pattern for gathering along second dimension
        base = torch.arange(N, dtype=torch.int32, device=device).reshape(N, 1, 1)
        offset = torch.arange(P, dtype=torch.int32, device=device).reshape(1, 1, P)
        indices = ((base + offset) % M).expand(N, M, P).contiguous()
    else:  # axis == 2
        # Pattern for gathering along third dimension
        base = torch.arange(N * M, dtype=torch.int32, device=device).reshape(N, M, 1)
        indices = (base % P).expand(N, M, P).contiguous()

    # Ensure indices is contiguous in C-style layout
    indices = indices.contiguous()

    # Compute expected result using torch.gather
    expected = torch.gather(tensor, axis, indices.long())

    output = torch.zeros((N, M, P), dtype=torch.float32, device=device)

    # Create layouts dynamically based on THREADS_PER_WARP
    layout_3d = ttgl.BlockedLayout(size_per_thread=[1, 1, 1], threads_per_warp=[4, 4, THREADS_PER_WARP // 16],
                                   warps_per_cta=[1, 1, 1], order=[2, 1, 0])
    shared_layout = ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[2, 1, 0])

    gather_3d_kernel[(1, )](
        tensor,
        indices,
        output,
        N=N,
        M=M,
        P=P,
        axis=axis,
        layout_3d=layout_3d,
        shared_layout=shared_layout,
        num_warps=1,
    )

    torch.testing.assert_close(output, expected)


@gluon.jit
def scatter_3d_kernel(
    indices_ptr,
    values_ptr,
    output_ptr,
    N: ttgl.constexpr,
    M: ttgl.constexpr,
    P: ttgl.constexpr,
    axis: ttgl.constexpr,
    layout_3d: ttgl.constexpr,
    shared_layout: ttgl.constexpr,
):
    """Test 3D scatter along specified axis."""
    idx_n = ttgl.arange(0, N)[:, None, None]
    idx_m = ttgl.arange(0, M)[None, :, None]
    idx_p = ttgl.arange(0, P)[None, None, :]

    offsets_3d = idx_n * (M * P) + idx_m * P + idx_p
    offsets_3d = ttgl.set_auto_layout(offsets_3d, layout_3d)

    # Initialize shared memory to zero
    smem = ttgl.allocate_shared_memory(ttgl.float32, [N, M, P], layout=shared_layout)
    zeros = ttgl.full([N, M, P], 0.0, ttgl.float32, layout=layout_3d)
    smem.store(zeros)

    # Load indices [N, M, P] and values [N, M, P]
    indices_data = ttgl.load(indices_ptr + offsets_3d)
    values_data = ttgl.load(values_ptr + offsets_3d)

    # Scatter along specified axis
    smem.scatter(values_data, indices_data, axis=axis)

    # Read back the result
    result = smem.load(layout=layout_3d)
    ttgl.store(output_ptr + offsets_3d, result)


@pytest.mark.parametrize("N,M,P,axis", [(16, 8, 4, 0), (16, 8, 4, 1), (16, 8, 4, 2)])
def test_scatter_3d_native(N, M, P, axis):
    """Test 3D scatter along different axes."""
    device = torch.device("cuda")

    # Create indices [N, M, P] that form a permutation along the scatter axis
    if axis == 0:
        # For axis 0: permute N dimension, keeping (M, P) coordinates fixed
        # Each (j, k) position has a unique permutation of N indices
        base = torch.arange(M * P, dtype=torch.int32, device=device).reshape(1, M, P)
        offset = torch.arange(N, dtype=torch.int32, device=device).reshape(N, 1, 1)
        indices = ((N - 1 - base - offset) % N).contiguous()
    elif axis == 1:
        # For axis 1: permute M dimension, keeping (N, P) coordinates fixed
        # Each (i, k) position has a unique permutation of M indices
        base = torch.arange(N * P, dtype=torch.int32, device=device).reshape(N, 1, P)
        offset = torch.arange(M, dtype=torch.int32, device=device).reshape(1, M, 1)
        indices = ((M - 1 - base - offset) % M).contiguous()
    else:  # axis == 2
        # For axis 2: permute P dimension, keeping (N, M) coordinates fixed
        # Each (i, j) position has a unique permutation of P indices
        base = torch.arange(N * M, dtype=torch.int32, device=device).reshape(N, M, 1)
        offset = torch.arange(P, dtype=torch.int32, device=device).reshape(1, 1, P)
        indices = ((P - 1 - base - offset) % P).contiguous()

    # Ensure indices is contiguous
    indices = indices.contiguous()

    # Create values to scatter
    values = (torch.arange(N * M * P, dtype=torch.float32, device=device).reshape(N, M, P) + 200.0).contiguous()

    output = torch.zeros((N, M, P), dtype=torch.float32, device=device)

    # Expected: scatter values according to indices
    expected = torch.zeros((N, M, P), dtype=torch.float32, device=device)
    expected.scatter_(axis, indices.long(), values)

    # Create layouts dynamically based on THREADS_PER_WARP
    layout_3d = ttgl.BlockedLayout(size_per_thread=[1, 1, 1], threads_per_warp=[4, 4, THREADS_PER_WARP // 16],
                                   warps_per_cta=[1, 1, 1], order=[2, 1, 0])
    shared_layout = ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[2, 1, 0])

    scatter_3d_kernel[(1, )](
        indices,
        values,
        output,
        N=N,
        M=M,
        P=P,
        axis=axis,
        layout_3d=layout_3d,
        shared_layout=shared_layout,
        num_warps=1,
    )

    torch.testing.assert_close(output, expected)


# =============================================================================
# Subslice Tests (2D slicing along individual dimensions)
# =============================================================================


@gluon.jit
def gather_subslice_2d_kernel(
    matrix_ptr,
    indices_ptr,
    output_ptr,
    M: ttgl.constexpr,
    N: ttgl.constexpr,
    SLICE_M_OFFSET: ttgl.constexpr,
    SLICE_N_OFFSET: ttgl.constexpr,
    SLICE_M: ttgl.constexpr,
    SLICE_N: ttgl.constexpr,
    layout_full: ttgl.constexpr,
    layout_slice: ttgl.constexpr,
    shared_layout: ttgl.constexpr,
):
    """Gather from a 2D subsliced shared memory descriptor."""
    # Load full matrix into shared memory
    offs_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, layout_full))[:, None]
    offs_n = ttgl.arange(0, N, layout=ttgl.SliceLayout(0, layout_full))[None, :]
    in_offs = offs_m * N + offs_n
    in_data = ttgl.load(matrix_ptr + in_offs)

    smem = ttgl.allocate_shared_memory(ttgl.float32, [M, N], layout=shared_layout)
    smem.store(in_data)

    # Create 2D subslice
    smem_slice = smem.slice(SLICE_M_OFFSET, SLICE_M, dim=0).slice(SLICE_N_OFFSET, SLICE_N, dim=1)

    # Load indices for gathering within the slice
    slice_offs_m = ttgl.arange(0, SLICE_M, layout=ttgl.SliceLayout(1, layout_slice))[:, None]
    slice_offs_n = ttgl.arange(0, SLICE_N, layout=ttgl.SliceLayout(0, layout_slice))[None, :]
    idx_offs = slice_offs_m * SLICE_N + slice_offs_n
    indices = ttgl.load(indices_ptr + idx_offs)

    # Gather along axis 0: result[i, j] = smem_slice[indices[i, j], j]
    gathered = smem_slice.gather(indices, axis=0)

    # Store result
    ttgl.store(output_ptr + idx_offs, gathered)


@pytest.mark.parametrize("M,N,slice_m_offset,slice_n_offset,slice_m,slice_n", [
    # Offset must be a multiple of tile (slice) size for each dimension
    (64, 64, 48, 16, 16, 16),  # offset 48 % 16 == 0, offset 16 % 16 == 0
    (64, 64, 32, 48, 32, 16),  # offset 32 % 32 == 0, offset 48 % 16 == 0
    (64, 64, 48, 32, 16, 32),  # offset 48 % 16 == 0, offset 32 % 32 == 0
])
def test_gather_subslice_2d(M, N, slice_m_offset, slice_n_offset, slice_m, slice_n):
    """Test gathering from a 2D subsliced shared memory descriptor."""
    device = torch.device("cuda")

    # Create input matrix
    matrix = torch.arange(M * N, dtype=torch.float32, device=device).reshape(M, N)

    # Create indices for gather (within the slice dimensions)
    # Each position gathers from a shifted row
    indices = torch.arange(slice_n, dtype=torch.int32, device=device)[None, :].expand(slice_m, slice_n)
    indices = (indices + torch.arange(slice_m, dtype=torch.int32, device=device)[:, None]) % slice_m

    output = torch.zeros((slice_m, slice_n), dtype=torch.float32, device=device)

    # Expected: gather from the subslice
    subslice = matrix[slice_m_offset:slice_m_offset + slice_m, slice_n_offset:slice_n_offset + slice_n]
    expected = torch.gather(subslice, 0, indices.long())

    # Layouts
    layout_full = ttgl.BlockedLayout(
        size_per_thread=[1, 1],
        threads_per_warp=[THREADS_PER_WARP // 4, 4],
        warps_per_cta=[1, 1],
        order=[1, 0],
    )
    layout_slice = ttgl.BlockedLayout(
        size_per_thread=[1, 1],
        threads_per_warp=[THREADS_PER_WARP // 4, 4],
        warps_per_cta=[1, 1],
        order=[1, 0],
    )
    # Use non-swizzled layout for subslicing
    shared_layout = ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])

    gather_subslice_2d_kernel[(1, )](
        matrix,
        indices,
        output,
        M=M,
        N=N,
        SLICE_M_OFFSET=slice_m_offset,
        SLICE_N_OFFSET=slice_n_offset,
        SLICE_M=slice_m,
        SLICE_N=slice_n,
        layout_full=layout_full,
        layout_slice=layout_slice,
        shared_layout=shared_layout,
        num_warps=1,
    )

    torch.testing.assert_close(output, expected)


@gluon.jit
def scatter_subslice_2d_kernel(
    indices_ptr,
    values_ptr,
    output_ptr,
    M: ttgl.constexpr,
    N: ttgl.constexpr,
    SLICE_M_OFFSET: ttgl.constexpr,
    SLICE_N_OFFSET: ttgl.constexpr,
    SLICE_M: ttgl.constexpr,
    SLICE_N: ttgl.constexpr,
    layout_full: ttgl.constexpr,
    layout_slice: ttgl.constexpr,
    shared_layout: ttgl.constexpr,
):
    """Scatter to a 2D subsliced shared memory descriptor."""
    # Initialize shared memory with -1
    offs_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, layout_full))[:, None]
    offs_n = ttgl.arange(0, N, layout=ttgl.SliceLayout(0, layout_full))[None, :]
    full_offs = offs_m * N + offs_n
    init_data = ttgl.full([M, N], -1.0, dtype=ttgl.float32, layout=layout_full)

    smem = ttgl.allocate_shared_memory(ttgl.float32, [M, N], layout=shared_layout)
    smem.store(init_data)

    # Create 2D subslice
    smem_slice = smem.slice(SLICE_M_OFFSET, SLICE_M, dim=0).slice(SLICE_N_OFFSET, SLICE_N, dim=1)

    # Load indices and values for scattering within the slice
    slice_offs_m = ttgl.arange(0, SLICE_M, layout=ttgl.SliceLayout(1, layout_slice))[:, None]
    slice_offs_n = ttgl.arange(0, SLICE_N, layout=ttgl.SliceLayout(0, layout_slice))[None, :]
    idx_offs = slice_offs_m * SLICE_N + slice_offs_n
    indices = ttgl.load(indices_ptr + idx_offs)
    values = ttgl.load(values_ptr + idx_offs)

    # Scatter along axis 0: smem_slice[indices[i, j], j] = values[i, j]
    smem_slice.scatter(values, indices, axis=0)

    # Load back full matrix
    result = smem.load(layout=layout_full)
    ttgl.store(output_ptr + full_offs, result)


@pytest.mark.parametrize("M,N,slice_m_offset,slice_n_offset,slice_m,slice_n", [
    # Offset must be a multiple of tile (slice) size for each dimension
    (64, 64, 48, 16, 16, 16),  # offset 48 % 16 == 0, offset 16 % 16 == 0
    (64, 64, 32, 48, 32, 16),  # offset 32 % 32 == 0, offset 48 % 16 == 0
])
def test_scatter_subslice_2d(M, N, slice_m_offset, slice_n_offset, slice_m, slice_n):
    """Test scattering to a 2D subsliced shared memory descriptor."""
    device = torch.device("cuda")

    # Create indices (reverse pattern for scatter)
    indices = torch.arange(slice_n, dtype=torch.int32, device=device)[None, :].expand(slice_m, slice_n)
    indices = (slice_m - 1 - indices - torch.arange(slice_m, dtype=torch.int32, device=device)[:, None]) % slice_m

    # Create values to scatter
    values = torch.arange(slice_m * slice_n, dtype=torch.float32, device=device).reshape(slice_m, slice_n) + 100.0

    output = torch.zeros((M, N), dtype=torch.float32, device=device)

    # Expected: -1 everywhere, then scatter into the subslice region
    expected = torch.full((M, N), -1.0, dtype=torch.float32, device=device)
    subslice_expected = torch.zeros((slice_m, slice_n), dtype=torch.float32, device=device)
    subslice_expected.scatter_(0, indices.long(), values)
    expected[slice_m_offset:slice_m_offset + slice_m, slice_n_offset:slice_n_offset + slice_n] = subslice_expected

    # Layouts
    layout_full = ttgl.BlockedLayout(
        size_per_thread=[1, 1],
        threads_per_warp=[THREADS_PER_WARP // 4, 4],
        warps_per_cta=[1, 1],
        order=[1, 0],
    )
    layout_slice = ttgl.BlockedLayout(
        size_per_thread=[1, 1],
        threads_per_warp=[THREADS_PER_WARP // 4, 4],
        warps_per_cta=[1, 1],
        order=[1, 0],
    )
    shared_layout = ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])

    scatter_subslice_2d_kernel[(1, )](
        indices,
        values,
        output,
        M=M,
        N=N,
        SLICE_M_OFFSET=slice_m_offset,
        SLICE_N_OFFSET=slice_n_offset,
        SLICE_M=slice_m,
        SLICE_N=slice_n,
        layout_full=layout_full,
        layout_slice=layout_slice,
        shared_layout=shared_layout,
        num_warps=1,
    )

    torch.testing.assert_close(output, expected)


# =============================================================================
# Padded Layout Tests
# =============================================================================


@gluon.jit
def gather_padded_kernel(
    matrix_ptr,
    indices_ptr,
    output_ptr,
    M: ttgl.constexpr,
    N: ttgl.constexpr,
    layout_2d: ttgl.constexpr,
    padded_layout: ttgl.constexpr,
):
    """Gather from shared memory with a padded layout."""
    # Load matrix into padded shared memory
    offs_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, layout_2d))[:, None]
    offs_n = ttgl.arange(0, N, layout=ttgl.SliceLayout(0, layout_2d))[None, :]
    in_offs = offs_m * N + offs_n
    in_data = ttgl.load(matrix_ptr + in_offs)

    smem = ttgl.allocate_shared_memory(ttgl.float32, [M, N], layout=padded_layout)
    smem.store(in_data)

    # Load indices
    indices = ttgl.load(indices_ptr + in_offs)

    # Gather along axis 0
    gathered = smem.gather(indices, axis=0)

    ttgl.store(output_ptr + in_offs, gathered)


@pytest.mark.parametrize("M,N", [(64, 64)])
@pytest.mark.parametrize("interval_pairs", [[[32, 4]], [[16, 4]], [[16, 4], [64, 8]]])
@pytest.mark.parametrize("order", [[0, 1], [1, 0]])
def test_gather_padded(M, N, interval_pairs, order):
    """Test gathering from shared memory with a padded layout."""
    device = torch.device("cuda")

    # Create input matrix
    matrix = torch.arange(M * N, dtype=torch.float32, device=device).reshape(M, N)

    # Create indices for gather along axis 0
    indices = torch.arange(N, dtype=torch.int32, device=device)[None, :].expand(M, N)
    indices = (indices + torch.arange(M, dtype=torch.int32, device=device)[:, None]) % M

    output = torch.zeros((M, N), dtype=torch.float32, device=device)

    # Expected: gather along axis 0
    expected = torch.gather(matrix, 0, indices.long())

    # Layouts
    layout_2d = ttgl.BlockedLayout(
        size_per_thread=[1, 1],
        threads_per_warp=[THREADS_PER_WARP // 4, 4],
        warps_per_cta=[1, 1],
        order=[1, 0],
    )
    padded_layout = ttgl.PaddedSharedLayout.with_identity_for(interval_pairs, [M, N], order)

    gather_padded_kernel[(1, )](
        matrix,
        indices,
        output,
        M=M,
        N=N,
        layout_2d=layout_2d,
        padded_layout=padded_layout,
        num_warps=1,
    )

    torch.testing.assert_close(output, expected)


@gluon.jit
def scatter_padded_kernel(
    indices_ptr,
    values_ptr,
    output_ptr,
    M: ttgl.constexpr,
    N: ttgl.constexpr,
    layout_2d: ttgl.constexpr,
    padded_layout: ttgl.constexpr,
):
    """Scatter to shared memory with a padded layout."""
    # Initialize padded shared memory with zeros
    offs_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, layout_2d))[:, None]
    offs_n = ttgl.arange(0, N, layout=ttgl.SliceLayout(0, layout_2d))[None, :]
    full_offs = offs_m * N + offs_n
    zeros = ttgl.zeros([M, N], ttgl.float32, layout=layout_2d)

    smem = ttgl.allocate_shared_memory(ttgl.float32, [M, N], layout=padded_layout)
    smem.store(zeros)

    # Load indices and values
    indices = ttgl.load(indices_ptr + full_offs)
    values = ttgl.load(values_ptr + full_offs)

    # Scatter along axis 0
    smem.scatter(values, indices, axis=0)

    # Load back
    result = smem.load(layout=layout_2d)
    ttgl.store(output_ptr + full_offs, result)


@pytest.mark.parametrize("M,N", [(64, 64)])
@pytest.mark.parametrize("interval_pairs", [[[32, 4]], [[16, 4]]])
@pytest.mark.parametrize("order", [[0, 1], [1, 0]])
def test_scatter_padded(M, N, interval_pairs, order):
    """Test scattering to shared memory with a padded layout."""
    device = torch.device("cuda")

    # Create indices (reverse pattern)
    indices = torch.arange(N, dtype=torch.int32, device=device)[None, :].expand(M, N)
    indices = (M - 1 - indices - torch.arange(M, dtype=torch.int32, device=device)[:, None]) % M

    # Create values
    values = torch.arange(M * N, dtype=torch.float32, device=device).reshape(M, N) + 100.0

    output = torch.zeros((M, N), dtype=torch.float32, device=device)

    # Expected: scatter along axis 0
    expected = torch.zeros((M, N), dtype=torch.float32, device=device)
    expected.scatter_(0, indices.long(), values)

    # Layouts
    layout_2d = ttgl.BlockedLayout(
        size_per_thread=[1, 1],
        threads_per_warp=[THREADS_PER_WARP // 4, 4],
        warps_per_cta=[1, 1],
        order=[1, 0],
    )
    padded_layout = ttgl.PaddedSharedLayout.with_identity_for(interval_pairs, [M, N], order)

    scatter_padded_kernel[(1, )](
        indices,
        values,
        output,
        M=M,
        N=N,
        layout_2d=layout_2d,
        padded_layout=padded_layout,
        num_warps=1,
    )

    torch.testing.assert_close(output, expected)


# =============================================================================
# Padded Layout with Subslice Tests
# =============================================================================


@gluon.jit
def gather_padded_subslice_kernel(
    matrix_ptr,
    indices_ptr,
    output_ptr,
    M: ttgl.constexpr,
    N: ttgl.constexpr,
    SLICE_M_OFFSET: ttgl.constexpr,
    SLICE_N_OFFSET: ttgl.constexpr,
    SLICE_M: ttgl.constexpr,
    SLICE_N: ttgl.constexpr,
    layout_full: ttgl.constexpr,
    layout_slice: ttgl.constexpr,
    padded_layout: ttgl.constexpr,
):
    """Gather from a subsliced padded shared memory descriptor."""
    # Load full matrix into padded shared memory
    offs_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, layout_full))[:, None]
    offs_n = ttgl.arange(0, N, layout=ttgl.SliceLayout(0, layout_full))[None, :]
    in_offs = offs_m * N + offs_n
    in_data = ttgl.load(matrix_ptr + in_offs)

    smem = ttgl.allocate_shared_memory(ttgl.float32, [M, N], layout=padded_layout)
    smem.store(in_data)

    # Create 2D subslice
    smem_slice = smem.slice(SLICE_M_OFFSET, SLICE_M, dim=0).slice(SLICE_N_OFFSET, SLICE_N, dim=1)

    # Load indices for gathering within the slice
    slice_offs_m = ttgl.arange(0, SLICE_M, layout=ttgl.SliceLayout(1, layout_slice))[:, None]
    slice_offs_n = ttgl.arange(0, SLICE_N, layout=ttgl.SliceLayout(0, layout_slice))[None, :]
    idx_offs = slice_offs_m * SLICE_N + slice_offs_n
    indices = ttgl.load(indices_ptr + idx_offs)

    # Gather along axis 0
    gathered = smem_slice.gather(indices, axis=0)

    ttgl.store(output_ptr + idx_offs, gathered)


@pytest.mark.parametrize("interval_pairs", [[[32, 4]], [[16, 4]]])
@pytest.mark.parametrize("order", [[0, 1], [1, 0]])
@pytest.mark.parametrize("slice_m_offset,slice_n_offset,slice_m,slice_n", [
    (48, 16, 16, 16),
    (32, 48, 32, 16),
    (48, 32, 16, 32),
])
def test_gather_padded_subslice(interval_pairs, order, slice_m_offset, slice_n_offset, slice_m, slice_n):
    """Test gathering from a subsliced padded shared memory descriptor."""
    M, N = 64, 64
    device = torch.device("cuda")

    # Create input matrix
    matrix = torch.arange(M * N, dtype=torch.float32, device=device).reshape(M, N)

    # Create indices for gather within the slice
    indices = torch.arange(slice_n, dtype=torch.int32, device=device)[None, :].expand(slice_m, slice_n)
    indices = (indices + torch.arange(slice_m, dtype=torch.int32, device=device)[:, None]) % slice_m

    output = torch.zeros((slice_m, slice_n), dtype=torch.float32, device=device)

    # Expected: gather from the subslice
    subslice = matrix[slice_m_offset:slice_m_offset + slice_m, slice_n_offset:slice_n_offset + slice_n]
    expected = torch.gather(subslice, 0, indices.long())

    # Layouts
    layout_full = ttgl.BlockedLayout(
        size_per_thread=[1, 1],
        threads_per_warp=[THREADS_PER_WARP // 4, 4],
        warps_per_cta=[1, 1],
        order=[1, 0],
    )
    layout_slice = ttgl.BlockedLayout(
        size_per_thread=[1, 1],
        threads_per_warp=[THREADS_PER_WARP // 4, 4],
        warps_per_cta=[1, 1],
        order=[1, 0],
    )
    padded_layout = ttgl.PaddedSharedLayout.with_identity_for(interval_pairs, [M, N], order)

    gather_padded_subslice_kernel[(1, )](
        matrix,
        indices,
        output,
        M=M,
        N=N,
        SLICE_M_OFFSET=slice_m_offset,
        SLICE_N_OFFSET=slice_n_offset,
        SLICE_M=slice_m,
        SLICE_N=slice_n,
        layout_full=layout_full,
        layout_slice=layout_slice,
        padded_layout=padded_layout,
        num_warps=1,
    )

    torch.testing.assert_close(output, expected)


@gluon.jit
def scatter_padded_subslice_kernel(
    indices_ptr,
    values_ptr,
    output_ptr,
    M: ttgl.constexpr,
    N: ttgl.constexpr,
    SLICE_M_OFFSET: ttgl.constexpr,
    SLICE_N_OFFSET: ttgl.constexpr,
    SLICE_M: ttgl.constexpr,
    SLICE_N: ttgl.constexpr,
    layout_full: ttgl.constexpr,
    layout_slice: ttgl.constexpr,
    padded_layout: ttgl.constexpr,
):
    """Scatter to a subsliced padded shared memory descriptor."""
    # Initialize padded shared memory with -1
    offs_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, layout_full))[:, None]
    offs_n = ttgl.arange(0, N, layout=ttgl.SliceLayout(0, layout_full))[None, :]
    full_offs = offs_m * N + offs_n
    init_data = ttgl.full([M, N], -1.0, dtype=ttgl.float32, layout=layout_full)

    smem = ttgl.allocate_shared_memory(ttgl.float32, [M, N], layout=padded_layout)
    smem.store(init_data)

    # Create 2D subslice
    smem_slice = smem.slice(SLICE_M_OFFSET, SLICE_M, dim=0).slice(SLICE_N_OFFSET, SLICE_N, dim=1)

    # Load indices and values for scattering within the slice
    slice_offs_m = ttgl.arange(0, SLICE_M, layout=ttgl.SliceLayout(1, layout_slice))[:, None]
    slice_offs_n = ttgl.arange(0, SLICE_N, layout=ttgl.SliceLayout(0, layout_slice))[None, :]
    idx_offs = slice_offs_m * SLICE_N + slice_offs_n
    indices = ttgl.load(indices_ptr + idx_offs)
    values = ttgl.load(values_ptr + idx_offs)

    # Scatter along axis 0
    smem_slice.scatter(values, indices, axis=0)

    # Load back full matrix
    result = smem.load(layout=layout_full)
    ttgl.store(output_ptr + full_offs, result)


@pytest.mark.parametrize("interval_pairs", [[[32, 4]], [[16, 4]]])
@pytest.mark.parametrize("order", [[0, 1], [1, 0]])
@pytest.mark.parametrize("slice_m_offset,slice_n_offset,slice_m,slice_n", [
    (48, 16, 16, 16),
    (32, 48, 32, 16),
])
def test_scatter_padded_subslice(interval_pairs, order, slice_m_offset, slice_n_offset, slice_m, slice_n):
    """Test scattering to a subsliced padded shared memory descriptor."""
    M, N = 64, 64
    device = torch.device("cuda")

    # Create indices (reverse pattern)
    indices = torch.arange(slice_n, dtype=torch.int32, device=device)[None, :].expand(slice_m, slice_n)
    indices = (slice_m - 1 - indices - torch.arange(slice_m, dtype=torch.int32, device=device)[:, None]) % slice_m

    # Create values
    values = torch.arange(slice_m * slice_n, dtype=torch.float32, device=device).reshape(slice_m, slice_n) + 100.0

    output = torch.zeros((M, N), dtype=torch.float32, device=device)

    # Expected: -1 everywhere, then scatter into the subslice region
    expected = torch.full((M, N), -1.0, dtype=torch.float32, device=device)
    subslice_expected = torch.zeros((slice_m, slice_n), dtype=torch.float32, device=device)
    subslice_expected.scatter_(0, indices.long(), values)
    expected[slice_m_offset:slice_m_offset + slice_m, slice_n_offset:slice_n_offset + slice_n] = subslice_expected

    # Layouts
    layout_full = ttgl.BlockedLayout(
        size_per_thread=[1, 1],
        threads_per_warp=[THREADS_PER_WARP // 4, 4],
        warps_per_cta=[1, 1],
        order=[1, 0],
    )
    layout_slice = ttgl.BlockedLayout(
        size_per_thread=[1, 1],
        threads_per_warp=[THREADS_PER_WARP // 4, 4],
        warps_per_cta=[1, 1],
        order=[1, 0],
    )
    padded_layout = ttgl.PaddedSharedLayout.with_identity_for(interval_pairs, [M, N], order)

    scatter_padded_subslice_kernel[(1, )](
        indices,
        values,
        output,
        M=M,
        N=N,
        SLICE_M_OFFSET=slice_m_offset,
        SLICE_N_OFFSET=slice_n_offset,
        SLICE_M=slice_m,
        SLICE_N=slice_n,
        layout_full=layout_full,
        layout_slice=layout_slice,
        padded_layout=padded_layout,
        num_warps=1,
    )

    torch.testing.assert_close(output, expected)


# --- TMEM Load with Reduction Tests ---


@gluon.jit
def tmem_reduction_kernel(
    in_ptr,
    out_ptr,
    red_ptr,
    M: ttgl.constexpr,
    N: ttgl.constexpr,
    RED_OP: ttgl.constexpr,
    USE_ABS: ttgl.constexpr,
    PROPAGATE_NAN: ttgl.constexpr,
    num_warps: ttgl.constexpr,
):
    """Kernel to test TMEM load with hardware reduction."""
    global_memory_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [1, 32], [1, num_warps], [1, 0])
    global_memory_layout_1d: ttgl.constexpr = ttgl.BlockedLayout([1], [32], [num_warps], [0])

    # Offsets for 2D tensor
    offs_m = ttgl.arange(0, M, ttgl.SliceLayout(1, global_memory_layout))
    offs_n = ttgl.arange(0, N, ttgl.SliceLayout(0, global_memory_layout))
    offs_2d = offs_m[:, None] * N + offs_n[None, :]

    # Load input from global memory
    input_data = ttgl.load(in_ptr + offs_2d)

    # Setup TMEM layout - blockN must match N for single reduction value per row
    tmem_layout: ttgl.constexpr = TensorMemoryLayout(block=(128, N), col_stride=1,  # packed for f32
                                                     )

    # Allocate TMEM
    tmem = allocate_tensor_memory(
        element_ty=in_ptr.dtype.element_ty,
        shape=[M, N],
        layout=tmem_layout,
    )

    # Get register layout for TMEM access
    tmem_reg_layout: ttgl.constexpr = get_tmem_reg_layout(
        in_ptr.dtype.element_ty,
        (M, N),
        tmem_layout,
        num_warps=num_warps,
    )

    # Store input to TMEM
    input_data = ttgl.convert_layout(input_data, tmem_reg_layout)
    tmem.store(input_data)

    # Load from TMEM with reduction
    if RED_OP == "min":
        output, reduced = tmem.load_min(tmem_reg_layout, abs=USE_ABS, propagate_nan=PROPAGATE_NAN)
    elif RED_OP == "max":
        output, reduced = tmem.load_max(tmem_reg_layout, abs=USE_ABS, propagate_nan=PROPAGATE_NAN)

    # Store full output
    output = ttgl.convert_layout(output, global_memory_layout)
    ttgl.store(out_ptr + offs_2d, output)

    # Store reduced output (1D tensor of shape [M])
    offs_1d = ttgl.arange(0, M, global_memory_layout_1d)
    reduced = ttgl.convert_layout(reduced, global_memory_layout_1d)
    ttgl.store(red_ptr + offs_1d, reduced)


@pytest.mark.skipif(not is_blackwell_ultra(), reason="Requires Blackwell Ultra")
@pytest.mark.parametrize("red_op", ["min", "max"])
@pytest.mark.parametrize("use_abs", [False, True])
@pytest.mark.parametrize("propagate_nan", [tl.PropagateNan.NONE, tl.PropagateNan.ALL])
@pytest.mark.parametrize(
    "M, N, num_warps",
    [(128, 32, 4), (128, 64, 4), (128, 128, 4), (128, 256, 4), (256, 128, 8)],
)
def test_tmem_reduction(red_op, use_abs, propagate_nan, M, N, num_warps):
    """Test TMEM load with hardware reduction on MxN tile

    Note: With M=128, only 4 warps can be used (warpsPerCTA=[4,1]) since all
    warps must fit in the M dimension for reduction. 8 warps would require
    M=256 (8*32=256). The N=256 case tests partial reduction combining where
    4 hardware reductions are combined via llvm.minnum/maxnum.
    """

    # Create test input with some negative values
    input_tensor = torch.randn(M, N, dtype=torch.float32, device="cuda")

    # Inject NaN for testing if needed
    use_nan = False if propagate_nan == tl.PropagateNan.NONE else True
    if use_nan:
        input_tensor[10, 5] = float("nan")
        input_tensor[50, 15] = float("nan")

    # Output tensors
    output = torch.empty_like(input_tensor)
    red_output = torch.empty(M, dtype=torch.float32, device="cuda")

    # Run kernel
    tmem_reduction_kernel[(1, )](
        input_tensor,
        output,
        red_output,
        M,
        N,
        red_op,
        use_abs,
        propagate_nan,
        num_warps=num_warps,
    )

    # Verify full output matches input (tmem store/load roundtrip)
    # Use equal_nan=True when we have NaN values in the input
    torch.testing.assert_close(input_tensor, output, atol=0, rtol=0, equal_nan=use_nan)

    # Compute expected reduction
    ref_input = torch.abs(input_tensor) if use_abs else input_tensor
    torch_red = torch.min if red_op == "min" else torch.max
    expected_red = torch_red(ref_input, dim=1).values

    # Verify reduction output
    # Use equal_nan=True when testing NaN propagation
    torch.testing.assert_close(expected_red, red_output, atol=1e-5, rtol=1e-5, equal_nan=use_nan)
