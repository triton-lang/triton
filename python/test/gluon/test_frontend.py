import expecttest
import torch
import pytest

from triton import knobs
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl
from triton.experimental.gluon.language.nvidia import blackwell
from triton.experimental.gluon.language.nvidia.blackwell import mbarrier, tma
from triton._filecheck import filecheck_test
import triton.language as tl
from triton._internal_testing import is_cuda
from triton.tools.tensor_descriptor import TensorDescriptor


@gluon.jit
def convert_layout_kernel(XBLOCK: ttgl.constexpr, layout_a: ttgl.constexpr, layout_b: ttgl.constexpr):
    x = ttgl.arange(0, XBLOCK, layout=layout_a)
    res = ttgl.convert_layout(x, layout_b)  # noqa: F841


def test_convert_layout(fresh_knobs):
    knobs.compilation.disable_line_info = True

    layout_a = ttgl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32], warps_per_cta=[4], order=[0])
    layout_b = ttgl.SliceLayout(
        1, ttgl.BlockedLayout(size_per_thread=[1, 1], threads_per_warp=[1, 32], warps_per_cta=[1, 4], order=[1, 0]))
    h = convert_layout_kernel.warmup(128, layout_a, layout_b, num_warps=layout_a.warps_per_cta[0], grid=(1, ))
    expecttest.assert_expected_inline(
        h.asm["source"], """\
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func public @convert_layout_kernel() attributes {noinline = false} {
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked> loc(#loc)
    %1 = ttg.convert_layout %0 : tensor<128xi32, #blocked> -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> loc(#loc)
    tt.return loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
""")
    expecttest.assert_expected_inline(
        h.asm["ttgir"], """\
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func public @convert_layout_kernel() attributes {noinline = false} {
    tt.return loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
""")


@gluon.jit
def shared_memory_kernel(XBLOCK: ttgl.constexpr, YBLOCK: ttgl.constexpr, layout_a: ttgl.constexpr,
                         layout_b: ttgl.constexpr, smem_layout: ttgl.constexpr):
    unused = ttgl.allocate_shared_memory(ttgl.int32, [XBLOCK, YBLOCK], smem_layout)
    a = ttgl.full([XBLOCK, YBLOCK], 0, ttgl.int32, layout_a)
    mem = ttgl.allocate_shared_memory(ttgl.int32, a.shape, smem_layout, a)
    b = mem.load(layout_b)  # noqa: F841
    mem.store(a)
    unused._keep_alive()


def test_shared_memory(fresh_knobs):
    knobs.compilation.disable_line_info = True

    layout_a = ttgl.BlockedLayout(size_per_thread=[1, 1], threads_per_warp=[1, 32], warps_per_cta=[4, 1], order=[1, 0])
    layout_b = ttgl.BlockedLayout(size_per_thread=[1, 4], threads_per_warp=[1, 32], warps_per_cta=[4, 1], order=[1, 0])
    smem_layout = ttgl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=32, rank=2)
    h = shared_memory_kernel.warmup(8, 32, layout_a, layout_b, smem_layout, num_warps=layout_a.warps_per_cta[0],
                                    grid=(1, ))
    expecttest.assert_expected_inline(
        h.asm["source"], """\
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func public @shared_memory_kernel() attributes {noinline = false} {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable> loc(#loc)
    %c0_i32 = arith.constant 0 : i32 loc(#loc)
    %cst = arith.constant dense<0> : tensor<8x32xi32, #blocked> loc(#loc)
    %1 = ttg.local_alloc %cst : (tensor<8x32xi32, #blocked>) -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable> loc(#loc)
    %2 = ttg.local_load %1 : !ttg.memdesc<8x32xi32, #shared, #smem, mutable> -> tensor<8x32xi32, #blocked1> loc(#loc)
    ttg.local_store %cst, %1 : tensor<8x32xi32, #blocked> -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable> loc(#loc)
    ttg.local_dealloc %0 : !ttg.memdesc<8x32xi32, #shared, #smem, mutable> loc(#loc)
    tt.return loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
""")


@gluon.jit
def tensor_memory_kernel(layout: ttgl.constexpr, tmem_layout: ttgl.constexpr):
    XBLOCK: ttgl.constexpr = tmem_layout.block[0]
    YBLOCK: ttgl.constexpr = tmem_layout.block[1]
    a = ttgl.full([XBLOCK, YBLOCK], 0, ttgl.int32, layout)
    _ = ttgl.nvidia.blackwell.allocate_tensor_memory(ttgl.int32, a.shape, tmem_layout)
    mem = ttgl.nvidia.blackwell.allocate_tensor_memory(ttgl.int32, a.shape, tmem_layout, a)
    b = mem.load(layout)  # noqa: F841
    mem.store(a)
    slice1 = mem.subslice(0, YBLOCK // 2)  # noqa: F841
    slice2 = mem.subslice(YBLOCK // 2, YBLOCK // 2)  # noqa: F841


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] != 10,
                    reason="Requires blackwell tensor cores")
def test_tensor_memory(fresh_knobs):
    knobs.compilation.disable_line_info = True

    layout = ttgl.BlockedLayout(size_per_thread=[1, 64], threads_per_warp=[32, 1], warps_per_cta=[4, 1], order=[0, 1])
    tmem_layout = ttgl.nvidia.blackwell.TensorMemoryLayout(block=[128, 128], unpacked=True)
    h = tensor_memory_kernel.warmup(layout, tmem_layout, num_warps=4, grid=(1, ))
    expecttest.assert_expected_inline(
        h.asm["source"], """\
#blocked = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func public @tensor_memory_kernel() attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32 loc(#loc)
    %cst = arith.constant dense<0> : tensor<128x128xi32, #blocked> loc(#loc)
    %result = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xi32, #tmem, #ttng.tensor_memory, mutable> loc(#loc)
    %result_0 = ttng.tmem_alloc %cst : (tensor<128x128xi32, #blocked>) -> !ttg.memdesc<128x128xi32, #tmem, #ttng.tensor_memory, mutable> loc(#loc)
    %result_1 = ttng.tmem_load %result_0 : !ttg.memdesc<128x128xi32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xi32, #blocked> loc(#loc)
    %true = arith.constant true loc(#loc)
    ttng.tmem_store %cst, %result_0, %true : tensor<128x128xi32, #blocked> -> !ttg.memdesc<128x128xi32, #tmem, #ttng.tensor_memory, mutable> loc(#loc)
    %0 = ttng.tmem_subslice %result_0 {N = 0 : i32} : !ttg.memdesc<128x128xi32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xi32, #tmem, #ttng.tensor_memory, mutable, 128x128> loc(#loc)
    %1 = ttng.tmem_subslice %result_0 {N = 64 : i32} : !ttg.memdesc<128x128xi32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xi32, #tmem, #ttng.tensor_memory, mutable, 128x128> loc(#loc)
    tt.return loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
""")


@gluon.jit
def warp_specialize_default(a, b):
    return b, a


@gluon.jit
def warp_specialize_worker0(a, b):
    pass


@gluon.jit
def warp_specialize_worker1(a, b):
    pass


@tl.core._aggregate
class Pair:
    first: tl.tensor
    second: tl.tensor

    def __init__(self, first, second):
        self.first = first
        self.second = second


@gluon.jit
def anchor(x):
    pass


@filecheck_test
@gluon.jit
def test_warp_specialize():
    # CHECK-LABEL: tt.func public @test_warp_specialize
    # CHECK-NEXT:    [[A:%.*]] = tt.make_range {end = 1 : i32, start = 0 : i32}
    # CHECK-NEXT:    [[B:%.*]] = tt.make_range {end = 2 : i32, start = 0 : i32}
    # CHECK-NEXT:    [[C:%.*]] = tt.make_range {end = 4 : i32, start = 0 : i32}
    # CHECK-NEXT:    [[OUTS:%.*]]:3 = ttg.warp_specialize([[A]], [[B]], [[C]]) {{.*}}requestedRegisters = array<i32: 24, 48>
    # CHECK-NEXT:    default {
    # CHECK-NEXT:      [[RESULTS:%.*]]:3 = tt.call @"warp_specialize_default{{.*}}"([[A]], [[B]], [[C]])
    # CHECK-NEXT:      warp_yield [[RESULTS]]#0, [[RESULTS]]#1, [[RESULTS]]#2
    # CHECK-NEXT:    }
    # CHECK-NEXT:    partition0(%arg0: tensor<1xi32>, %arg1: tensor<2xi32>, %arg2: tensor<4xi32>) num_warps(4) {
    # CHECK-NEXT:      call @"warp_specialize_worker0{{.*}}"(%arg0, %arg1, %arg2)
    # CHECK-NEXT:      warp_return
    # CHECK-NEXT:    }
    # CHECK-NEXT:    partition1(%arg0: tensor<1xi32>, %arg1: tensor<2xi32>, %arg2: tensor<4xi32>) num_warps(4) {
    # CHECK-NEXT:      call @"warp_specialize_worker1{{.*}}"(%arg0, %arg1, %arg2)
    # CHECK-NEXT:      warp_return
    # CHECK-NEXT:    }
    # CHECK-NEXT:    call @anchor{{.*}}([[OUTS]]#0)
    # CHECK-NEXT:    call @"anchor{{.*}}"([[OUTS]]#1, [[OUTS]]#2)
    pair = Pair(tl.arange(0, 1), tl.arange(0, 2))
    a, b = ttgl.warp_specialize((pair, tl.arange(0, 4)), warp_specialize_default,
                                [warp_specialize_worker0, warp_specialize_worker1], [4, 4], [24, 48])
    anchor(a)
    anchor(b)


@gluon.jit
def mbarrier_kernel():
    bar = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    mbarrier.expect(bar, 4)
    mbarrier.arrive(bar, 1)
    phase = 0
    mbarrier.wait(bar, phase, deps=[bar])
    mbarrier.invalidate(bar)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires hopper or newer")
def test_mbarrier(fresh_knobs):
    knobs.compilation.disable_line_info = True

    h = mbarrier_kernel.warmup(grid=(1, ))
    expecttest.assert_expected_inline(
        h.asm["source"], """\
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func public @mbarrier_kernel() attributes {noinline = false} {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable> loc(#loc)
    ttng.init_barrier %0, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable> loc(#loc)
    %true = arith.constant true loc(#loc)
    ttng.barrier_expect %0, 4, %true : !ttg.memdesc<1xi64, #shared, #smem, mutable> loc(#loc)
    %true_0 = arith.constant true loc(#loc)
    ttng.arrive_barrier %0, 1, %true_0 : !ttg.memdesc<1xi64, #shared, #smem, mutable> loc(#loc)
    %c0_i32 = arith.constant 0 : i32 loc(#loc)
    %true_1 = arith.constant true loc(#loc)
    ttng.wait_barrier %0, %c0_i32, %true_1 deps %0 : !ttg.memdesc<1xi64, #shared, #smem, mutable>, !ttg.memdesc<1xi64, #shared, #smem, mutable> loc(#loc)
    ttng.inval_barrier %0 : !ttg.memdesc<1xi64, #shared, #smem, mutable> loc(#loc)
    tt.return loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
""")


@gluon.jit
def tcgen05_mma_kernel(nvmma_layout: ttgl.constexpr, acc_layout: ttgl.constexpr):
    a = ttgl.allocate_shared_memory(ttgl.float16, [128, 128], nvmma_layout)
    b = ttgl.allocate_shared_memory(ttgl.float16, [128, 128], nvmma_layout)
    acc = blackwell.allocate_tensor_memory(ttgl.float16, [128, 128], acc_layout)
    blackwell.tcgen05_mma(a, b, acc)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] != 10,
                    reason="Requires blackwell tensor core")
def test_tcgen05_mma(fresh_knobs):
    knobs.compilation.disable_line_info = True

    nvmma_layout = ttgl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2)
    acc_layout = blackwell.TensorMemoryLayout([128, 128], unpacked=True)

    h = tcgen05_mma_kernel.warmup(nvmma_layout, acc_layout, grid=(1, ))
    expecttest.assert_expected_inline(
        h.asm["source"], """\
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func public @tcgen05_mma_kernel() attributes {noinline = false} {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable> loc(#loc)
    %1 = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable> loc(#loc)
    %result = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable> loc(#loc)
    %true = arith.constant true loc(#loc)
    %true_0 = arith.constant true loc(#loc)
    %2 = ttng.tc_gen5_mma %0, %1, %result[], %true, %true_0 : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable> loc(#loc)
    tt.return loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
""")


@gluon.jit
def async_tma_kernel(input_desc, XBLOCK: ttgl.constexpr, smem_layout: ttgl.constexpr):
    smem = ttgl.allocate_shared_memory(ttgl.float16, [XBLOCK, XBLOCK], smem_layout)
    bar = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)

    tma.async_copy_global_to_local(input_desc, [0, 0], bar, smem)
    mbarrier.expect(bar, XBLOCK * XBLOCK * ttgl.float16.primitive_bitwidth // 8)
    mbarrier.wait(bar, 0)

    mbarrier.invalidate(bar)

    tma.async_copy_local_to_global(input_desc, [0, 0], smem)
    tma.store_wait(0)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="TMA requires at least Hopper")
def test_async_tma(fresh_knobs):
    knobs.compilation.disable_line_info = True

    input = torch.randn((1024, 1024), device="cuda", dtype=torch.float16)
    XBLOCK = 128
    input_desc = TensorDescriptor.from_tensor(input, [XBLOCK, XBLOCK])
    shared_layout = ttgl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2)

    h = async_tma_kernel.warmup(input_desc, XBLOCK, shared_layout, grid=(1, ), num_warps=4)
    expecttest.assert_expected_inline(h.asm["source"], """\
#loc = loc("/root/code/triton/python/test/gluon/test_frontend.py":273:0)
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func public @async_tma_kernel(%arg0: !tt.tensordesc<tensor<128x128xf16>> loc("/root/code/triton/python/test/gluon/test_frontend.py":273:0), %arg1: i32 loc("/root/code/triton/python/test/gluon/test_frontend.py":273:0), %arg2: i32 loc("/root/code/triton/python/test/gluon/test_frontend.py":273:0), %arg3: i64 loc("/root/code/triton/python/test/gluon/test_frontend.py":273:0), %arg4: i64 loc("/root/code/triton/python/test/gluon/test_frontend.py":273:0)) attributes {noinline = false} {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable> loc(#loc1)
    %1 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable> loc(#loc2)
    ttng.init_barrier %1, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable> loc(#loc3)
    %c0_i32 = arith.constant 0 : i32 loc(#loc4)
    %c0_i32_0 = arith.constant 0 : i32 loc(#loc4)
    %true = arith.constant true loc(#loc4)
    %2 = ttng.tensor_desc_to_tma_ptr %arg0 : !tt.tensordesc<tensor<128x128xf16>> to !tt.ptr<i8> loc(#loc4)
    ttng.async_tma_copy_global_to_local %2[%c0_i32, %c0_i32_0] %0, %1, %true : !tt.ptr<i8>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable> loc(#loc4)
    %true_1 = arith.constant true loc(#loc5)
    ttng.barrier_expect %1, 32768, %true_1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable> loc(#loc5)
    %c0_i32_2 = arith.constant 0 : i32 loc(#loc6)
    %true_3 = arith.constant true loc(#loc6)
    ttng.wait_barrier %1, %c0_i32_2, %true_3 : !ttg.memdesc<1xi64, #shared1, #smem, mutable> loc(#loc6)
    ttng.inval_barrier %1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable> loc(#loc7)
    %c0_i32_4 = arith.constant 0 : i32 loc(#loc8)
    %c0_i32_5 = arith.constant 0 : i32 loc(#loc8)
    %3 = ttng.tensor_desc_to_tma_ptr %arg0 : !tt.tensordesc<tensor<128x128xf16>> to !tt.ptr<i8> loc(#loc8)
    ttng.async_tma_copy_local_to_global %3[%c0_i32_4, %c0_i32_5] %0 : !tt.ptr<i8>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable> loc(#loc8)
    ttng.async_tma_store_wait {pendings = 0 : i32} loc(#loc9)
    tt.return loc(#loc10)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("/root/code/triton/python/test/gluon/test_frontend.py":274:71)
#loc2 = loc("/root/code/triton/python/test/gluon/test_frontend.py":275:55)
#loc3 = loc("/root/code/triton/python/test/gluon/test_frontend.py":276:18)
#loc4 = loc("/root/code/triton/python/test/gluon/test_frontend.py":278:60)
#loc5 = loc("/root/code/triton/python/test/gluon/test_frontend.py":279:25)
#loc6 = loc("/root/code/triton/python/test/gluon/test_frontend.py":280:23)
#loc7 = loc("/root/code/triton/python/test/gluon/test_frontend.py":282:24)
#loc8 = loc("/root/code/triton/python/test/gluon/test_frontend.py":284:55)
#loc9 = loc("/root/code/triton/python/test/gluon/test_frontend.py":285:19)
#loc10 = loc("/root/code/triton/python/test/gluon/test_frontend.py":285:4)
""")


@gluon.jit
def async_tma_blackwell_kernel(input_desc, XBLOCK: ttgl.constexpr, smem_layout: ttgl.constexpr):
    smem = ttgl.allocate_shared_memory(ttgl.float16, [XBLOCK, XBLOCK], smem_layout)
    bar = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)

    offset_layout: tl.constexpr = ttgl.BlockedLayout([1, 4], [32, 1], [1, 4], [1, 0])
    x_offsets = ttgl.arange(0, XBLOCK, layout=ttgl.SliceLayout(0, offset_layout))
    tma.async_gather(input_desc, x_offsets, 0, bar, smem)
    mbarrier.expect(bar, XBLOCK * XBLOCK * ttgl.float16.primitive_bitwidth // 8)
    mbarrier.wait(bar, 0)

    mbarrier.invalidate(bar)

    tma.async_scatter(input_desc, x_offsets, 0, smem)
    tma.store_wait(0)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] != 10, reason="Requires Blackwell")
def test_async_tma_blackwell(fresh_knobs):
    knobs.compilation.disable_line_info = True

    input = torch.randn((1024, 1024), device="cuda", dtype=torch.float16)
    XBLOCK = 128
    input_desc = TensorDescriptor.from_tensor(input, [1, XBLOCK])
    shared_layout = ttgl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2)

    h = async_tma_blackwell_kernel.warmup(input_desc, XBLOCK, shared_layout, grid=(1, ), num_warps=4)
    expecttest.assert_expected_inline(h.asm["source"], """\
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [1, 0]}>
#loc = loc("/root/code/triton/python/test/gluon/test_frontend.py":300:0)
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func public @async_tma_blackwell_kernel(%arg0: !tt.tensordesc<tensor<1x128xf16>> loc("/root/code/triton/python/test/gluon/test_frontend.py":300:0), %arg1: i32 loc("/root/code/triton/python/test/gluon/test_frontend.py":300:0), %arg2: i32 loc("/root/code/triton/python/test/gluon/test_frontend.py":300:0), %arg3: i64 loc("/root/code/triton/python/test/gluon/test_frontend.py":300:0), %arg4: i64 loc("/root/code/triton/python/test/gluon/test_frontend.py":300:0)) attributes {noinline = false} {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable> loc(#loc1)
    %1 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable> loc(#loc2)
    ttng.init_barrier %1, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable> loc(#loc3)
    %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc4)
    %true = arith.constant true loc(#loc5)
    %c0_i32 = arith.constant 0 : i32 loc(#loc5)
    %3 = ttng.tensor_desc_to_tma_ptr %arg0 : !tt.tensordesc<tensor<1x128xf16>> to !tt.ptr<i8> loc(#loc5)
    ttng.async_tma_gather %3[%2, %c0_i32] %0, %1, %true : !tt.ptr<i8>, tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, i32, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, i1 loc(#loc5)
    %true_0 = arith.constant true loc(#loc6)
    ttng.barrier_expect %1, 32768, %true_0 : !ttg.memdesc<1xi64, #shared1, #smem, mutable> loc(#loc6)
    %c0_i32_1 = arith.constant 0 : i32 loc(#loc7)
    %true_2 = arith.constant true loc(#loc7)
    ttng.wait_barrier %1, %c0_i32_1, %true_2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable> loc(#loc7)
    ttng.inval_barrier %1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable> loc(#loc8)
    %4 = ttng.tensor_desc_to_tma_ptr %arg0 : !tt.tensordesc<tensor<1x128xf16>> to !tt.ptr<i8> loc(#loc9)
    %c0_i32_3 = arith.constant 0 : i32 loc(#loc9)
    ttng.async_tma_scatter %4[%2, %c0_i32_3] %0 : !tt.ptr<i8>, tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, i32, !ttg.memdesc<128x128xf16, #shared, #smem, mutable> loc(#loc9)
    ttng.async_tma_store_wait {pendings = 0 : i32} loc(#loc10)
    tt.return loc(#loc11)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("/root/code/triton/python/test/gluon/test_frontend.py":301:71)
#loc2 = loc("/root/code/triton/python/test/gluon/test_frontend.py":302:55)
#loc3 = loc("/root/code/triton/python/test/gluon/test_frontend.py":303:18)
#loc4 = loc("/root/code/triton/python/test/gluon/test_frontend.py":306:31)
#loc5 = loc("/root/code/triton/python/test/gluon/test_frontend.py":307:52)
#loc6 = loc("/root/code/triton/python/test/gluon/test_frontend.py":308:25)
#loc7 = loc("/root/code/triton/python/test/gluon/test_frontend.py":309:23)
#loc8 = loc("/root/code/triton/python/test/gluon/test_frontend.py":311:24)
#loc9 = loc("/root/code/triton/python/test/gluon/test_frontend.py":313:48)
#loc10 = loc("/root/code/triton/python/test/gluon/test_frontend.py":314:19)
#loc11 = loc("/root/code/triton/python/test/gluon/test_frontend.py":314:4)
""")
