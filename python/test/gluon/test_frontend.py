import expecttest
import torch
import pytest

from triton import knobs
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl
from triton.experimental.gluon.language.nvidia import blackwell
from triton.experimental.gluon.language.nvidia.blackwell import mbarrier, tma
from triton._filecheck import filecheck_test, run_parser
import triton.language as tl
from triton._internal_testing import is_cuda
from triton.tools.tensor_descriptor import TensorDescriptor
from triton.compiler.errors import CompilationError


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
    slice1 = mem.split(0, YBLOCK // 2)  # noqa: F841
    slice2 = mem.split(YBLOCK // 2, YBLOCK // 2)  # noqa: F841

    buffers = ttgl.nvidia.blackwell.allocate_tensor_memory(ttgl.float32, [2, XBLOCK, YBLOCK], tmem_layout)
    for i in range(2):
        buffers.subslice(i).load(layout)


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
    %result_2 = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable> loc(#loc)
    %c0_i32_3 = arith.constant 0 : i32 loc(#loc)
    %c2_i32 = arith.constant 2 : i32 loc(#loc)
    %c1_i32 = arith.constant 1 : i32 loc(#loc)
    %2 = arith.bitcast %c0_i32_3 : i32 to i32 loc(#loc)
    %3 = arith.bitcast %c2_i32 : i32 to i32 loc(#loc)
    %4 = arith.bitcast %c1_i32 : i32 to i32 loc(#loc)
    %5 = ub.poison : i32 loc(#loc)
    scf.for %arg0 = %2 to %3 step %4  : i32 {
      %c0_i32_4 = arith.constant 0 : i32 loc(#loc)
      %6 = ttg.memdesc_subview %result_2[%arg0, %c0_i32_4, %c0_i32_4] : !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> loc(#loc)
      %result_5 = ttng.tmem_load %6 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x128xf32, #blocked> loc(#loc)
    } loc(#loc)
    tt.return loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
""")


@gluon.jit
def shared_memory_subview_kernel(XBLOCK: ttgl.constexpr, layout: ttgl.constexpr, smem_layout: ttgl.constexpr):
    XHALF: tl.constexpr = XBLOCK // 2
    smem = ttgl.allocate_shared_memory(ttgl.int32, [XBLOCK, XBLOCK], smem_layout)
    view = smem.split(XHALF, XHALF, dim=1)
    value = view.load(layout)
    view = smem.split(XHALF, XHALF, dim=0)
    view.store(value.trans())


def test_shared_memory_subview(fresh_knobs):
    knobs.compilation.disable_line_info = True

    layout = ttgl.BlockedLayout(size_per_thread=[1, 1], threads_per_warp=[1, 32], warps_per_cta=[4, 1], order=[1, 0])
    smem_layout = ttgl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    h = shared_memory_subview_kernel.warmup(256, layout, smem_layout, num_warps=4, grid=(1, ))
    expecttest.assert_expected_inline(
        h.asm["source"], """\
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func public @shared_memory_subview_kernel() attributes {noinline = false} {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<256x256xi32, #shared, #smem, mutable> loc(#loc)
    %c0_i32 = arith.constant 0 : i32 loc(#loc)
    %c128_i32 = arith.constant 128 : i32 loc(#loc)
    %1 = ttg.memdesc_subview %0[%c0_i32, %c128_i32] : !ttg.memdesc<256x256xi32, #shared, #smem, mutable> -> !ttg.memdesc<256x128xi32, #shared, #smem, mutable, 256x256> loc(#loc)
    %2 = ttg.local_load %1 : !ttg.memdesc<256x128xi32, #shared, #smem, mutable, 256x256> -> tensor<256x128xi32, #blocked> loc(#loc)
    %c0_i32_0 = arith.constant 0 : i32 loc(#loc)
    %c128_i32_1 = arith.constant 128 : i32 loc(#loc)
    %3 = ttg.memdesc_subview %0[%c128_i32_1, %c0_i32_0] : !ttg.memdesc<256x256xi32, #shared, #smem, mutable> -> !ttg.memdesc<128x256xi32, #shared, #smem, mutable, 256x256> loc(#loc)
    %4 = tt.trans %2 {order = array<i32: 1, 0>} : tensor<256x128xi32, #blocked> -> tensor<128x256xi32, #blocked1> loc(#loc)
    ttg.local_store %4, %3 : tensor<128x256xi32, #blocked1> -> !ttg.memdesc<128x256xi32, #shared, #smem, mutable, 256x256> loc(#loc)
    tt.return loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
""")


@gluon.jit
def shared_memory_subslice_kernel(XBLOCK: ttgl.constexpr, layout: ttgl.constexpr, smem_layout: ttgl.constexpr):
    smem = ttgl.allocate_shared_memory(ttgl.int32, [4, XBLOCK], smem_layout)
    for i in range(4):
        smem.subslice(i).load(layout)


def test_shared_memory_subslice(fresh_knobs):
    knobs.compilation.disable_line_info = True

    layout = ttgl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32], warps_per_cta=[4], order=[0])
    smem_layout = ttgl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=32, rank=2)
    h = shared_memory_subslice_kernel.warmup(256, layout, smem_layout, num_warps=4, grid=(1, ))
    expecttest.assert_expected_inline(
        h.asm["source"], """\
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func public @shared_memory_subslice_kernel() attributes {noinline = false} {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<4x256xi32, #shared, #smem, mutable> loc(#loc)
    %c0_i32 = arith.constant 0 : i32 loc(#loc)
    %c4_i32 = arith.constant 4 : i32 loc(#loc)
    %c1_i32 = arith.constant 1 : i32 loc(#loc)
    %1 = arith.bitcast %c0_i32 : i32 to i32 loc(#loc)
    %2 = arith.bitcast %c4_i32 : i32 to i32 loc(#loc)
    %3 = arith.bitcast %c1_i32 : i32 to i32 loc(#loc)
    %4 = ub.poison : i32 loc(#loc)
    scf.for %arg0 = %1 to %2 step %3  : i32 {
      %c0_i32_0 = arith.constant 0 : i32 loc(#loc)
      %5 = ttg.memdesc_subview %0[%arg0, %c0_i32_0] : !ttg.memdesc<4x256xi32, #shared, #smem, mutable> -> !ttg.memdesc<256xi32, #shared, #smem, mutable, 4x256> loc(#loc)
      %6 = ttg.local_load %5 : !ttg.memdesc<256xi32, #shared, #smem, mutable, 4x256> -> tensor<256xi32, #blocked> loc(#loc)
    } loc(#loc)
    tt.return loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
""")


@gluon.jit
def shared_memory_cast_kernel():
    layout_a: ttgl.constexpr = ttgl.NVMMASharedLayout(swizzle_byte_width=64, transposed=False, element_bitwidth=8,
                                                      rank=2)
    layout_T: ttgl.constexpr = ttgl.NVMMASharedLayout(swizzle_byte_width=64, transposed=True, element_bitwidth=8,
                                                      rank=2)
    smem = ttgl.allocate_shared_memory(ttgl.int8, [256, 128], layout_a)
    smem.permute((1, 0), layout_T)

    layout_b: ttgl.constexpr = ttgl.NVMMASharedLayout(swizzle_byte_width=64, transposed=False, element_bitwidth=16,
                                                      rank=4, cta_order=[3, 2, 1, 0])
    layout_reshape: ttgl.constexpr = ttgl.NVMMASharedLayout(swizzle_byte_width=64, transposed=False,
                                                            element_bitwidth=16, rank=2)
    smem = ttgl.allocate_shared_memory(ttgl.float16, [32, 1, 4, 64], layout_b)
    smem.reshape((128, 64), layout_reshape)

    smem._reinterpret(ttgl.int8, [1024], ttgl.SwizzledSharedLayout(1, 1, 1, [0, 1]))


def test_shared_memory_cast(fresh_knobs):
    expecttest.assert_expected_inline(
        run_parser(shared_memory_cast_kernel).str_nodebug(), """\
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = true, elementBitWidth = 8}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16, CTAsPerCGA = [1, 1, 1, 1], CTASplitNum = [1, 1, 1, 1], CTAOrder = [3, 2, 1, 0]}>
#shared3 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#shared4 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
#smem = #ttg.shared_memory
module {
  tt.func public @shared_memory_cast_kernel() attributes {noinline = false} {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<256x128xi8, #shared, #smem, mutable>
    %1 = ttg.memdesc_trans %0 {order = array<i32: 1, 0>} : !ttg.memdesc<256x128xi8, #shared, #smem, mutable> -> !ttg.memdesc<128x256xi8, #shared1, #smem, mutable>
    %2 = ttg.local_alloc : () -> !ttg.memdesc<32x1x4x64xf16, #shared2, #smem, mutable>
    %3 = ttg.memdesc_reshape %2 : !ttg.memdesc<32x1x4x64xf16, #shared2, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared3, #smem, mutable, 32x1x4x64>
    %4 = ttg.memdesc_reinterpret %2 : !ttg.memdesc<32x1x4x64xf16, #shared2, #smem, mutable> -> !ttg.memdesc<1024xi8, #shared4, #smem, mutable>
    tt.return
  }
}
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
    # CHECK-LABEL: test_warp_specialize
    # CHECK-NEXT:    [[A:%.*]] = tt.make_range {end = 1 : i32, start = 0 : i32}
    # CHECK-NEXT:    [[B:%.*]] = tt.make_range {end = 2 : i32, start = 0 : i32}
    # CHECK-NEXT:    [[C:%.*]] = tt.make_range {end = 4 : i32, start = 0 : i32}
    # CHECK-NEXT:    [[OUTS:%.*]]:3 = ttg.warp_specialize([[A]], [[B]], [[C]]) {{.*}}requestedRegisters = array<i32: 24, 48>
    # CHECK-NEXT:    default {
    # CHECK-NEXT:      [[RESULTS:%.*]]:3 = tt.call @{{.*}}warp_specialize_default{{.*}}([[A]], [[B]], [[C]])
    # CHECK-NEXT:      warp_yield [[RESULTS]]#0, [[RESULTS]]#1, [[RESULTS]]#2
    # CHECK-NEXT:    }
    # CHECK-NEXT:    partition0(%arg0: tensor<1xi32>, %arg1: tensor<2xi32>, %arg2: tensor<4xi32>) num_warps(4) {
    # CHECK-NEXT:      call @{{.*}}warp_specialize_worker0{{.*}}(%arg0, %arg1, %arg2)
    # CHECK-NEXT:      warp_return
    # CHECK-NEXT:    }
    # CHECK-NEXT:    partition1(%arg0: tensor<1xi32>, %arg1: tensor<2xi32>, %arg2: tensor<4xi32>) num_warps(4) {
    # CHECK-NEXT:      call @{{.*}}warp_specialize_worker1{{.*}}(%arg0, %arg1, %arg2)
    # CHECK-NEXT:      warp_return
    # CHECK-NEXT:    }
    # CHECK-NEXT:    call @{{.*}}anchor{{.*}}([[OUTS]]#0)
    # CHECK-NEXT:    call @{{.*}}anchor{{.*}}([[OUTS]]#1, [[OUTS]]#2)
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

    tma.async_copy_global_to_shared(input_desc, [0, 0], bar, smem)
    mbarrier.expect(bar, XBLOCK * XBLOCK * ttgl.float16.primitive_bitwidth // 8)
    mbarrier.wait(bar, 0)

    mbarrier.invalidate(bar)

    tma.async_copy_shared_to_global(input_desc, [0, 0], smem)
    tma.store_wait(0)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="TMA requires at least Hopper")
def test_async_tma(fresh_knobs):
    knobs.compilation.disable_line_info = True

    input = torch.randn((1024, 1024), device="cuda", dtype=torch.float16)
    XBLOCK = 128
    input_desc = TensorDescriptor.from_tensor(input, [XBLOCK, XBLOCK])
    shared_layout = ttgl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2)

    h = async_tma_kernel.warmup(input_desc, XBLOCK, shared_layout, grid=(1, ), num_warps=4)
    expecttest.assert_expected_inline(
        h.asm["source"], """\
#loc = loc(unknown)
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func public @async_tma_kernel(%arg0: !tt.tensordesc<tensor<128x128xf16>> loc(unknown), %arg1: i32 loc(unknown), %arg2: i32 loc(unknown), %arg3: i64 loc(unknown), %arg4: i64 loc(unknown)) attributes {noinline = false} {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable> loc(#loc)
    %1 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable> loc(#loc)
    ttng.init_barrier %1, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable> loc(#loc)
    %c0_i32 = arith.constant 0 : i32 loc(#loc)
    %c0_i32_0 = arith.constant 0 : i32 loc(#loc)
    %true = arith.constant true loc(#loc)
    ttng.async_tma_copy_global_to_local %arg0[%c0_i32, %c0_i32_0] %0, %1, %true : !tt.tensordesc<tensor<128x128xf16>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable> loc(#loc)
    %true_1 = arith.constant true loc(#loc)
    ttng.barrier_expect %1, 32768, %true_1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable> loc(#loc)
    %c0_i32_2 = arith.constant 0 : i32 loc(#loc)
    %true_3 = arith.constant true loc(#loc)
    ttng.wait_barrier %1, %c0_i32_2, %true_3 : !ttg.memdesc<1xi64, #shared1, #smem, mutable> loc(#loc)
    ttng.inval_barrier %1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable> loc(#loc)
    %c0_i32_4 = arith.constant 0 : i32 loc(#loc)
    %c0_i32_5 = arith.constant 0 : i32 loc(#loc)
    ttng.async_tma_copy_local_to_global %arg0[%c0_i32_4, %c0_i32_5] %0 : !tt.tensordesc<tensor<128x128xf16>>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable> loc(#loc)
    ttng.async_tma_store_wait {pendings = 0 : i32} loc(#loc)
    tt.return loc(#loc)
  } loc(#loc)
} loc(#loc)
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
    expecttest.assert_expected_inline(
        h.asm["source"], """\
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [1, 0]}>
#loc = loc(unknown)
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func public @async_tma_blackwell_kernel(%arg0: !tt.tensordesc<tensor<1x128xf16>> loc(unknown), %arg1: i32 loc(unknown), %arg2: i32 loc(unknown), %arg3: i64 loc(unknown), %arg4: i64 loc(unknown)) attributes {noinline = false} {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable> loc(#loc)
    %1 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable> loc(#loc)
    ttng.init_barrier %1, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable> loc(#loc)
    %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc)
    %true = arith.constant true loc(#loc)
    %c0_i32 = arith.constant 0 : i32 loc(#loc)
    ttng.async_tma_gather %arg0[%2, %c0_i32] %0, %1, %true : !tt.tensordesc<tensor<1x128xf16>>, tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, i32, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, i1 loc(#loc)
    %true_0 = arith.constant true loc(#loc)
    ttng.barrier_expect %1, 32768, %true_0 : !ttg.memdesc<1xi64, #shared1, #smem, mutable> loc(#loc)
    %c0_i32_1 = arith.constant 0 : i32 loc(#loc)
    %true_2 = arith.constant true loc(#loc)
    ttng.wait_barrier %1, %c0_i32_1, %true_2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable> loc(#loc)
    ttng.inval_barrier %1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable> loc(#loc)
    %c0_i32_3 = arith.constant 0 : i32 loc(#loc)
    ttng.async_tma_scatter %arg0[%2, %c0_i32_3] %0 : !tt.tensordesc<tensor<1x128xf16>>, tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, i32, !ttg.memdesc<128x128xf16, #shared, #smem, mutable> loc(#loc)
    ttng.async_tma_store_wait {pendings = 0 : i32} loc(#loc)
    tt.return loc(#loc)
  } loc(#loc)
} loc(#loc)
""")


def test_mlir_attr_error():

    @gluon.jit
    def kernel():
        ttgl.arange(0, 1, layout=ttgl.BlockedLayout([1], [32], [4], [1]))

    with pytest.raises(CompilationError) as e:
        run_parser(kernel)

    assert "order must be a permutation of 0..(rank-1), but was [1]" in str(e.value.__cause__)


@gluon.jit
def tmem_subslice_kernel():
    layout: ttgl.constexpr = ttgl.nvidia.blackwell.TensorMemoryLayout(block=[128, 128], unpacked=True)
    tmem = ttgl.nvidia.blackwell.allocate_tensor_memory(ttgl.int32, [2, 256, 256], layout)
    tmem.subslice(0)


def test_tmem_subslice_constexpr():
    expecttest.assert_expected_inline(
        run_parser(tmem_subslice_kernel).str_nodebug(), """\
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module {
  tt.func public @tmem_subslice_kernel() attributes {noinline = false} {
    %result = ttng.tmem_alloc : () -> !ttg.memdesc<2x256x256xi32, #tmem, #ttng.tensor_memory, mutable>
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %0 = ttg.memdesc_subview %result[%c0_i32, %c0_i32_0, %c0_i32_0] : !ttg.memdesc<2x256x256xi32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<256x256xi32, #tmem, #ttng.tensor_memory, mutable, 2x256x256>
    tt.return
  }
}
""")


@gluon.jit
def smem_and_layout_user(smem, a: ttgl.constexpr):
    pass


def test_layout_mangling():

    @gluon.jit
    def kernel():
        a: ttgl.constexpr = ttgl.SwizzledSharedLayout(1, 1, 1, [1, 0])
        smem = ttgl.allocate_shared_memory(ttgl.int32, [32, 32], a)
        smem_and_layout_user(smem, a)

    expecttest.assert_expected_inline(
        run_parser(kernel).str_nodebug(), """\
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module {
  tt.func public @kernel() attributes {noinline = false} {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<32x32xi32, #shared, #smem, mutable>
    tt.call @"test_frontend.smem_and_layout_user__MDi32S32_32SLSSS_1_1_1_constexpr[1]_constexpr[0]____SSSLAS[32, 32]ASMD__(1,)cconstexpr_SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=(constexpr_1_ ,constexpr_0_), ctas_per_cga=None, cta_split_num=None, cta_order=None)_"(%0) : (!ttg.memdesc<32x32xi32, #shared, #smem, mutable>) -> ()
    tt.return
  }
  tt.func private @"test_frontend.smem_and_layout_user__MDi32S32_32SLSSS_1_1_1_constexpr[1]_constexpr[0]____SSSLAS[32, 32]ASMD__(1,)cconstexpr_SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=(constexpr_1_ ,constexpr_0_), ctas_per_cga=None, cta_split_num=None, cta_order=None)_"(%arg0: !ttg.memdesc<32x32xi32, #shared, #smem, mutable>) attributes {noinline = false} {
    tt.return
  }
}
""")
