import expecttest
from triton.runtime.jit import MockTensor
import torch
import pytest
import re

from triton import knobs
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl
from triton.experimental.gluon.language.nvidia import blackwell
from triton.experimental.gluon.language.nvidia.blackwell import mbarrier, tma, TensorMemoryLayout
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton._filecheck import filecheck_test, run_parser
import triton.language as tl
from triton._internal_testing import is_cuda
from triton.compiler.errors import CompilationError, CompileTimeAssertionFailure

TARGET_PAT = re.compile('ttg.target = "[^"]*"')


def anonymize_ir(ir):
    return TARGET_PAT.sub('ttg.target = "..."', ir)


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
        anonymize_ir(h.asm["source"]), """\
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "...", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @convert_layout_kernel() attributes {noinline = false} {
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked> loc(#loc)
    %1 = ttg.convert_layout %0 : tensor<128xi32, #blocked> -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> loc(#loc)
    tt.return loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
""")
    expecttest.assert_expected_inline(
        anonymize_ir(h.asm["ttgir"]), """\
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "...", "ttg.threads-per-warp" = 32 : i32} {
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
        anonymize_ir(h.asm["source"]), """\
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "...", "ttg.threads-per-warp" = 32 : i32} {
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
    slice1 = mem.slice(0, YBLOCK // 2)  # noqa: F841
    slice2 = mem.slice(YBLOCK // 2, YBLOCK // 2)  # noqa: F841

    buffers = ttgl.nvidia.blackwell.allocate_tensor_memory(ttgl.float32, [2, XBLOCK, YBLOCK], tmem_layout)
    for i in range(2):
        buffers.index(i).load(layout)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] != 10,
                    reason="Requires blackwell tensor cores")
def test_tensor_memory(fresh_knobs):
    knobs.compilation.disable_line_info = True

    layout = ttgl.BlockedLayout(size_per_thread=[1, 64], threads_per_warp=[32, 1], warps_per_cta=[4, 1], order=[0, 1])
    tmem_layout = TensorMemoryLayout(block=[128, 128], unpacked=True)
    h = tensor_memory_kernel.warmup(layout, tmem_layout, num_warps=4, grid=(1, ))
    expecttest.assert_expected_inline(
        anonymize_ir(h.asm["source"]), """\
#blocked = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "...", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tensor_memory_kernel() attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32 loc(#loc)
    %cst = arith.constant dense<0> : tensor<128x128xi32, #blocked> loc(#loc)
    %result = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xi32, #tmem, #ttng.tensor_memory, mutable> loc(#loc)
    %result_0 = ttng.tmem_alloc %cst : (tensor<128x128xi32, #blocked>) -> !ttg.memdesc<128x128xi32, #tmem, #ttng.tensor_memory, mutable> loc(#loc)
    %result_1 = ttng.tmem_load %result_0 : !ttg.memdesc<128x128xi32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xi32, #blocked> loc(#loc)
    %true = arith.constant true loc(#loc)
    ttng.tmem_store %cst, %result_0, %true : tensor<128x128xi32, #blocked> -> !ttg.memdesc<128x128xi32, #tmem, #ttng.tensor_memory, mutable> loc(#loc)
    %0 = ttng.tmem_subslice %result_0 {N = 0 : i32} : !ttg.memdesc<128x128xi32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xi32, #tmem1, #ttng.tensor_memory, mutable, 128x128> loc(#loc)
    %1 = ttng.tmem_subslice %result_0 {N = 64 : i32} : !ttg.memdesc<128x128xi32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xi32, #tmem1, #ttng.tensor_memory, mutable, 128x128> loc(#loc)
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
    XHALF: ttgl.constexpr = XBLOCK // 2
    smem = ttgl.allocate_shared_memory(ttgl.int32, [XBLOCK, XBLOCK], smem_layout)
    view = smem.slice(XHALF, XHALF, dim=1)
    value = view.load(layout)
    view = smem.slice(XHALF, XHALF, dim=0)
    view.store(value.trans())


def test_shared_memory_subview(fresh_knobs):
    knobs.compilation.disable_line_info = True

    layout = ttgl.BlockedLayout(size_per_thread=[1, 1], threads_per_warp=[1, 32], warps_per_cta=[4, 1], order=[1, 0])
    smem_layout = ttgl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    h = shared_memory_subview_kernel.warmup(256, layout, smem_layout, num_warps=4, grid=(1, ))
    expecttest.assert_expected_inline(
        anonymize_ir(h.asm["source"]), """\
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "...", "ttg.threads-per-warp" = 32 : i32} {
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
def shared_memory_index_kernel(XBLOCK: ttgl.constexpr, layout: ttgl.constexpr, smem_layout: ttgl.constexpr):
    smem = ttgl.allocate_shared_memory(ttgl.int32, [4, XBLOCK], smem_layout)
    for i in range(4):
        smem.index(i).load(layout)


def test_shared_memory_index(fresh_knobs):
    knobs.compilation.disable_line_info = True

    layout = ttgl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32], warps_per_cta=[4], order=[0])
    smem_layout = ttgl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=32, rank=2)
    h = shared_memory_index_kernel.warmup(256, layout, smem_layout, num_warps=4, grid=(1, ))
    expecttest.assert_expected_inline(
        anonymize_ir(h.asm["source"]), """\
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "...", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @shared_memory_index_kernel() attributes {noinline = false} {
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
                                                      rank=2, ctas_per_cga=[1, 1], cta_split_num=[1,
                                                                                                  1], cta_order=[1, 0])
    smem = ttgl.allocate_shared_memory(ttgl.int8, [2, 256, 128], layout_a)
    perm = smem.index(0).permute((1, 0))
    ttgl.static_assert(perm.type.layout == layout_T)
    # Check that the MLIR type and Gluon types match by emitting a call.
    anchor_noinline(perm)

    layout_b: ttgl.constexpr = ttgl.NVMMASharedLayout(swizzle_byte_width=64, transposed=False, element_bitwidth=16,
                                                      rank=4, cta_order=[3, 2, 1, 0])
    layout_reshape: ttgl.constexpr = ttgl.NVMMASharedLayout(swizzle_byte_width=64, transposed=False,
                                                            element_bitwidth=16, rank=2)
    smem = ttgl.allocate_shared_memory(ttgl.float16, [32, 1, 4, 64], layout_b)
    smem.reshape((128, 64), layout_reshape)

    smem._reinterpret(ttgl.int8, [1024], ttgl.SwizzledSharedLayout(1, 1, 1, [0, 1]))


def test_shared_memory_cast(fresh_knobs):
    expecttest.assert_expected_inline(
        anonymize_ir(run_parser(shared_memory_cast_kernel).str_nodebug()), """\
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = true, elementBitWidth = 8}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16, CTAsPerCGA = [1, 1, 1, 1], CTASplitNum = [1, 1, 1, 1], CTAOrder = [3, 2, 1, 0]}>
#shared3 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#shared4 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "...", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @shared_memory_cast_kernel() attributes {noinline = false} {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<2x256x128xi8, #shared, #smem, mutable>
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %1 = ttg.memdesc_subview %0[%c0_i32_0, %c0_i32, %c0_i32] : !ttg.memdesc<2x256x128xi8, #shared, #smem, mutable> -> !ttg.memdesc<256x128xi8, #shared, #smem, mutable, 2x256x128>
    %2 = ttg.memdesc_trans %1 {order = array<i32: 1, 0>} : !ttg.memdesc<256x128xi8, #shared, #smem, mutable, 2x256x128> -> !ttg.memdesc<128x256xi8, #shared1, #smem, mutable, 2x128x256>
    tt.call @"test_frontend.anchor_noinline__MDi8S128_256SLNVMMA_64_8_True_False_NVMMALAS[2, 128, 256]ASMD__"(%2) : (!ttg.memdesc<128x256xi8, #shared1, #smem, mutable, 2x128x256>) -> ()
    %3 = ttg.local_alloc : () -> !ttg.memdesc<32x1x4x64xf16, #shared2, #smem, mutable>
    %4 = ttg.memdesc_reshape %3 : !ttg.memdesc<32x1x4x64xf16, #shared2, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared3, #smem, mutable, 32x1x4x64>
    %5 = ttg.memdesc_reinterpret %3 : !ttg.memdesc<32x1x4x64xf16, #shared2, #smem, mutable> -> !ttg.memdesc<1024xi8, #shared4, #smem, mutable>
    tt.return
  }
  tt.func private @"test_frontend.anchor_noinline__MDi8S128_256SLNVMMA_64_8_True_False_NVMMALAS[2, 128, 256]ASMD__"(%arg0: !ttg.memdesc<128x256xi8, #shared1, #smem, mutable, 2x128x256>) attributes {noinline = true} {
    tt.return
  }
}
""")


@gluon.jit
def warp_specialize_default(a, b, e: ttgl.constexpr):
    return b, a


@gluon.jit
def warp_specialize_worker0(a, b, e: ttgl.constexpr):
    pass


@gluon.jit
def warp_specialize_worker1(a, b, e: ttgl.constexpr):
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


@gluon.jit(noinline=True)
def anchor_noinline(x):
    pass


@filecheck_test
@gluon.jit
def test_warp_specialize():
    # CHECK:       [[BLOCKED:#.*]] = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
    # CHECK-LABEL: test_warp_specialize
    # CHECK-NEXT:    [[A:%.*]] = tt.make_range {end = 1 : i32, start = 0 : i32}
    # CHECK-NEXT:    [[B:%.*]] = tt.make_range {end = 2 : i32, start = 0 : i32}
    # CHECK-NEXT:    [[C:%.*]] = tt.make_range {end = 4 : i32, start = 0 : i32}
    # CHECK-NEXT:    [[OUTS:%.*]]:3 = ttg.warp_specialize([[A]], [[B]], [[C]]) {{.*}}requestedRegisters = array<i32: 24, 48>
    # CHECK-NEXT:    default {
    # CHECK-NEXT:      [[RESULTS:%.*]]:3 = tt.call @{{.*}}warp_specialize_default{{.*}}cconstexpr_42{{.*}}([[A]], [[B]], [[C]])
    # CHECK-NEXT:      warp_yield [[RESULTS]]#0, [[RESULTS]]#1, [[RESULTS]]#2
    # CHECK-NEXT:    }
    # CHECK-NEXT:    partition0(%arg0: tensor<1xi32, [[BLOCKED]]>, %arg1: tensor<2xi32, [[BLOCKED]]>, %arg2: tensor<4xi32, [[BLOCKED]]>) num_warps(4) {
    # CHECK-NEXT:      call @{{.*}}warp_specialize_worker0{{.*}}cconstexpr_42{{.*}}(%arg0, %arg1, %arg2)
    # CHECK-NEXT:      warp_return
    # CHECK-NEXT:    }
    # CHECK-NEXT:    partition1(%arg0: tensor<1xi32, [[BLOCKED]]>, %arg1: tensor<2xi32, [[BLOCKED]]>, %arg2: tensor<4xi32, [[BLOCKED]]>) num_warps(4) {
    # CHECK-NEXT:      call @{{.*}}warp_specialize_worker1{{.*}}cconstexpr_42{{.*}}(%arg0, %arg1, %arg2)
    # CHECK-NEXT:      warp_return
    # CHECK-NEXT:    }
    # CHECK-NEXT:    call @{{.*}}anchor{{.*}}([[OUTS]]#0)
    # CHECK-NEXT:    call @{{.*}}anchor{{.*}}([[OUTS]]#1, [[OUTS]]#2)
    layout: ttgl.constexpr = ttgl.BlockedLayout([1], [32], [4], [0])
    a = ttgl.arange(0, 1, layout=layout)
    b = ttgl.arange(0, 2, layout=layout)
    c = ttgl.arange(0, 4, layout=layout)
    pair = Pair(a, b)
    e: ttgl.constexpr = 42
    a, b = ttgl.warp_specialize((pair, c, e), warp_specialize_default,
                                [warp_specialize_worker0, warp_specialize_worker1], [4, 4], [24, 48])
    anchor(a)
    anchor(b)

    # CHECK: ttg.warp_specialize([[A]], [[B]], [[C]])
    # CHECK: (tensor<1xi32, [[BLOCKED]]>, tensor<2xi32, [[BLOCKED]]>, tensor<4xi32, [[BLOCKED]]>) -> ()
    ttgl.warp_specialize((pair, c, e), warp_specialize_worker0, [warp_specialize_worker1], [4], [48])


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
        anonymize_ir(h.asm["source"]), """\
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "...", "ttg.threads-per-warp" = 32 : i32} {
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
    acc_layout = TensorMemoryLayout([128, 128], unpacked=True)

    h = tcgen05_mma_kernel.warmup(nvmma_layout, acc_layout, grid=(1, ))
    expecttest.assert_expected_inline(
        anonymize_ir(h.asm["source"]), """\
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "...", "ttg.threads-per-warp" = 32 : i32} {
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
def async_tma_kernel(input_desc, XBLOCK: ttgl.constexpr):
    smem = ttgl.allocate_shared_memory(ttgl.float16, [XBLOCK, XBLOCK], input_desc.layout)
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
    shared_layout = ttgl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2)
    input_desc = TensorDescriptor.from_tensor(input, [XBLOCK, XBLOCK], shared_layout)

    h = async_tma_kernel.warmup(input_desc, XBLOCK, grid=(1, ), num_warps=4)
    expecttest.assert_expected_inline(
        anonymize_ir(h.asm["source"]), """\
#loc = loc(unknown)
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "...", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @async_tma_kernel(%arg0: !tt.tensordesc<tensor<128x128xf16, #shared>> loc(unknown), %arg1: i32 loc(unknown), %arg2: i32 loc(unknown), %arg3: i64 loc(unknown), %arg4: i64 loc(unknown)) attributes {noinline = false} {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable> loc(#loc)
    %1 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable> loc(#loc)
    ttng.init_barrier %1, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable> loc(#loc)
    %c0_i32 = arith.constant 0 : i32 loc(#loc)
    %c0_i32_0 = arith.constant 0 : i32 loc(#loc)
    %true = arith.constant true loc(#loc)
    ttng.async_tma_copy_global_to_local %arg0[%c0_i32, %c0_i32_0] %0, %1, %true : !tt.tensordesc<tensor<128x128xf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable> loc(#loc)
    %true_1 = arith.constant true loc(#loc)
    ttng.barrier_expect %1, 32768, %true_1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable> loc(#loc)
    %c0_i32_2 = arith.constant 0 : i32 loc(#loc)
    %true_3 = arith.constant true loc(#loc)
    ttng.wait_barrier %1, %c0_i32_2, %true_3 : !ttg.memdesc<1xi64, #shared1, #smem, mutable> loc(#loc)
    ttng.inval_barrier %1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable> loc(#loc)
    %c0_i32_4 = arith.constant 0 : i32 loc(#loc)
    %c0_i32_5 = arith.constant 0 : i32 loc(#loc)
    ttng.async_tma_copy_local_to_global %arg0[%c0_i32_4, %c0_i32_5] %0 : !tt.tensordesc<tensor<128x128xf16, #shared>>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable> loc(#loc)
    ttng.async_tma_store_wait {pendings = 0 : i32} loc(#loc)
    tt.return loc(#loc)
  } loc(#loc)
} loc(#loc)
""")


@gluon.jit
def async_tma_blackwell_kernel(input_desc, XBLOCK: ttgl.constexpr):
    smem = ttgl.allocate_shared_memory(ttgl.float16, [XBLOCK, XBLOCK], input_desc.layout)
    bar = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)

    offset_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 4], [32, 1], [1, 4], [1, 0])
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
    shared_layout = ttgl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2)
    input_desc = TensorDescriptor.from_tensor(input, [1, XBLOCK], shared_layout)

    h = async_tma_blackwell_kernel.warmup(input_desc, XBLOCK, grid=(1, ), num_warps=4)
    expecttest.assert_expected_inline(
        anonymize_ir(h.asm["source"]), """\
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [1, 0]}>
#loc = loc(unknown)
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "...", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @async_tma_blackwell_kernel(%arg0: !tt.tensordesc<tensor<1x128xf16, #shared>> loc(unknown), %arg1: i32 loc(unknown), %arg2: i32 loc(unknown), %arg3: i64 loc(unknown), %arg4: i64 loc(unknown)) attributes {noinline = false} {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable> loc(#loc)
    %1 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable> loc(#loc)
    ttng.init_barrier %1, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable> loc(#loc)
    %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc)
    %true = arith.constant true loc(#loc)
    %c0_i32 = arith.constant 0 : i32 loc(#loc)
    ttng.async_tma_gather %arg0[%2, %c0_i32] %0, %1, %true : !tt.tensordesc<tensor<1x128xf16, #shared>>, tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, i32, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, i1 loc(#loc)
    %true_0 = arith.constant true loc(#loc)
    ttng.barrier_expect %1, 32768, %true_0 : !ttg.memdesc<1xi64, #shared1, #smem, mutable> loc(#loc)
    %c0_i32_1 = arith.constant 0 : i32 loc(#loc)
    %true_2 = arith.constant true loc(#loc)
    ttng.wait_barrier %1, %c0_i32_1, %true_2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable> loc(#loc)
    ttng.inval_barrier %1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable> loc(#loc)
    %c0_i32_3 = arith.constant 0 : i32 loc(#loc)
    ttng.async_tma_scatter %arg0[%2, %c0_i32_3] %0 : !tt.tensordesc<tensor<1x128xf16, #shared>>, tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, i32, !ttg.memdesc<128x128xf16, #shared, #smem, mutable> loc(#loc)
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
def tmem_index_kernel():
    layout: ttgl.constexpr = TensorMemoryLayout(block=[128, 128], unpacked=True)
    tmem = ttgl.nvidia.blackwell.allocate_tensor_memory(ttgl.int32, [2, 256, 256], layout)
    tmem.index(0)


def test_tmem_index_constexpr():
    expecttest.assert_expected_inline(
        anonymize_ir(run_parser(tmem_index_kernel).str_nodebug()), """\
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "...", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tmem_index_kernel() attributes {noinline = false} {
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
        anonymize_ir(run_parser(kernel).str_nodebug()), """\
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "...", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @kernel() attributes {noinline = false} {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<32x32xi32, #shared, #smem, mutable>
    tt.call @"test_frontend.smem_and_layout_user__MDi32S32_32SLSSS_1_1_1_1_0____SSSLAS[32, 32]ASMD__(1,)cconstexpr_SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=(1 ,0), ctas_per_cga=None, cta_split_num=None, cta_order=None)_"(%0) : (!ttg.memdesc<32x32xi32, #shared, #smem, mutable>) -> ()
    tt.return
  }
  tt.func private @"test_frontend.smem_and_layout_user__MDi32S32_32SLSSS_1_1_1_1_0____SSSLAS[32, 32]ASMD__(1,)cconstexpr_SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=(1 ,0), ctas_per_cga=None, cta_split_num=None, cta_order=None)_"(%arg0: !ttg.memdesc<32x32xi32, #shared, #smem, mutable>) attributes {noinline = false} {
    tt.return
  }
}
""")


@gluon.jit
def broadcast_kernel():
    layout: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [2, 16], [4, 1], [1, 0])
    a = ttgl.arange(0, 16, layout=ttgl.SliceLayout(0, layout))[None, :]
    b = ttgl.arange(0, 16, layout=ttgl.SliceLayout(1, layout))[:, None]
    0 + a + b


def test_broadcast(fresh_knobs):
    knobs.compilation.disable_line_info = True

    h = broadcast_kernel.warmup(sanitize_overflow=False, grid=(1, ))
    expecttest.assert_expected_inline(
        anonymize_ir(h.asm["source"]), """\
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "...", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @broadcast_kernel() attributes {noinline = false} {
    %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc)
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x16xi32, #blocked> loc(#loc)
    %2 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc)
    %3 = tt.expand_dims %2 {axis = 1 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<16x1xi32, #blocked> loc(#loc)
    %c0_i32 = arith.constant 0 : i32 loc(#loc)
    %c0_i32_0 = arith.constant 0 : i32 loc(#loc)
    %cst = arith.constant dense<0> : tensor<1x16xi32, #blocked> loc(#loc)
    %4 = arith.addi %cst, %1 : tensor<1x16xi32, #blocked> loc(#loc)
    %5 = tt.broadcast %4 : tensor<1x16xi32, #blocked> -> tensor<16x16xi32, #blocked> loc(#loc)
    %6 = tt.broadcast %3 : tensor<16x1xi32, #blocked> -> tensor<16x16xi32, #blocked> loc(#loc)
    %7 = arith.addi %5, %6 : tensor<16x16xi32, #blocked> loc(#loc)
    tt.return loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
""")


@gluon.jit
def math_kernel():
    layout: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [1, 32], [4, 1], [1, 0])
    a = ttgl.full([16, 16], 1, ttgl.float32, layout)
    b = ttgl.full([16, 16], 2, ttgl.float32, layout)
    c = ttgl.full([16, 16], 4, ttgl.float32, layout)
    d = ttgl.full([16, 16], 1, ttgl.int32, layout)
    e = ttgl.full([16, 16], 1, ttgl.int32, layout)
    ttgl.umulhi(d, e)
    ttgl.exp(a)
    ttgl.exp2(a)
    ttgl.log(a)
    ttgl.log2(a)
    ttgl.cos(a)
    ttgl.sin(a)
    ttgl.sqrt(a)
    ttgl.sqrt_rn(a)
    ttgl.rsqrt(a)
    ttgl.abs(a)
    ttgl.fdiv(a, b)
    ttgl.div_rn(a, b)
    ttgl.erf(a)
    ttgl.floor(a)
    ttgl.ceil(a)
    ttgl.fma(a, b, c)


def test_math(fresh_knobs):
    knobs.compilation.disable_line_info = True

    h = math_kernel.warmup(sanitize_overflow=False, grid=(1, ))
    expecttest.assert_expected_inline(
        anonymize_ir(h.asm["source"]), """\
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "...", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @math_kernel() attributes {noinline = false} {
    %cst = arith.constant 1.000000e+00 : f32 loc(#loc)
    %cst_0 = arith.constant dense<1.000000e+00> : tensor<16x16xf32, #blocked> loc(#loc)
    %cst_1 = arith.constant 2.000000e+00 : f32 loc(#loc)
    %cst_2 = arith.constant dense<2.000000e+00> : tensor<16x16xf32, #blocked> loc(#loc)
    %cst_3 = arith.constant 4.000000e+00 : f32 loc(#loc)
    %cst_4 = arith.constant dense<4.000000e+00> : tensor<16x16xf32, #blocked> loc(#loc)
    %c1_i32 = arith.constant 1 : i32 loc(#loc)
    %cst_5 = arith.constant dense<1> : tensor<16x16xi32, #blocked> loc(#loc)
    %c1_i32_6 = arith.constant 1 : i32 loc(#loc)
    %cst_7 = arith.constant dense<1> : tensor<16x16xi32, #blocked> loc(#loc)
    %0 = tt.mulhiui %cst_5, %cst_7 : tensor<16x16xi32, #blocked> loc(#loc)
    %1 = math.exp %cst_0 : tensor<16x16xf32, #blocked> loc(#loc)
    %2 = math.exp2 %cst_0 : tensor<16x16xf32, #blocked> loc(#loc)
    %3 = math.log %cst_0 : tensor<16x16xf32, #blocked> loc(#loc)
    %4 = math.log2 %cst_0 : tensor<16x16xf32, #blocked> loc(#loc)
    %5 = math.cos %cst_0 : tensor<16x16xf32, #blocked> loc(#loc)
    %6 = math.sin %cst_0 : tensor<16x16xf32, #blocked> loc(#loc)
    %7 = math.sqrt %cst_0 : tensor<16x16xf32, #blocked> loc(#loc)
    %8 = tt.precise_sqrt %cst_0 : tensor<16x16xf32, #blocked> loc(#loc)
    %9 = math.rsqrt %cst_0 : tensor<16x16xf32, #blocked> loc(#loc)
    %10 = math.absf %cst_0 : tensor<16x16xf32, #blocked> loc(#loc)
    %11 = arith.divf %cst_0, %cst_2 : tensor<16x16xf32, #blocked> loc(#loc)
    %12 = tt.precise_divf %cst_0, %cst_2 : tensor<16x16xf32, #blocked> loc(#loc)
    %13 = math.erf %cst_0 : tensor<16x16xf32, #blocked> loc(#loc)
    %14 = math.floor %cst_0 : tensor<16x16xf32, #blocked> loc(#loc)
    %15 = math.ceil %cst_0 : tensor<16x16xf32, #blocked> loc(#loc)
    %16 = math.fma %cst_0, %cst_2, %cst_4 : tensor<16x16xf32, #blocked> loc(#loc)
    tt.return loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
""")


@gluon.jit
def pair_add(a0, a1, b0, b1):
    return a0 + b0, a1 + b1


@gluon.jit
def reduce_kernel(out):
    layout: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [1, 32], [4, 1], [1, 0])
    a = ttgl.full([16, 16], 1, ttgl.float32, layout)
    b = ttgl.full([16, 16], 2, ttgl.float32, layout)
    s0 = ttgl.sum(a, 0)
    ttgl.static_assert(s0.type.layout == ttgl.SliceLayout(0, layout))
    s1 = ttgl.sum(a, 1)
    ttgl.static_assert(s1.type.layout == ttgl.SliceLayout(1, layout))

    scalar = ttgl.max(s0, 0)
    ttgl.static_assert(scalar.type == ttgl.float32)

    s1 = ttgl.convert_layout(s1, s0.type.layout)

    pairs = ttgl.reduce((a, b), 0, pair_add)
    ttgl.static_assert(pairs[0].type.layout == ttgl.SliceLayout(0, layout))
    ttgl.static_assert(pairs[1].type.layout == ttgl.SliceLayout(0, layout))
    result = scalar + s1 + pairs[0] + pairs[1]
    tl.store(out + ttgl.arange(0, 16, s0.type.layout), result)


def test_reduce(fresh_knobs):
    knobs.compilation.disable_line_info = True

    h = reduce_kernel.warmup(MockTensor(ttgl.float32), sanitize_overflow=False, grid=(1, ))
    expecttest.assert_expected_inline(
        anonymize_ir(h.asm["ttgir"]), """\
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#loc = loc(unknown)
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "...", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reduce_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc(unknown)) attributes {noinline = false} {
    %cst = arith.constant dense<2.000000e+00> : tensor<16x16xf32, #blocked> loc(#loc)
    %cst_0 = arith.constant dense<1.000000e+00> : tensor<16x16xf32, #blocked> loc(#loc)
    %0 = "tt.reduce"(%cst_0) <{axis = 0 : i32}> ({
    ^bb0(%arg1: f32 loc(unknown), %arg2: f32 loc(unknown)):
      %12 = arith.addf %arg1, %arg2 : f32 loc(#loc)
      tt.reduce.return %12 : f32 loc(#loc)
    }) : (tensor<16x16xf32, #blocked>) -> tensor<16xf32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc)
    %1 = "tt.reduce"(%cst_0) <{axis = 1 : i32}> ({
    ^bb0(%arg1: f32 loc(unknown), %arg2: f32 loc(unknown)):
      %12 = arith.addf %arg1, %arg2 : f32 loc(#loc)
      tt.reduce.return %12 : f32 loc(#loc)
    }) : (tensor<16x16xf32, #blocked>) -> tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc)
    %2 = "tt.reduce"(%0) <{axis = 0 : i32}> ({
    ^bb0(%arg1: f32 loc(unknown), %arg2: f32 loc(unknown)):
      %12 = arith.maxnumf %arg1, %arg2 : f32 loc(#loc)
      tt.reduce.return %12 : f32 loc(#loc)
    }) : (tensor<16xf32, #ttg.slice<{dim = 0, parent = #blocked}>>) -> f32 loc(#loc)
    %3 = ttg.convert_layout %1 : tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<16xf32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc)
    %4:2 = "tt.reduce"(%cst_0, %cst) <{axis = 0 : i32}> ({
    ^bb0(%arg1: f32 loc(unknown), %arg2: f32 loc(unknown), %arg3: f32 loc(unknown), %arg4: f32 loc(unknown)):
      %12 = arith.addf %arg1, %arg3 : f32 loc(#loc)
      %13 = arith.addf %arg2, %arg4 : f32 loc(#loc)
      tt.reduce.return %12, %13 : f32, f32 loc(#loc)
    }) : (tensor<16x16xf32, #blocked>, tensor<16x16xf32, #blocked>) -> (tensor<16xf32, #ttg.slice<{dim = 0, parent = #blocked}>>, tensor<16xf32, #ttg.slice<{dim = 0, parent = #blocked}>>) loc(#loc)
    %5 = tt.splat %2 : f32 -> tensor<16xf32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc)
    %6 = arith.addf %5, %3 : tensor<16xf32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc)
    %7 = arith.addf %6, %4#0 : tensor<16xf32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc)
    %8 = arith.addf %7, %4#1 : tensor<16xf32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc)
    %9 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc)
    %10 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc)
    %11 = tt.addptr %10, %9 : tensor<16x!tt.ptr<f32>, #ttg.slice<{dim = 0, parent = #blocked}>>, tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc)
    tt.store %11, %8 : tensor<16x!tt.ptr<f32>, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc)
    tt.return loc(#loc)
  } loc(#loc)
} loc(#loc)
""")


@filecheck_test
@gluon.jit
def test_elementwise_core():
    # CHECK: [[BLOCKED:#.*]] = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
    # CHECK: @test_elementwise_core
    layout: ttgl.constexpr = ttgl.BlockedLayout([1], [32], [4], [0])
    x = ttgl.arange(0, 16, layout)
    y = ttgl.arange(16, 32, layout)

    # CHECK: arith.select {{.*}} : tensor<16xi1, [[BLOCKED]]>, tensor<16xi32, [[BLOCKED]]>
    a = ttgl.where(x > 8, x, y)
    # CHECK: arith.maxsi {{.*}} : tensor<16xi32, [[BLOCKED]]>
    b = ttgl.maximum(x, y)
    # CHECK: arith.minsi {{.*}} : tensor<16xi32, [[BLOCKED]]>
    c = ttgl.minimum(x, y)
    ttgl.static_assert(a.type == x.type)
    ttgl.static_assert(b.type == x.type)
    ttgl.static_assert(c.type == x.type)


@gluon.jit
def linear_layout_kernel():
    ll: tl.constexpr = ttgl.DistributedLinearLayout(reg_bases=[[1]], lane_bases=[[2], [4], [8], [16], [32]],
                                                    warp_bases=[[64], [128]], block_bases=[], shape=[256])
    ttgl.arange(0, 256, layout=ll)


def test_linear_layout(fresh_knobs):
    knobs.compilation.disable_line_info = True
    h = linear_layout_kernel.warmup(grid=(1, ))
    expecttest.assert_expected_inline(
        anonymize_ir(h.asm["source"]), """\
#linear = #ttg.linear<{register = [[1]], lane = [[2], [4], [8], [16], [32]], warp = [[64], [128]], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "...", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @linear_layout_kernel() attributes {noinline = false} {
    %0 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #linear> loc(#loc)
    tt.return loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
""")


@filecheck_test
@gluon.jit
def test_tensor_permute():
    # CHECK-DAG: [[BLOCKED:#.*]] = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
    # CHECK-DAG: [[BLOCKED1:#.*]] = #ttg.blocked<{sizePerThread = [2, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
    layout: ttgl.constexpr = ttgl.BlockedLayout([1, 2], [4, 8], [4, 1], [1, 0])
    a = ttgl.full([32, 16], 0, ttgl.int32, layout=layout)
    # CHECK: tt.trans{{.*}} : tensor<32x16xi32, [[BLOCKED]]> -> tensor<16x32xi32, [[BLOCKED1]]>
    res = ttgl.permute(a, [1, 0])
    permuted_layout: ttgl.constexpr = ttgl.BlockedLayout([2, 1], [8, 4], [1, 4], [0, 1], [1, 1], [1, 1], [1, 0])
    ttgl.static_assert(permuted_layout == res.type.layout)


@filecheck_test
@gluon.jit
def test_split_join():
    # CHECK: [[BLOCKED:#.*]] = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
    # CHECK: [[BLOCKED1:#.*]] = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
    layout: ttgl.constexpr = ttgl.BlockedLayout([2], [32], [4], [0], [1], [1], [0])
    a = ttgl.full([128], 1, ttgl.int32, layout)
    b = ttgl.full([128], 2, ttgl.int32, layout)
    # CHECK: tt.join {{.*}} : tensor<128xi32, [[BLOCKED]]> -> tensor<128x2xi32, [[BLOCKED1]]>
    res = ttgl.join(a, b)
    expect_layout: ttgl.constexpr = ttgl.BlockedLayout([2, 2], [32, 1], [4, 1], [1, 0], [1, 1], [1, 1], [1, 0])
    ttgl.static_assert(res.type.layout == expect_layout)

    # CHECK: tt.split {{.*}} : tensor<128x2xi32, [[BLOCKED1]]> -> tensor<128xi32, [[BLOCKED]]>
    c, d = ttgl.split(res)
    ttgl.static_assert(c.type.layout == layout)
    ttgl.static_assert(d.type.layout == layout)


@filecheck_test
@gluon.jit
def test_tensor_reshape():
    # CHECK: [[BLOCKED:#.*]] = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
    # CHECK: [[BLOCKED1:#.*]] = #ttg.blocked<{sizePerThread = [1, 1, 2], threadsPerWarp = [2, 4, 4], warpsPerCTA = [4, 1, 1], order = [2, 1, 0]}>
    layout: ttgl.constexpr = ttgl.BlockedLayout([2], [32], [4], [0])
    a = ttgl.full([256], 1, ttgl.int32, layout)
    # CHECK: tt.reshape {{.*}} : tensor<256xi32, [[BLOCKED]]> -> tensor<8x4x8xi32, [[BLOCKED1]]>
    v = a.reshape([8, 4, 8])
    expect_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 1, 2], [2, 4, 4], [4, 1, 1], [2, 1, 0], [1, 1, 1], [1, 1, 1],
                                                       [2, 1, 0])
    ttgl.static_assert(v.type.layout == expect_layout)


@gluon.jit
def static_assert_kernel():
    ttgl.static_assert(False)


def test_static_assert():
    with pytest.raises(CompileTimeAssertionFailure):
        run_parser(static_assert_kernel)


@filecheck_test
@gluon.jit
def test_zeros():
    # CHECK: [[BLOCKED:#.*]] = #ttg.blocked<{sizePerThread = [2]
    # CHECK: [[BLOCKED2D:#.*]] = #ttg.blocked<{sizePerThread = [1, 2]
    layout: ttgl.constexpr = ttgl.BlockedLayout([2], [32], [4], [0])
    layout_2d: ttgl.constexpr = ttgl.BlockedLayout([1, 2], [4, 8], [4, 1], [1, 0])

    # CHECK: arith.constant dense<0.000000e+00> : tensor<32xf32, [[BLOCKED]]>
    a = ttgl.zeros([32], ttgl.float32, layout)

    # CHECK: arith.constant dense<7.000000e+00> : tensor<32xf32, [[BLOCKED]]>
    ttgl.full_like(a, 7)

    # CHECK: arith.constant dense<0.000000e+00> : tensor<32xf32, [[BLOCKED]]>
    ttgl.zeros_like(a)

    # CHECK: arith.constant dense<0.000000e+00> : tensor<64xf32, [[BLOCKED]]>
    ttgl.zeros_like(a, shape=[64])

    # CHECK: arith.constant dense<0> : tensor<16x16xi8, [[BLOCKED2D]]>
    ttgl.zeros_like(a, shape=[16, 16], dtype=ttgl.int8, layout=layout_2d)

    # CHECK: arith.constant dense<7> : tensor<8x8xi16, [[BLOCKED2D]]>
    ttgl.full_like(a, 7, shape=[8, 8], dtype=ttgl.int16, layout=layout_2d)


@filecheck_test
@gluon.jit
def test_barrier():
    # CHECK: gpu.barrier
    ttgl.thread_barrier()


@filecheck_test
@gluon.jit
def test_fence_async_shared():
    # CHECK: ttng.fence_async_shared {bCluster = false}
    blackwell.fence_async_shared()

    # CHECK-NEXT: ttng.fence_async_shared {bCluster = true}
    blackwell.fence_async_shared(cluster=True)


@filecheck_test
@gluon.jit
def test_inline_asm_elementwise():
    layout: ttgl.constexpr = ttgl.BlockedLayout([1], [32], [4], [0])
    x = ttgl.arange(0, 16, layout)
    # CHECK: elementwise_inline_asm {{.*}} : tensor<16xi32, [[BLOCKED:#.*]]> -> tensor<16xi32, [[BLOCKED]]>
    ttgl.inline_asm_elementwise("mov $0, $0;", "=r,r", [x], dtype=x.dtype, is_pure=True, pack=1)
