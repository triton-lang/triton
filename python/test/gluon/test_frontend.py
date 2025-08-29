import expecttest
import pytest
import re

from triton.backends.compiler import GPUTarget
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl
from triton.experimental.gluon.language.nvidia import blackwell
from triton.experimental.gluon.language.nvidia import hopper
from triton.experimental.gluon.language.nvidia.blackwell import mbarrier, tma, TensorMemoryLayout, TensorMemoryScalesLayout, async_copy
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.amd import _layouts as amd_layouts
from triton.experimental.gluon.language.amd.cdna4 import async_copy as cdna4_async_copy
from triton.experimental.gluon.language.extra import libdevice

from triton._filecheck import filecheck_test, run_parser
from triton.runtime.jit import MockTensor
import triton.language as tl
from triton.compiler.errors import CompilationError, CompileTimeAssertionFailure

TARGET_PAT = re.compile('ttg.target = "[^"]*"')
# HIP backend can add this attribute to function parameters
PTRRANGE_PAT = re.compile('(, )?tt.pointer_range = 32 : i32')
LIBDEVICE_PAT = re.compile('{libname = "", libpath = "", pure = true, symbol = "__.*"}')

BLACKWELL_TARGET = GPUTarget("cuda", 100, 32)
HOPPER_TARGET = GPUTarget("cuda", 90, 32)
AMPERE_TARGET = GPUTarget("cuda", 80, 32)
HIP_TARGET = GPUTarget("hip", "gfx1200", 32)
HIP_TARGET_CDNA3 = GPUTarget("hip", "gfx942", 64)
HIP_TARGET_CDNA4 = GPUTarget("hip", "gfx950", 64)

ALL_TARGETS = [AMPERE_TARGET, HOPPER_TARGET, BLACKWELL_TARGET, HIP_TARGET]


def anonymize_ir(ir):
    ir = TARGET_PAT.sub('ttg.target = "..."', ir)
    ir = PTRRANGE_PAT.sub('', ir)
    ir = LIBDEVICE_PAT.sub('{libname = "", libpath = "", pure = true, symbol = "..."}', ir)
    return ir


def make_args(*args, **kwargs):
    return args, kwargs


@gluon.jit
def convert_layout_kernel(XBLOCK: ttgl.constexpr, layout_a: ttgl.constexpr, layout_b: ttgl.constexpr):
    x = ttgl.arange(0, XBLOCK, layout=layout_a)
    res = ttgl.convert_layout(x, layout_b)  # noqa: F841


@pytest.mark.parametrize("target", ALL_TARGETS)
def test_convert_layout(target):
    layout_a = ttgl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32], warps_per_cta=[4], order=[0])
    layout_b = ttgl.SliceLayout(
        1, ttgl.BlockedLayout(size_per_thread=[1, 1], threads_per_warp=[1, 32], warps_per_cta=[1, 4], order=[1, 0]))
    mod = run_parser(
        convert_layout_kernel,
        *make_args(128, layout_a, layout_b, num_warps=layout_a.warps_per_cta[0]),
        target=target,
    )
    expecttest.assert_expected_inline(
        anonymize_ir(mod.str_nodebug()), """\
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "...", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @convert_layout_kernel() attributes {noinline = false} {
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %1 = ttg.convert_layout %0 : tensor<128xi32, #blocked> -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    tt.return
  }
}
""")


@filecheck_test
@gluon.jit
def test_histogram_frontend():
    # CHECK: #blocked = #ttg.blocked
    # CHECK-LABEL: test_histogram_frontend
    layout: ttgl.constexpr = ttgl.BlockedLayout([1], [32], [4], [0])
    x = ttgl.arange(0, 256, layout=layout)
    m = x < 128
    # CHECK: tt.histogram %{{.*}}, %{{.*}} : tensor<256xi32, #blocked> -> tensor<512xi32, #blocked>
    _ = ttgl.histogram(x, 512, mask=m, layout=layout)


@filecheck_test
@gluon.jit
def test_convert_layout_assert_trivial():
    # CHECK: test_convert_layout_assert_trivial
    parent_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 128], [32, 1], [4, 1], [0, 1])
    slice_layout: ttgl.constexpr = ttgl.SliceLayout(1, parent_layout)
    equiv_layout: ttgl.constexpr = ttgl.BlockedLayout([1], [32], [4], [0])

    value = ttgl.arange(0, 128, layout=slice_layout)
    # CHECK: ttg.convert_layout
    ttgl.convert_layout(value, equiv_layout, assert_trivial=True)


@pytest.mark.parametrize("target", ALL_TARGETS)
def test_convert_layout_not_trivial(target):

    @gluon.jit
    def kernel(src_layout: ttgl.constexpr, dst_layout: ttgl.constexpr):
        value = ttgl.arange(0, 128, layout=src_layout)
        ttgl.convert_layout(value, dst_layout, assert_trivial=True)

    with pytest.raises(CompilationError) as e:
        src_layout = ttgl.BlockedLayout([2], [32], [4], [0])
        dst_layout = ttgl.BlockedLayout([1], [32], [4], [0])
        run_parser(kernel, *make_args(src_layout, dst_layout), target=target)

    assert "layout conversion from BlockedLayout(size_per_thread=[2]" in str(e.value.__cause__)
    assert "to BlockedLayout(size_per_thread=[1]" in str(e.value.__cause__)
    assert "is not trivial" in str(e.value.__cause__)

    with pytest.raises(CompilationError) as e:
        src_layout = ttgl.BlockedLayout([2], [32], [4], [0])
        dst_layout = ttgl.AutoLayout()
        run_parser(kernel, *make_args(src_layout, dst_layout), target=target)

    assert "layout conversion from BlockedLayout(size_per_thread=[2]" in str(e.value.__cause__)
    assert "to AutoLayout() is not trivial" in str(e.value.__cause__)

    with pytest.raises(CompilationError) as e:
        src_layout: ttgl.constexpr = ttgl.AutoLayout()
        dst_layout: ttgl.constexpr = ttgl.BlockedLayout([2], [32], [4], [0])
        run_parser(kernel, *make_args(src_layout, dst_layout), target=target)

    assert "layout conversion from AutoLayout()" in str(e.value.__cause__)
    assert "to BlockedLayout(size_per_thread=[2]" in str(e.value.__cause__)
    assert "is not trivial" in str(e.value.__cause__)


@gluon.jit
def shared_memory_kernel(XBLOCK: ttgl.constexpr, YBLOCK: ttgl.constexpr, layout_a: ttgl.constexpr,
                         layout_b: ttgl.constexpr, smem_layout: ttgl.constexpr):
    unused = ttgl.allocate_shared_memory(ttgl.int32, [XBLOCK, YBLOCK], smem_layout)
    a = ttgl.full([XBLOCK, YBLOCK], 0, ttgl.int32, layout_a)
    ttgl.static_assert(a.numel == unused.numel)
    ttgl.static_assert(unused.numel == XBLOCK * YBLOCK)
    mem = ttgl.allocate_shared_memory(ttgl.int32, a.shape, smem_layout, a)
    b = mem.load(layout_b)  # noqa: F841
    mem.store(a)
    unused._keep_alive()


@pytest.mark.parametrize("target", ALL_TARGETS)
def test_shared_memory(target):
    layout_a = ttgl.BlockedLayout(size_per_thread=[1, 1], threads_per_warp=[1, 32], warps_per_cta=[4, 1], order=[1, 0])
    layout_b = ttgl.BlockedLayout(size_per_thread=[1, 4], threads_per_warp=[1, 32], warps_per_cta=[4, 1], order=[1, 0])
    smem_layout = ttgl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=32, rank=2)
    mod = run_parser(
        shared_memory_kernel,
        *make_args(8, 32, layout_a, layout_b, smem_layout, num_warps=layout_a.warps_per_cta[0]),
        target=target,
    )
    expecttest.assert_expected_inline(
        anonymize_ir(mod.str_nodebug()), """\
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "...", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @shared_memory_kernel() attributes {noinline = false} {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable>
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0> : tensor<8x32xi32, #blocked>
    %1 = ttg.local_alloc %cst : (tensor<8x32xi32, #blocked>) -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable>
    %2 = ttg.local_load %1 : !ttg.memdesc<8x32xi32, #shared, #smem, mutable> -> tensor<8x32xi32, #blocked1>
    ttg.local_store %cst, %1 : tensor<8x32xi32, #blocked> -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable>
    ttg.local_dealloc %0 : !ttg.memdesc<8x32xi32, #shared, #smem, mutable>
    tt.return
  }
}
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
    for ivar in range(2):
        buffers.index(ivar).load(layout)


def test_tensor_memory():
    layout = ttgl.BlockedLayout(size_per_thread=[1, 64], threads_per_warp=[32, 1], warps_per_cta=[4, 1], order=[0, 1])
    tmem_layout = TensorMemoryLayout(block=[128, 128], unpacked=True)
    mod = run_parser(
        tensor_memory_kernel,
        *make_args(layout, tmem_layout, num_warps=4),
        target=BLACKWELL_TARGET,
    )
    expecttest.assert_expected_inline(
        anonymize_ir(mod.str_nodebug()), """\
#blocked = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "...", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tensor_memory_kernel() attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0> : tensor<128x128xi32, #blocked>
    %result = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xi32, #tmem, #ttng.tensor_memory, mutable>
    %result_0 = ttng.tmem_alloc %cst : (tensor<128x128xi32, #blocked>) -> !ttg.memdesc<128x128xi32, #tmem, #ttng.tensor_memory, mutable>
    %result_1 = ttng.tmem_load %result_0 : !ttg.memdesc<128x128xi32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xi32, #blocked>
    %true = arith.constant true
    ttng.tmem_store %cst, %result_0, %true : tensor<128x128xi32, #blocked> -> !ttg.memdesc<128x128xi32, #tmem, #ttng.tensor_memory, mutable>
    %0 = ttng.tmem_subslice %result_0 {N = 0 : i32} : !ttg.memdesc<128x128xi32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xi32, #tmem1, #ttng.tensor_memory, mutable, 128x128>
    %1 = ttng.tmem_subslice %result_0 {N = 64 : i32} : !ttg.memdesc<128x128xi32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xi32, #tmem1, #ttng.tensor_memory, mutable, 128x128>
    %result_2 = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %c0_i32_3 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %2 = arith.bitcast %c0_i32_3 : i32 to i32
    %3 = arith.bitcast %c2_i32 : i32 to i32
    %4 = arith.bitcast %c1_i32 : i32 to i32
    %5 = ub.poison : i32
    scf.for %arg0 = %2 to %3 step %4  : i32 {
      %6 = ttg.memdesc_index %result_2[%arg0] : !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %result_4 = ttng.tmem_load %6 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x128xf32, #blocked>
    }
    tt.return
  }
}
""")


@gluon.jit
def shared_memory_subview_kernel(XBLOCK: ttgl.constexpr, layout: ttgl.constexpr, smem_layout: ttgl.constexpr):
    XHALF: ttgl.constexpr = XBLOCK // 2
    smem = ttgl.allocate_shared_memory(ttgl.int32, [XBLOCK, XBLOCK], smem_layout)
    view = smem.slice(XHALF, XHALF, dim=1)
    value = view.load(layout)
    view = smem.slice(XHALF, XHALF, dim=0)
    view.store(value.trans())


@pytest.mark.parametrize("target", ALL_TARGETS)
def test_shared_memory_subview(target):
    layout = ttgl.BlockedLayout(size_per_thread=[1, 1], threads_per_warp=[1, 32], warps_per_cta=[4, 1], order=[1, 0])
    smem_layout = ttgl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    mod = run_parser(
        shared_memory_subview_kernel,
        *make_args(256, layout, smem_layout, num_warps=4),
        target=target,
    )
    expecttest.assert_expected_inline(
        anonymize_ir(mod.str_nodebug()), """\
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "...", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @shared_memory_subview_kernel() attributes {noinline = false} {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<256x256xi32, #shared, #smem, mutable>
    %1 = ttg.memdesc_subslice %0[0, 128] : !ttg.memdesc<256x256xi32, #shared, #smem, mutable> -> !ttg.memdesc<256x128xi32, #shared, #smem, mutable, 256x256>
    %2 = ttg.local_load %1 : !ttg.memdesc<256x128xi32, #shared, #smem, mutable, 256x256> -> tensor<256x128xi32, #blocked>
    %3 = ttg.memdesc_subslice %0[128, 0] : !ttg.memdesc<256x256xi32, #shared, #smem, mutable> -> !ttg.memdesc<128x256xi32, #shared, #smem, mutable, 256x256>
    %4 = tt.trans %2 {order = array<i32: 1, 0>} : tensor<256x128xi32, #blocked> -> tensor<128x256xi32, #blocked1>
    ttg.local_store %4, %3 : tensor<128x256xi32, #blocked1> -> !ttg.memdesc<128x256xi32, #shared, #smem, mutable, 256x256>
    tt.return
  }
}
""")


@gluon.jit
def shared_memory_index_kernel(XBLOCK: ttgl.constexpr, layout: ttgl.constexpr, smem_layout: ttgl.constexpr):
    smem = ttgl.allocate_shared_memory(ttgl.int32, [4, XBLOCK], smem_layout)
    for ivar in range(4):
        smem.index(ivar).load(layout)


@pytest.mark.parametrize("target", ALL_TARGETS)
def test_shared_memory_index(target):
    layout = ttgl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32], warps_per_cta=[4], order=[0])
    smem_layout = ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0])
    mod = run_parser(
        shared_memory_index_kernel,
        *make_args(256, layout, smem_layout, num_warps=4),
        target=target,
    )
    expecttest.assert_expected_inline(
        anonymize_ir(mod.str_nodebug()), """\
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "...", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @shared_memory_index_kernel() attributes {noinline = false} {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<4x256xi32, #shared, #smem, mutable>
    %c0_i32 = arith.constant 0 : i32
    %c4_i32 = arith.constant 4 : i32
    %c1_i32 = arith.constant 1 : i32
    %1 = arith.bitcast %c0_i32 : i32 to i32
    %2 = arith.bitcast %c4_i32 : i32 to i32
    %3 = arith.bitcast %c1_i32 : i32 to i32
    %4 = ub.poison : i32
    scf.for %arg0 = %1 to %2 step %3  : i32 {
      %5 = ttg.memdesc_index %0[%arg0] : !ttg.memdesc<4x256xi32, #shared, #smem, mutable> -> !ttg.memdesc<256xi32, #shared, #smem, mutable, 4x256>
      %6 = ttg.local_load %5 : !ttg.memdesc<256xi32, #shared, #smem, mutable, 4x256> -> tensor<256xi32, #blocked>
    }
    tt.return
  }
}
""")


@gluon.jit
def shared_memory_cast_kernel():
    layout_a: ttgl.constexpr = ttgl.NVMMASharedLayout(swizzle_byte_width=64, transposed=False, element_bitwidth=8,
                                                      rank=2)
    layout_T: ttgl.constexpr = ttgl.NVMMASharedLayout(swizzle_byte_width=64, transposed=True, element_bitwidth=8,
                                                      rank=2)
    smem = ttgl.allocate_shared_memory(ttgl.int8, [2, 256, 128], layout_a)
    perm = smem.index(0).permute((1, 0))
    ttgl.static_assert(perm.type.layout == layout_T)
    # Check that the MLIR type and Gluon types match by emitting a call.
    anchor_noinline(perm)

    layout_b: ttgl.constexpr = ttgl.NVMMASharedLayout(swizzle_byte_width=64, transposed=False, element_bitwidth=16,
                                                      rank=4, cta_order=[3, 2, 1, 0])
    smem = ttgl.allocate_shared_memory(ttgl.float16, [32, 1, 4, 64], layout_b)
    smem.reshape((128, 64))

    smem._reinterpret(ttgl.int8, [1024], ttgl.SwizzledSharedLayout(1, 1, 1, [0, 1]))


@pytest.mark.parametrize("target", ALL_TARGETS)
def test_shared_memory_cast(target):
    mod = run_parser(shared_memory_cast_kernel, target=target)
    expecttest.assert_expected_inline(
        anonymize_ir(mod.str_nodebug()), """\
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
    %1 = ttg.memdesc_index %0[%c0_i32] : !ttg.memdesc<2x256x128xi8, #shared, #smem, mutable> -> !ttg.memdesc<256x128xi8, #shared, #smem, mutable, 2x256x128>
    %2 = ttg.memdesc_trans %1 {order = array<i32: 1, 0>} : !ttg.memdesc<256x128xi8, #shared, #smem, mutable, 2x256x128> -> !ttg.memdesc<128x256xi8, #shared1, #smem, mutable, 2x128x256>
    tt.call @"test_frontend.anchor_noinline__MDi8S128_256SLNVMMA_64_8_True_False_NVMMALAS[2, 128, 256]ASMD__"(%2) : (!ttg.memdesc<128x256xi8, #shared1, #smem, mutable, 2x128x256>) -> ()
    %3 = ttg.local_alloc : () -> !ttg.memdesc<32x1x4x64xf16, #shared2, #smem, mutable>
    %4 = ttg.memdesc_reshape %3 : !ttg.memdesc<32x1x4x64xf16, #shared2, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared3, #smem, mutable>
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
    a, b = ttgl.warp_specialize((pair, c, e), warp_specialize_default, (pair, c, e),
                                [warp_specialize_worker0, warp_specialize_worker1], [4, 4], [24, 48])
    anchor(a)
    anchor(b)

    # CHECK: ttg.warp_specialize([[A]], [[B]], [[C]])
    # CHECK: (tensor<1xi32, [[BLOCKED]]>, tensor<2xi32, [[BLOCKED]]>, tensor<4xi32, [[BLOCKED]]>) -> ()
    ttgl.warp_specialize((pair, c, e), warp_specialize_worker0, (pair, c, e), [warp_specialize_worker1], [4], [48])


@gluon.jit
def ws_body(num_warps: ttgl.constexpr):
    anchor(ttgl.arange(0, 128, layout=ttgl.BlockedLayout([1], [32], [num_warps], [0])))


@gluon.jit
def ws_test_default():
    ws_body(4)


@gluon.jit
def ws_test_worker0():
    ws_body(2)


@gluon.jit
def ws_test_worker1():
    ws_body(1)


@filecheck_test
@gluon.jit
def test_num_warps_caller_context():
    # CHECK-DAG: [[BLOCKED_NW4:#.*]] = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
    # CHECK-DAG: [[BLOCKED_NW2:#.*]] = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>
    # CHECK-DAG: [[BLOCKED_NW1:#.*]] = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>

    # CHECK: func private @{{.*}}ws_test_default{{.*}}() attributes {noinline = false}
    # CHECK: func private @{{.*}}ws_body{{.*}}() attributes {noinline = false}
    # CHECK: func private @{{.*}}anchor{{.*}}(%arg0: tensor<128xi32, [[BLOCKED_NW4]]>) attributes {noinline = false}

    # CHECK: func private @{{.*}}ws_test_worker0{{.*}}_NW2() attributes {noinline = false, "ttg.num-warps" = 2 : i32}
    # CHECK: func private @{{.*}}ws_body{{.*}}_NW2"() attributes {noinline = false, "ttg.num-warps" = 2 : i32}
    # CHECK: func private @{{.*}}anchor{{.*}}_NW2(%arg0: tensor<128xi32, [[BLOCKED_NW2]]>) attributes {noinline = false, "ttg.num-warps" = 2 : i32}

    # CHECK: func private @{{.*}}ws_test_worker1{{.*}}_NW1() attributes {noinline = false, "ttg.num-warps" = 1 : i32}
    # CHECK: func private @{{.*}}ws_body{{.*}}_NW1"() attributes {noinline = false, "ttg.num-warps" = 1 : i32}
    # CHECK: func private @{{.*}}anchor{{.*}}_NW1(%arg0: tensor<128xi32, [[BLOCKED_NW1]]>) attributes {noinline = false, "ttg.num-warps" = 1 : i32}
    ttgl.warp_specialize((), ws_test_default, (), [ws_test_worker0, ws_test_worker1], [2, 1], [80, 80])


@gluon.jit
def mbarrier_kernel():
    bar = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    mbarrier.expect(bar, 4)
    mbarrier.arrive(bar, count=1)
    phase = 0
    mbarrier.wait(bar, phase, deps=[bar])
    mbarrier.invalidate(bar)


@pytest.mark.parametrize("target", [HOPPER_TARGET, BLACKWELL_TARGET])
def test_mbarrier(target):
    mod = run_parser(mbarrier_kernel, target=target)
    expecttest.assert_expected_inline(
        anonymize_ir(mod.str_nodebug()), """\
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "...", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @mbarrier_kernel() attributes {noinline = false} {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %0, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %true = arith.constant true
    ttng.barrier_expect %0, 4, %true : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %true_0 = arith.constant true
    ttng.arrive_barrier %0, 1, %true_0 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %c0_i32 = arith.constant 0 : i32
    %true_1 = arith.constant true
    ttng.wait_barrier %0, %c0_i32, %true_1 deps %0 : !ttg.memdesc<1xi64, #shared, #smem, mutable>, !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.inval_barrier %0 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }
}
""")


@gluon.jit
def tcgen05_mma_kernel(nvmma_layout: ttgl.constexpr, acc_layout: ttgl.constexpr):
    a = ttgl.allocate_shared_memory(ttgl.float16, [128, 128], nvmma_layout)
    b = ttgl.allocate_shared_memory(ttgl.float16, [128, 128], nvmma_layout)
    acc = blackwell.allocate_tensor_memory(ttgl.float16, [128, 128], acc_layout)
    blackwell.tcgen05_mma(a, b, acc)


def test_tcgen05_mma():
    nvmma_layout = ttgl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2)
    acc_layout = TensorMemoryLayout([128, 128], unpacked=True)

    mod = run_parser(tcgen05_mma_kernel, *make_args(nvmma_layout, acc_layout), target=BLACKWELL_TARGET)
    expecttest.assert_expected_inline(
        anonymize_ir(mod.str_nodebug()), """\
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "...", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tcgen05_mma_kernel() attributes {noinline = false} {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %result = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    %true = arith.constant true
    %true_0 = arith.constant true
    %2 = ttng.tc_gen5_mma %0, %1, %result[], %true, %true_0 : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}
""")


@gluon.jit
def tcgen05_mma_mbar_kernel(nvmma_layout: ttgl.constexpr, acc_layout: ttgl.constexpr):
    a = ttgl.allocate_shared_memory(ttgl.float16, [128, 128], nvmma_layout)
    b = ttgl.allocate_shared_memory(ttgl.float16, [128, 128], nvmma_layout)
    bar = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    acc = blackwell.allocate_tensor_memory(ttgl.float16, [128, 128], acc_layout)
    blackwell.tcgen05_mma(a, b, acc, mbarriers=[bar])


def test_tcgen05_mma_mbar():
    nvmma_layout = ttgl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2)
    acc_layout = TensorMemoryLayout([128, 128], unpacked=True)

    mod = run_parser(tcgen05_mma_mbar_kernel, *make_args(nvmma_layout, acc_layout), target=BLACKWELL_TARGET)
    expecttest.assert_expected_inline(
        anonymize_ir(mod.str_nodebug()), """\
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "...", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tcgen05_mma_mbar_kernel() attributes {noinline = false} {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %2 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %result = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    %true = arith.constant true
    %true_0 = arith.constant true
    %true_1 = arith.constant true
    %3 = ttng.tc_gen5_mma %0, %1, %result[], %true, %true_0, %2[%true_1] {is_async} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    tt.return
  }
}
""")


@filecheck_test
@gluon.jit
def test_tcgen05_copy():
    # CHECK-LABEL: test_tcgen05_copy
    smem_h: ttgl.constexpr = 256
    num_cols: ttgl.constexpr = smem_h * 4 // 32

    shared_layout: ttgl.constexpr = ttgl.NVMMASharedLayout(swizzle_byte_width=0, element_bitwidth=8, rank=2)
    tmem_layout: ttgl.constexpr = TensorMemoryScalesLayout()
    # CHECK: [[SRC:%.*]] = ttg.local_alloc
    src = ttgl.allocate_shared_memory(ttgl.int8, [smem_h, 4], shared_layout)
    # CHECK: [[DST:%.*]] = ttng.tmem_alloc
    dst = blackwell.allocate_tensor_memory(ttgl.int8, [128, num_cols], tmem_layout)
    # CHECK: ttng.tmem_copy [[SRC]], [[DST]]
    blackwell.tcgen05_copy(src, dst)


@filecheck_test
@gluon.jit
def test_tcgen05_commit():
    # CHECK-LABEL: test_tcgen05_commit
    barrier = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    # CHECK: [[BARRIER:%.*]] = ttg.local_alloc
    # CHECK: ttng.tc_gen5_commit [[BARRIER]]
    blackwell.tcgen05_commit(barrier)


@gluon.jit
def warpgroup_mma_kernel(nvmma_layout: ttgl.constexpr, acc_layout: ttgl.constexpr):
    a = ttgl.allocate_shared_memory(ttgl.float16, [128, 128], nvmma_layout)
    b = ttgl.allocate_shared_memory(ttgl.float16, [128, 128], nvmma_layout)
    acc = ttgl.full([128, 128], 0, dtype=ttgl.float16, layout=acc_layout)
    acc = hopper.warpgroup_mma(a, b, acc)
    ttgl.static_assert(isinstance(acc, ttgl.tensor))

    acc = hopper.warpgroup_mma(a, b, acc, is_async=True)
    ttgl.static_assert(isinstance(acc, hopper.warpgroup_mma_accumulator))


def test_warpgroup_mma():
    nvmma_layout = ttgl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2)
    mma_layout = ttgl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, 32, 16])
    mod = run_parser(
        warpgroup_mma_kernel,
        *make_args(nvmma_layout, mma_layout),
        target=HOPPER_TARGET,
    )
    expecttest.assert_expected_inline(
        anonymize_ir(mod.str_nodebug()), """\
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 32, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "...", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @warpgroup_mma_kernel() attributes {noinline = false} {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %cst = arith.constant 0.000000e+00 : f16
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf16, #mma>
    %true = arith.constant true
    %2 = ttng.warp_group_dot %0, %1, %cst_0, %true {inputPrecision = 0 : i32} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> * !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #mma>
    %true_1 = arith.constant true
    %3 = ttng.warp_group_dot %0, %1, %2, %true_1 {inputPrecision = 0 : i32, isAsync = true} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> * !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #mma>
    tt.return
  }
}
""")


@gluon.jit
def warpgroup_mma_wait_kernel():
    layout: ttgl.constexpr = ttgl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, 32, 16])
    acc = hopper.warpgroup_mma_init(ttgl.full([128, 128], 0, dtype=ttgl.float16, layout=layout))
    acc = hopper.warpgroup_mma_wait(num_outstanding=1, deps=[acc])
    _ = acc + acc


def test_warpgroup_mma_wait():
    mod = run_parser(warpgroup_mma_wait_kernel, target=HOPPER_TARGET)
    expecttest.assert_expected_inline(
        anonymize_ir(mod.str_nodebug()), """\
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 32, 16]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "...", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @warpgroup_mma_wait_kernel() attributes {noinline = false} {
    %cst = arith.constant 0.000000e+00 : f16
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf16, #mma>
    %0 = ttng.warp_group_dot_wait %cst_0 {pendings = 1 : i32} : tensor<128x128xf16, #mma>
    %1 = arith.addf %0, %0 : tensor<128x128xf16, #mma>
    tt.return
  }
}
""")


@gluon.jit
def async_tma_kernel(input_desc, XBLOCK: ttgl.constexpr):
    smem = ttgl.allocate_shared_memory(ttgl.float16, [XBLOCK, XBLOCK], input_desc.layout)
    bar = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)

    tma.async_copy_global_to_shared(input_desc, [0, 0], bar, smem)
    ttgl.static_assert(input_desc.block_type.nbytes == XBLOCK * XBLOCK * 2)
    mbarrier.expect(bar, input_desc.block_type.nbytes)
    mbarrier.wait(bar, 0)

    mbarrier.invalidate(bar)

    tma.async_copy_shared_to_global(input_desc, [0, 0], smem)
    tma.store_wait(0)


@pytest.mark.parametrize("target", [HOPPER_TARGET, BLACKWELL_TARGET])
def test_async_tma(target):
    input = MockTensor(ttgl.float16, (1024, 1024))
    XBLOCK = 128
    shared_layout = ttgl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2)
    input_desc = TensorDescriptor.from_tensor(input, [XBLOCK, XBLOCK], shared_layout)

    mod = run_parser(
        async_tma_kernel,
        *make_args(input_desc, XBLOCK, num_warps=4),
        target=target,
    )
    expecttest.assert_expected_inline(
        anonymize_ir(mod.str_nodebug()), """\
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "...", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @async_tma_kernel(%arg0: !tt.tensordesc<tensor<128x128xf16, #shared>>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64) attributes {noinline = false} {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %1, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %true = arith.constant true
    ttng.async_tma_copy_global_to_local %arg0[%c0_i32, %c0_i32_0] %0, %1, %true : !tt.tensordesc<tensor<128x128xf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %true_1 = arith.constant true
    ttng.barrier_expect %1, 32768, %true_1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %c0_i32_2 = arith.constant 0 : i32
    %true_3 = arith.constant true
    ttng.wait_barrier %1, %c0_i32_2, %true_3 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %c0_i32_4 = arith.constant 0 : i32
    %c0_i32_5 = arith.constant 0 : i32
    ttng.async_tma_copy_local_to_global %arg0[%c0_i32_4, %c0_i32_5] %0 : !tt.tensordesc<tensor<128x128xf16, #shared>>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    ttng.async_tma_store_wait {pendings = 0 : i32}
    tt.return
  }
}
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


def test_async_tma_blackwell():
    input = MockTensor(ttgl.float16, (1024, 1024))
    XBLOCK = 128
    shared_layout = ttgl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2)
    input_desc = TensorDescriptor.from_tensor(input, [1, XBLOCK], shared_layout)

    mod = run_parser(
        async_tma_blackwell_kernel,
        *make_args(input_desc, XBLOCK, num_warps=4),
        target=BLACKWELL_TARGET,
    )
    expecttest.assert_expected_inline(
        anonymize_ir(mod.str_nodebug()), """\
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "...", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @async_tma_blackwell_kernel(%arg0: !tt.tensordesc<tensor<1x128xf16, #shared>>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64) attributes {noinline = false} {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %1, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    ttng.async_tma_gather %arg0[%2, %c0_i32] %0, %1, %true : !tt.tensordesc<tensor<1x128xf16, #shared>>, tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, i32, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, i1
    %true_0 = arith.constant true
    ttng.barrier_expect %1, 32768, %true_0 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %c0_i32_1 = arith.constant 0 : i32
    %true_2 = arith.constant true
    ttng.wait_barrier %1, %c0_i32_1, %true_2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %c0_i32_3 = arith.constant 0 : i32
    ttng.async_tma_scatter %arg0[%2, %c0_i32_3] %0 : !tt.tensordesc<tensor<1x128xf16, #shared>>, tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, i32, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    ttng.async_tma_store_wait {pendings = 0 : i32}
    tt.return
  }
}
""")


def test_mlir_attr_error():

    @gluon.jit
    def kernel():
        ttgl.arange(0, 1, layout=ttgl.BlockedLayout([1], [32], [4], [1]))

    with pytest.raises(CompilationError) as e:
        run_parser(kernel)

    assert "order must be a permutation of 0..(rank-1), but was [1]" in str(e.value.__cause__)


def test_tensor_layout_type_changed():

    @gluon.jit
    def kernel():
        layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1, 1], threads_per_warp=[1, 32],
                                                    warps_per_cta=[1, 4], order=[1, 0])
        x = ttgl.zeros([128], ttgl.float32)
        y = ttgl.zeros([128, 128], ttgl.float32, layout=layout)
        c = ttgl.to_tensor(True)
        while c:
            x = x + y.sum(axis=0)

    with pytest.raises(CompilationError) as e:
        run_parser(kernel)

    assert "Loop-carried variable x has initial type" in str(e.value)


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
    %0 = ttg.memdesc_index %result[%c0_i32] : !ttg.memdesc<2x256x256xi32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<256x256xi32, #tmem, #ttng.tensor_memory, mutable, 2x256x256>
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
    tt.call @"test_frontend.smem_and_layout_user__MDi32S32_32SLSSS_1_1_1_1_0_1_1_1_1_1_0_SSSLAS[32, 32]ASMD__(1,)cconstexpr_SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=(1 ,0), ctas_per_cga=_1, 1_, cta_split_num=_1, 1_, cta_order=_1, 0_)_"(%0) : (!ttg.memdesc<32x32xi32, #shared, #smem, mutable>) -> ()
    tt.return
  }
  tt.func private @"test_frontend.smem_and_layout_user__MDi32S32_32SLSSS_1_1_1_1_0_1_1_1_1_1_0_SSSLAS[32, 32]ASMD__(1,)cconstexpr_SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=(1 ,0), ctas_per_cga=_1, 1_, cta_split_num=_1, 1_, cta_order=_1, 0_)_"(%arg0: !ttg.memdesc<32x32xi32, #shared, #smem, mutable>) attributes {noinline = false} {
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


@pytest.mark.parametrize("target", ALL_TARGETS)
def test_broadcast(target):
    mod = run_parser(broadcast_kernel, target=target)
    expecttest.assert_expected_inline(
        anonymize_ir(mod.str_nodebug()), """\
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "...", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @broadcast_kernel() attributes {noinline = false} {
    %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x16xi32, #blocked>
    %2 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %3 = tt.expand_dims %2 {axis = 1 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<16x1xi32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %cst = arith.constant dense<0> : tensor<1x16xi32, #blocked>
    %4 = arith.addi %cst, %1 : tensor<1x16xi32, #blocked>
    %5 = tt.broadcast %4 : tensor<1x16xi32, #blocked> -> tensor<16x16xi32, #blocked>
    %6 = tt.broadcast %3 : tensor<16x1xi32, #blocked> -> tensor<16x16xi32, #blocked>
    %7 = arith.addi %5, %6 : tensor<16x16xi32, #blocked>
    tt.return
  }
}
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


@pytest.mark.parametrize("target", ALL_TARGETS)
def test_math(target):
    mod = run_parser(math_kernel, target=target)
    expecttest.assert_expected_inline(
        anonymize_ir(mod.str_nodebug()), """\
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "...", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @math_kernel() attributes {noinline = false} {
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant dense<1.000000e+00> : tensor<16x16xf32, #blocked>
    %cst_1 = arith.constant 2.000000e+00 : f32
    %cst_2 = arith.constant dense<2.000000e+00> : tensor<16x16xf32, #blocked>
    %cst_3 = arith.constant 4.000000e+00 : f32
    %cst_4 = arith.constant dense<4.000000e+00> : tensor<16x16xf32, #blocked>
    %c1_i32 = arith.constant 1 : i32
    %cst_5 = arith.constant dense<1> : tensor<16x16xi32, #blocked>
    %c1_i32_6 = arith.constant 1 : i32
    %cst_7 = arith.constant dense<1> : tensor<16x16xi32, #blocked>
    %0 = tt.mulhiui %cst_5, %cst_7 : tensor<16x16xi32, #blocked>
    %1 = math.exp %cst_0 : tensor<16x16xf32, #blocked>
    %2 = math.exp2 %cst_0 : tensor<16x16xf32, #blocked>
    %3 = math.log %cst_0 : tensor<16x16xf32, #blocked>
    %4 = math.log2 %cst_0 : tensor<16x16xf32, #blocked>
    %5 = math.cos %cst_0 : tensor<16x16xf32, #blocked>
    %6 = math.sin %cst_0 : tensor<16x16xf32, #blocked>
    %7 = math.sqrt %cst_0 : tensor<16x16xf32, #blocked>
    %8 = tt.precise_sqrt %cst_0 : tensor<16x16xf32, #blocked>
    %9 = math.rsqrt %cst_0 : tensor<16x16xf32, #blocked>
    %10 = math.absf %cst_0 : tensor<16x16xf32, #blocked>
    %11 = arith.divf %cst_0, %cst_2 : tensor<16x16xf32, #blocked>
    %12 = tt.precise_divf %cst_0, %cst_2 : tensor<16x16xf32, #blocked>
    %13 = math.erf %cst_0 : tensor<16x16xf32, #blocked>
    %14 = math.floor %cst_0 : tensor<16x16xf32, #blocked>
    %15 = math.ceil %cst_0 : tensor<16x16xf32, #blocked>
    %16 = math.fma %cst_0, %cst_2, %cst_4 : tensor<16x16xf32, #blocked>
    tt.return
  }
}
""")


@gluon.jit
def libdevice_kernel():
    layout: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [1, 32], [4, 1], [1, 0])
    a = ttgl.full([4, 32], 1, ttgl.float32, layout)
    b = ttgl.full([4, 32], 2, ttgl.float32, layout)
    c = ttgl.full([4, 32], 4, ttgl.float32, layout)

    libdevice.abs(a)
    libdevice.fast_dividef(a, b)
    libdevice.fma(a, b, c)

    libdevice.isnan(a)
    libdevice.isinf(a)


@pytest.mark.parametrize("target", ALL_TARGETS)
def test_libdevice(target):
    mod = run_parser(libdevice_kernel, target=target)
    expecttest.assert_expected_inline(
        anonymize_ir(mod.str_nodebug()), """\
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "...", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @libdevice_kernel() attributes {noinline = false} {
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant dense<1.000000e+00> : tensor<4x32xf32, #blocked>
    %cst_1 = arith.constant 2.000000e+00 : f32
    %cst_2 = arith.constant dense<2.000000e+00> : tensor<4x32xf32, #blocked>
    %cst_3 = arith.constant 4.000000e+00 : f32
    %cst_4 = arith.constant dense<4.000000e+00> : tensor<4x32xf32, #blocked>
    %0 = tt.extern_elementwise %cst_0 {libname = "", libpath = "", pure = true, symbol = "..."} : (tensor<4x32xf32, #blocked>) -> tensor<4x32xf32, #blocked>
    %1 = tt.extern_elementwise %cst_0, %cst_2 {libname = "", libpath = "", pure = true, symbol = "..."} : (tensor<4x32xf32, #blocked>, tensor<4x32xf32, #blocked>) -> tensor<4x32xf32, #blocked>
    %2 = tt.extern_elementwise %cst_0, %cst_2, %cst_4 {libname = "", libpath = "", pure = true, symbol = "..."} : (tensor<4x32xf32, #blocked>, tensor<4x32xf32, #blocked>, tensor<4x32xf32, #blocked>) -> tensor<4x32xf32, #blocked>
    %3 = tt.extern_elementwise %cst_0 {libname = "", libpath = "", pure = true, symbol = "..."} : (tensor<4x32xf32, #blocked>) -> tensor<4x32xi32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %cst_5 = arith.constant dense<0> : tensor<4x32xi32, #blocked>
    %4 = arith.cmpi ne, %3, %cst_5 : tensor<4x32xi32, #blocked>
    %5 = tt.extern_elementwise %cst_0 {libname = "", libpath = "", pure = true, symbol = "..."} : (tensor<4x32xf32, #blocked>) -> tensor<4x32xi32, #blocked>
    %c0_i32_6 = arith.constant 0 : i32
    %cst_7 = arith.constant dense<0> : tensor<4x32xi32, #blocked>
    %6 = arith.cmpi ne, %5, %cst_7 : tensor<4x32xi32, #blocked>
    tt.return
  }
}
""")


@gluon.jit
def libdevice_implicit_broadcast_kernel():
    layout: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [1, 32], [4, 1], [1, 0])
    a = ttgl.full([4, 32], 1, ttgl.float32, layout)
    b = ttgl.full([32], 2, ttgl.float32, ttgl.SliceLayout(0, layout))[None, :]
    c = ttgl.full([4], 4, ttgl.float32, ttgl.SliceLayout(1, layout))[:, None]
    libdevice.abs(a)
    libdevice.fast_dividef(a, b)
    libdevice.fma(a, b, c)


@pytest.mark.parametrize("target", ALL_TARGETS)
def test_libdevice_implicit_broadcast(target):
    mod = run_parser(libdevice_implicit_broadcast_kernel, target=target)
    expecttest.assert_expected_inline(
        anonymize_ir(mod.str_nodebug()), """\
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "...", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @libdevice_implicit_broadcast_kernel() attributes {noinline = false} {
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant dense<1.000000e+00> : tensor<4x32xf32, #blocked>
    %cst_1 = arith.constant 2.000000e+00 : f32
    %cst_2 = arith.constant dense<2.000000e+00> : tensor<32xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %0 = tt.expand_dims %cst_2 {axis = 0 : i32} : tensor<32xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xf32, #blocked>
    %cst_3 = arith.constant 4.000000e+00 : f32
    %cst_4 = arith.constant dense<4.000000e+00> : tensor<4xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %1 = tt.expand_dims %cst_4 {axis = 1 : i32} : tensor<4xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<4x1xf32, #blocked>
    %2 = tt.extern_elementwise %cst_0 {libname = "", libpath = "", pure = true, symbol = "..."} : (tensor<4x32xf32, #blocked>) -> tensor<4x32xf32, #blocked>
    %3 = tt.broadcast %0 : tensor<1x32xf32, #blocked> -> tensor<4x32xf32, #blocked>
    %4 = tt.broadcast %0 : tensor<1x32xf32, #blocked> -> tensor<4x32xf32, #blocked>
    %5 = tt.extern_elementwise %cst_0, %4 {libname = "", libpath = "", pure = true, symbol = "..."} : (tensor<4x32xf32, #blocked>, tensor<4x32xf32, #blocked>) -> tensor<4x32xf32, #blocked>
    %6 = tt.broadcast %0 : tensor<1x32xf32, #blocked> -> tensor<4x32xf32, #blocked>
    %7 = tt.broadcast %1 : tensor<4x1xf32, #blocked> -> tensor<4x32xf32, #blocked>
    %8 = tt.broadcast %0 : tensor<1x32xf32, #blocked> -> tensor<4x32xf32, #blocked>
    %9 = tt.broadcast %1 : tensor<4x1xf32, #blocked> -> tensor<4x32xf32, #blocked>
    %10 = tt.extern_elementwise %cst_0, %8, %9 {libname = "", libpath = "", pure = true, symbol = "..."} : (tensor<4x32xf32, #blocked>, tensor<4x32xf32, #blocked>, tensor<4x32xf32, #blocked>) -> tensor<4x32xf32, #blocked>
    tt.return
  }
}
""")


@gluon.jit
def pair_add(a0, a1, b0, b1):
    return a0 + b0, a1 + b1


@gluon.jit
def reduce_kernel(out):
    layout: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [1, 32], [4, 1], [1, 0])
    a = ttgl.full([16, 16], 1, ttgl.float32, layout)
    b = ttgl.full([16, 16], 2, ttgl.float32, layout)
    s0 = a.sum(0)
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
    ttgl.store(out + ttgl.arange(0, 16, s0.type.layout), result)


@pytest.mark.parametrize("target", ALL_TARGETS)
def test_reduce(target):
    mod = run_parser(reduce_kernel, *make_args(MockTensor(ttgl.float32)), target=target)
    expecttest.assert_expected_inline(
        anonymize_ir(mod.str_nodebug()), """\
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "...", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reduce_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant dense<1.000000e+00> : tensor<16x16xf32, #blocked>
    %cst_1 = arith.constant 2.000000e+00 : f32
    %cst_2 = arith.constant dense<2.000000e+00> : tensor<16x16xf32, #blocked>
    %0 = tt.call @"triton.language.standard.sum__fp32S16_16SLB1_1B1_32B4_1B1_0B1_1B1_1B1_0BL__(1,)cconstexpr_0__(2,)cconstexpr_False__(3,)cNone"(%cst_0) : (tensor<16x16xf32, #blocked>) -> tensor<16xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %1 = tt.call @"triton.language.standard.sum__fp32S16_16SLB1_1B1_32B4_1B1_0B1_1B1_1B1_0BL__(1,)cconstexpr_1__(2,)cconstexpr_False__(3,)cNone"(%cst_0) : (tensor<16x16xf32, #blocked>) -> tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %2 = tt.call @"triton.language.standard.max__fp32S16SLSL0_B1_1B1_32B4_1B1_0B1_1B1_1B1_0BSLL__(1,)cconstexpr_0__(2,)cconstexpr_False__(3,)cconstexpr_True__(4,)cconstexpr_False_"(%0) : (tensor<16xf32, #ttg.slice<{dim = 0, parent = #blocked}>>) -> f32
    %3 = ttg.convert_layout %1 : tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<16xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %4:2 = "tt.reduce"(%cst_0, %cst_2) <{axis = 0 : i32}> ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32):
      %12:2 = tt.call @test_frontend.pair_add__fp32_fp32_fp32_fp32__(%arg1, %arg2, %arg3, %arg4) : (f32, f32, f32, f32) -> (f32, f32)
      tt.reduce.return %12#0, %12#1 : f32, f32
    }) : (tensor<16x16xf32, #blocked>, tensor<16x16xf32, #blocked>) -> (tensor<16xf32, #ttg.slice<{dim = 0, parent = #blocked}>>, tensor<16xf32, #ttg.slice<{dim = 0, parent = #blocked}>>)
    %5 = tt.splat %2 : f32 -> tensor<16xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %6 = arith.addf %5, %3 : tensor<16xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %7 = arith.addf %6, %4#0 : tensor<16xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %8 = arith.addf %7, %4#1 : tensor<16xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %9 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %10 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>, #ttg.slice<{dim = 0, parent = #blocked}>>
    %11 = tt.addptr %10, %9 : tensor<16x!tt.ptr<f32>, #ttg.slice<{dim = 0, parent = #blocked}>>, tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    tt.store %11, %8 : tensor<16x!tt.ptr<f32>, #ttg.slice<{dim = 0, parent = #blocked}>>
    tt.return
  }
  tt.func private @"triton.language.standard.sum__fp32S16_16SLB1_1B1_32B4_1B1_0B1_1B1_1B1_0BL__(1,)cconstexpr_0__(2,)cconstexpr_False__(3,)cNone"(%arg0: tensor<16x16xf32, #blocked>) -> tensor<16xf32, #ttg.slice<{dim = 0, parent = #blocked}>> attributes {noinline = false} {
    %0 = "tt.reduce"(%arg0) <{axis = 0 : i32}> ({
    ^bb0(%arg1: f32, %arg2: f32):
      %2 = tt.call @triton.language.standard._sum_combine__fp32_fp32__(%arg1, %arg2) : (f32, f32) -> f32
      tt.reduce.return %2 : f32
    }) : (tensor<16x16xf32, #blocked>) -> tensor<16xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    tt.return %0 : tensor<16xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
  ^bb1:  // no predecessors
    %1 = ub.poison : tensor<16xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    tt.return %1 : tensor<16xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
  }
  tt.func private @triton.language.standard._sum_combine__fp32_fp32__(%arg0: f32, %arg1: f32) -> f32 attributes {noinline = false} {
    %0 = arith.addf %arg0, %arg1 : f32
    tt.return %0 : f32
  ^bb1:  // no predecessors
    %1 = ub.poison : f32
    tt.return %1 : f32
  }
  tt.func private @"triton.language.standard.sum__fp32S16_16SLB1_1B1_32B4_1B1_0B1_1B1_1B1_0BL__(1,)cconstexpr_1__(2,)cconstexpr_False__(3,)cNone"(%arg0: tensor<16x16xf32, #blocked>) -> tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked}>> attributes {noinline = false} {
    %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
    ^bb0(%arg1: f32, %arg2: f32):
      %2 = tt.call @triton.language.standard._sum_combine__fp32_fp32__(%arg1, %arg2) : (f32, f32) -> f32
      tt.reduce.return %2 : f32
    }) : (tensor<16x16xf32, #blocked>) -> tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    tt.return %0 : tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
  ^bb1:  // no predecessors
    %1 = ub.poison : tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    tt.return %1 : tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
  }
  tt.func private @"triton.language.standard.max__fp32S16SLSL0_B1_1B1_32B4_1B1_0B1_1B1_1B1_0BSLL__(1,)cconstexpr_0__(2,)cconstexpr_False__(3,)cconstexpr_True__(4,)cconstexpr_False_"(%arg0: tensor<16xf32, #ttg.slice<{dim = 0, parent = #blocked}>>) -> f32 attributes {noinline = false} {
    %0 = "tt.reduce"(%arg0) <{axis = 0 : i32}> ({
    ^bb0(%arg1: f32, %arg2: f32):
      %2 = tt.call @triton.language.standard._elementwise_max__fp32_fp32__(%arg1, %arg2) : (f32, f32) -> f32
      tt.reduce.return %2 : f32
    }) : (tensor<16xf32, #ttg.slice<{dim = 0, parent = #blocked}>>) -> f32
    tt.return %0 : f32
  ^bb1:  // no predecessors
    %1 = ub.poison : f32
    tt.return %1 : f32
  }
  tt.func private @triton.language.standard._elementwise_max__fp32_fp32__(%arg0: f32, %arg1: f32) -> f32 attributes {noinline = false} {
    %0 = arith.maxnumf %arg0, %arg1 : f32
    tt.return %0 : f32
  ^bb1:  // no predecessors
    %1 = ub.poison : f32
    tt.return %1 : f32
  }
  tt.func private @test_frontend.pair_add__fp32_fp32_fp32_fp32__(%arg0: f32, %arg1: f32, %arg2: f32, %arg3: f32) -> (f32, f32) attributes {noinline = false} {
    %0 = arith.addf %arg0, %arg2 : f32
    %1 = arith.addf %arg1, %arg3 : f32
    tt.return %0, %1 : f32, f32
  ^bb1:  // no predecessors
    %2 = ub.poison : f32
    %3 = ub.poison : f32
    tt.return %2, %3 : f32, f32
  }
}
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
    ll: ttgl.constexpr = ttgl.DistributedLinearLayout(reg_bases=[[1]], lane_bases=[[2], [4], [8], [16], [32]],
                                                      warp_bases=[[64], [128]], block_bases=[], shape=[256])
    ttgl.arange(0, 256, layout=ll)


@pytest.mark.parametrize("target", ALL_TARGETS)
def test_linear_layout(target):
    mod = run_parser(linear_layout_kernel, target=target)
    expecttest.assert_expected_inline(
        anonymize_ir(mod.str_nodebug()), """\
#linear = #ttg.linear<{register = [[1]], lane = [[2], [4], [8], [16], [32]], warp = [[64], [128]], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "...", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @linear_layout_kernel() attributes {noinline = false} {
    %0 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #linear>
    tt.return
  }
}
""")


@filecheck_test
@gluon.jit
def test_dot_operand_layout():
    # CHECK: [[NVMMA:#.*]] = #ttg.nvidia_mma
    # CHECK: test_dot_operand_layout
    mma_layout: ttgl.constexpr = ttgl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1],
                                                             instr_shape=[16, 32, 16])
    layout: ttgl.constexpr = ttgl.DotOperandLayout(operand_index=0, parent=mma_layout, k_width=2)
    # CHECK: arith.constant {{.*}} tensor<256x128xf16, #ttg.dot_op<{opIdx = 0, parent = [[NVMMA]], kWidth = 2}>>
    x = ttgl.full([256, 128], 0.0, ttgl.float16, layout)
    y = x.sum(axis=1)
    ttgl.static_assert(y.type.layout.parent == layout)


@filecheck_test
@gluon.jit
def test_tensor_permute():
    # CHECK-DAG: [[BLOCKED:#.*]] = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
    # CHECK-DAG: [[BLOCKED1:#.*]] = #ttg.blocked<{sizePerThread = [2, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
    layout: ttgl.constexpr = ttgl.BlockedLayout([1, 2], [4, 8], [4, 1], [1, 0])
    a = ttgl.full([32, 16], 0, ttgl.int32, layout=layout)
    # CHECK: tt.trans{{.*}} : tensor<32x16xi32, [[BLOCKED]]> -> tensor<16x32xi32, [[BLOCKED1]]>
    res = ttgl.permute(a, [1, 0])
    permuted_layout: ttgl.constexpr = ttgl.BlockedLayout([2, 1], [8, 4], [1, 4], [0, 1])
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
    expect_layout: ttgl.constexpr = ttgl.BlockedLayout([2, 2], [32, 1], [4, 1], [1, 0])
    ttgl.static_assert(res.type.layout == expect_layout)

    # CHECK: tt.split {{.*}} : tensor<128x2xi32, [[BLOCKED1]]> -> tensor<128xi32, #ttg.slice<{dim = 1, parent = [[BLOCKED1]]}>>
    c, d = ttgl.split(res)
    ttgl.static_assert(c.type.layout == ttgl.SliceLayout(1, expect_layout))
    ttgl.static_assert(d.type.layout == ttgl.SliceLayout(1, expect_layout))


@filecheck_test
@gluon.jit
def test_reshape_linear_layout():
    # CHECK: [[BLOCKED:#.*]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
    # CHECK: [[LINEAR:#.*]] = #ttg.linear
    layout: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [32, 1], [4, 1], [0, 1])
    x = ttgl.full([128, 1], 1, ttgl.int32, layout=layout)
    # CHECK: tt.reshape %{{.*}} : tensor<128x1xi32, [[BLOCKED]]> -> tensor<128xi32, [[LINEAR]]>
    x.reshape([128])


@filecheck_test
@gluon.jit
def test_tensor_reshape():
    # CHECK: [[BLOCKED:#.*]] = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
    # CHECK: [[BLOCKED1:#.*]] = #ttg.blocked<{sizePerThread = [1, 1, 2], threadsPerWarp = [2, 4, 4], warpsPerCTA = [4, 1, 1], order = [2, 1, 0]}>
    layout: ttgl.constexpr = ttgl.BlockedLayout([2], [32], [4], [0])
    a = ttgl.full([256], 1, ttgl.int32, layout)
    # CHECK: tt.reshape {{.*}} : tensor<256xi32, [[BLOCKED]]> -> tensor<8x4x8xi32, [[BLOCKED1]]>
    v = a.reshape([8, 4, 8])
    expect_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 1, 2], [2, 4, 4], [4, 1, 1], [2, 1, 0])
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


@gluon.jit
def load_kernel(inp, xnumel):
    block_layout: ttgl.constexpr = ttgl.BlockedLayout([1], [32], [4], [0])
    xindex = ttgl.arange(0, 128, block_layout)
    mask = xindex < xnumel
    ttgl.load(inp + xindex, mask=mask, other=0.0)


@pytest.mark.parametrize("target", ALL_TARGETS)
def test_load(target):
    mod = run_parser(load_kernel, *make_args(MockTensor(ttgl.float32), xnumel=100), target=target)
    expecttest.assert_expected_inline(
        anonymize_ir(mod.str_nodebug()), """\
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "...", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @load_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: i32) attributes {noinline = false} {
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %1 = tt.splat %arg1 : i32 -> tensor<128xi32, #blocked>
    %2 = arith.cmpi slt, %0, %1 : tensor<128xi32, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %4 = tt.addptr %3, %0 : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128xf32, #blocked>
    %5 = tt.load %4, %2, %cst_0 : tensor<128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
""")


@gluon.jit
def async_copy_kernel(inp, xnumel, XBLOCK: ttgl.constexpr):
    smem = ttgl.allocate_shared_memory(inp.dtype.element_ty, [XBLOCK], ttgl.SwizzledSharedLayout(1, 1, 1, order=[0]))
    block_layout: ttgl.constexpr = ttgl.BlockedLayout([2], [32], [4], [0])
    xindex = ttgl.arange(0, XBLOCK, block_layout)
    mask = ttgl.max_constancy(xindex < xnumel, 2)

    async_copy.async_copy_global_to_shared(smem, inp + xindex)
    async_copy.async_copy_global_to_shared(smem, inp + xindex, mask, cache_modifier=".ca", eviction_policy="evict_last",
                                           volatile=True)

    mbar = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    async_copy.mbarrier_arrive(mbar)
    async_copy.mbarrier_arrive(mbar, increment_count=False)
    async_copy.commit_group()
    async_copy.wait_group(0)


@pytest.mark.parametrize("target", [AMPERE_TARGET, HOPPER_TARGET, BLACKWELL_TARGET])
def test_async_copy(target):
    mod = run_parser(
        async_copy_kernel,
        *make_args(MockTensor(ttgl.float16), xnumel=100, XBLOCK=128),
        target=target,
    )
    expecttest.assert_expected_inline(
        anonymize_ir(mod.str_nodebug()), """\
#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "...", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @async_copy_kernel(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: i32) attributes {noinline = false} {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<128xf16, #shared, #smem, mutable>
    %1 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %2 = tt.splat %arg1 : i32 -> tensor<128xi32, #blocked>
    %3 = arith.cmpi slt, %1, %2 {tt.constancy = dense<2> : tensor<1xi32>} : tensor<128xi32, #blocked>
    %4 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x!tt.ptr<f16>, #blocked>
    %5 = tt.addptr %4, %1 : tensor<128x!tt.ptr<f16>, #blocked>, tensor<128xi32, #blocked>
    %6 = ttg.async_copy_global_to_local %5, %0 : tensor<128x!tt.ptr<f16>, #blocked> -> <128xf16, #shared, #smem, mutable>
    %7 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x!tt.ptr<f16>, #blocked>
    %8 = tt.addptr %7, %1 : tensor<128x!tt.ptr<f16>, #blocked>, tensor<128xi32, #blocked>
    %9 = ttg.async_copy_global_to_local %8, %0 mask %3 cacheModifier = ca evictionPolicy = evict_last {isVolatile = true} : tensor<128x!tt.ptr<f16>, #blocked> -> <128xf16, #shared, #smem, mutable>
    %10 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.async_copy_mbarrier_arrive %10 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.async_copy_mbarrier_arrive %10 {noIncrement} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %11 = ttg.async_commit_group
    %12 = ttg.async_wait {num = 0 : i32}
    tt.return
  }
}
""")


@pytest.mark.parametrize("target", ALL_TARGETS)
def test_split_join_subtile(target):

    @gluon.jit
    def kernel():
        layout: ttgl.constexpr = ttgl.BlockedLayout([1, 128], [32, 1], [4, 1], [0, 1])
        x = ttgl.full([128, 128], 1, ttgl.int32, layout=layout)

        a, b = x.reshape([128, 2, 64]).permute([0, 2, 1]).split()
        y = ttgl.join(a, b).permute([0, 2, 1]).reshape([128, 128])
        _ = x + y

    mod = run_parser(kernel, target=target)
    expecttest.assert_expected_inline(
        anonymize_ir(mod.str_nodebug()), """\
#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 2, 64], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 2, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 64, 2], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 1, 2]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "...", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @kernel() attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant dense<1> : tensor<128x128xi32, #blocked>
    %0 = tt.reshape %cst : tensor<128x128xi32, #blocked> -> tensor<128x2x64xi32, #blocked1>
    %1 = tt.trans %0 {order = array<i32: 0, 2, 1>} : tensor<128x2x64xi32, #blocked1> -> tensor<128x64x2xi32, #blocked2>
    %outLHS, %outRHS = tt.split %1 : tensor<128x64x2xi32, #blocked2> -> tensor<128x64xi32, #ttg.slice<{dim = 2, parent = #blocked2}>>
    %2 = tt.join %outLHS, %outRHS : tensor<128x64xi32, #ttg.slice<{dim = 2, parent = #blocked2}>> -> tensor<128x64x2xi32, #blocked2>
    %3 = tt.trans %2 {order = array<i32: 0, 2, 1>} : tensor<128x64x2xi32, #blocked2> -> tensor<128x2x64xi32, #blocked1>
    %4 = tt.reshape %3 : tensor<128x2x64xi32, #blocked1> -> tensor<128x128xi32, #blocked>
    %5 = arith.addi %cst, %4 : tensor<128x128xi32, #blocked>
    tt.return
  }
}
""")


@filecheck_test
@gluon.jit
def test_auto_layout():
    # CHECK-DAG: [[BLOCKED:#.*]] = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
    # CHECK: [[X_1D:%.*]] = arith.constant dense<7> : tensor<16xi32, #gluon.auto_encoding>
    # CHECK: [[Y_1D:%.*]] = arith.constant dense<2> : tensor<8xi32, #gluon.auto_encoding>
    x = ttgl.full([16], 7, ttgl.int32, layout=ttgl.AutoLayout())[:, None]
    y = ttgl.full([8], 2, ttgl.int32, layout=ttgl.AutoLayout())[None, :]
    # CHECK: arith.addi {{.*}} : tensor<16x8xi32, #gluon.auto_encoding>
    z = x + y
    # CHECK: (tensor<16x8xi32, #gluon.auto_encoding>) -> tensor<16xi32, #gluon.auto_encoding
    ttgl.sum(z, axis=1)

    # CHECK: [[I:%.*]] = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #gluon.auto_encoding>
    i = ttgl.arange(0, 32)

    # CHECK: gluon.set_auto_layout [[I]] : tensor<32xi32, #gluon.auto_encoding> -> tensor<32xi32, [[BLOCKED]]
    ttgl.set_auto_layout(i, ttgl.BlockedLayout([1], [32], [4], [0]))


@filecheck_test
@gluon.jit
def test_auto_layout_broadcast():
    # CHECK: [[BLOCKED:#.*]] = #ttg.blocked
    # CHECK: [[X:%.*]] = arith.constant dense<1> : tensor<16x1xi32, #gluon.auto_encoding>
    # CHECK: [[Y:%.*]] = arith.constant dense<2> : tensor<1x16xi32, [[BLOCKED]]>
    x = ttgl.full([16, 1], 1, ttgl.int32, layout=ttgl.AutoLayout())
    y = ttgl.full([1, 16], 2, ttgl.int32, layout=ttgl.BlockedLayout([1, 1], [1, 32], [4, 1], [1, 0]))

    # CHECK: [[XCVT:%.*]] = gluon.set_auto_layout [[X]] : tensor<16x1xi32, #gluon.auto_encoding> -> tensor<16x1xi32, [[BLOCKED]]>
    # CHECK: [[XBCAST:%.*]] = tt.broadcast [[XCVT]]
    # CHECK: [[YBCAST:%.*]] = tt.broadcast [[Y]]
    # CHECK: arith.addi [[XBCAST]], [[YBCAST]] : tensor<16x16xi32, [[BLOCKED]]>
    _ = x + y

    # CHECK: [[XCVT2:%.*]] = gluon.set_auto_layout [[X]] : tensor<16x1xi32, #gluon.auto_encoding> -> tensor<16x1xi32, [[BLOCKED]]>
    # CHECK: [[YBCAST2:%.*]] = tt.broadcast [[Y]]
    # CHECK: [[XBCAST2:%.*]] = tt.broadcast [[XCVT2]]
    # CHECK: arith.muli [[YBCAST2]], [[XBCAST2]] : tensor<16x16xi32, [[BLOCKED]]>
    _ = y * x


@filecheck_test
@gluon.jit
def test_atomic_rmw():
    x0 = ttgl.full([1], 1, ttgl.int64, layout=ttgl.AutoLayout())
    ptr0 = x0.cast(ttgl.pointer_type(ttgl.int32), bitcast=True).item()
    # CHECK: [[c1:%.*]] = arith.constant 1 : i32
    # CHECK: {{.*}} = tt.atomic_rmw exch, acq_rel, gpu, %{{.*}}, [[c1]], %true : (!tt.ptr<i32>, i32, i1) -> i32
    ttgl.atomic_xchg(ptr0, 1)

    BLOCK: ttgl.constexpr = 128
    x = ttgl.full([BLOCK], 0, ttgl.int64, layout=ttgl.AutoLayout())
    ptr = x.cast(ttgl.pointer_type(ttgl.int32), bitcast=True)
    val = ttgl.full([BLOCK], 1, ttgl.int32, layout=ttgl.AutoLayout())
    mask = ttgl.full([BLOCK], True, ttgl.int1, layout=ttgl.AutoLayout())
    offset = ttgl.arange(0, BLOCK, layout=ttgl.AutoLayout())
    # CHECK: [[val:%.*]] = arith.constant dense<1> : tensor<128xi32, #gluon.auto_encoding>
    # CHECK: {{.*}} = tt.atomic_rmw min, acq_rel, gpu, %{{.*}}, [[val]], %{{.*}} : (tensor<128x!tt.ptr<i32>, #gluon.auto_encoding>, tensor<128xi32, #gluon.auto_encoding>, tensor<128xi1, #gluon.auto_encoding>) -> tensor<128xi32, #gluon.auto_encoding>
    # CHECK: {{.*}} = tt.atomic_rmw max, acq_rel, gpu, %{{.*}}, [[val]], %{{.*}} : (tensor<128x!tt.ptr<i32>, #gluon.auto_encoding>, tensor<128xi32, #gluon.auto_encoding>, tensor<128xi1, #gluon.auto_encoding>) -> tensor<128xi32, #gluon.auto_encoding>
    # CHECK: {{.*}} = tt.atomic_rmw add, acq_rel, gpu, %{{.*}}, [[val]], %{{.*}} : (tensor<128x!tt.ptr<i32>, #gluon.auto_encoding>, tensor<128xi32, #gluon.auto_encoding>, tensor<128xi1, #gluon.auto_encoding>) -> tensor<128xi32, #gluon.auto_encoding>
    # CHECK: {{.*}} = tt.atomic_rmw and, acq_rel, gpu, %{{.*}}, [[val]], %{{.*}} : (tensor<128x!tt.ptr<i32>, #gluon.auto_encoding>, tensor<128xi32, #gluon.auto_encoding>, tensor<128xi1, #gluon.auto_encoding>) -> tensor<128xi32, #gluon.auto_encoding>
    # CHECK: {{.*}} = tt.atomic_rmw or, acq_rel, gpu, %{{.*}}, [[val]], %{{.*}} : (tensor<128x!tt.ptr<i32>, #gluon.auto_encoding>, tensor<128xi32, #gluon.auto_encoding>, tensor<128xi1, #gluon.auto_encoding>) -> tensor<128xi32, #gluon.auto_encoding>
    # CHECK: {{.*}} = tt.atomic_rmw xor, acq_rel, gpu, %{{.*}}, [[val]], %{{.*}} : (tensor<128x!tt.ptr<i32>, #gluon.auto_encoding>, tensor<128xi32, #gluon.auto_encoding>, tensor<128xi1, #gluon.auto_encoding>) -> tensor<128xi32, #gluon.auto_encoding>
    # CHECK: {{.*}} = tt.atomic_rmw max, acq_rel, gpu, %{{.*}}, [[val]], %{{.*}} : (tensor<128x!tt.ptr<i32>, #gluon.auto_encoding>, tensor<128xi32, #gluon.auto_encoding>, tensor<128xi1, #gluon.auto_encoding>) -> tensor<128xi32, #gluon.auto_encoding>
    # CHECK: {{.*}} = tt.atomic_rmw add, relaxed, gpu, %{{.*}}, [[val]], %{{.*}} : (tensor<128x!tt.ptr<i32>, #gluon.auto_encoding>, tensor<128xi32, #gluon.auto_encoding>, tensor<128xi1, #gluon.auto_encoding>) -> tensor<128xi32, #gluon.auto_encoding>
    ttgl.atomic_min(offset + ptr, val)
    ttgl.atomic_max(offset + ptr, val)
    ttgl.atomic_add(offset + ptr, val)
    ttgl.atomic_and(offset + ptr, val)
    ttgl.atomic_or(offset + ptr, val)
    ttgl.atomic_xor(offset + ptr, val)
    ttgl.atomic_max(offset + ptr, val, mask=mask)
    ttgl.atomic_add(offset + ptr, val, mask=mask, sem="relaxed")


@filecheck_test
@gluon.jit
def test_atomic_cas():
    # CHECK: {{.*}} = arith.constant dense<1> : tensor<1xi64, #gluon.auto_encoding>
    x0 = ttgl.full([1], 1, ttgl.int64, layout=ttgl.AutoLayout())
    ptr0 = x0.cast(ttgl.pointer_type(ttgl.int32), bitcast=True).item()
    # CHECK: [[c0:%.*]] = arith.constant 0 : i32
    # CHECK: [[c1:%.*]] = arith.constant 1 : i32
    # CHECK: {{.*}} = tt.atomic_cas acq_rel, gpu, %{{.*}}, [[c0]], [[c1]] : (!tt.ptr<i32>, i32, i32) -> i32
    ttgl.atomic_cas(ptr0, 0, 1)

    BLOCK: ttgl.constexpr = 128
    x = ttgl.full([BLOCK], 0, ttgl.int64, layout=ttgl.AutoLayout())
    ptr = x.cast(ttgl.pointer_type(ttgl.int32), bitcast=True)
    # CHECK: {{.*}} = arith.constant dense<0> : tensor<128xi64, #gluon.auto_encoding>
    offset = ttgl.arange(0, BLOCK, layout=ttgl.AutoLayout())
    old = ttgl.full([BLOCK], 0, ttgl.int32, layout=ttgl.AutoLayout())
    new = ttgl.full([BLOCK], 1, ttgl.int32, layout=ttgl.AutoLayout())
    # CHECK: [[old:%.*]] = arith.constant dense<0> : tensor<128xi32, #gluon.auto_encoding>
    # CHECK: [[new:%.*]] = arith.constant dense<1> : tensor<128xi32, #gluon.auto_encoding>
    # CHECK: {{.*}} = tt.atomic_cas relaxed, gpu, %{{.*}}, [[old]], [[new]] : (tensor<128x!tt.ptr<i32>, #gluon.auto_encoding>, tensor<128xi32, #gluon.auto_encoding>, tensor<128xi32, #gluon.auto_encoding>) -> tensor<128xi32, #gluon.auto_encoding>
    # CHECK: {{.*}} = tt.atomic_cas acq_rel, gpu, %{{.*}}, [[old]], [[new]] : (tensor<128x!tt.ptr<i32>, #gluon.auto_encoding>, tensor<128xi32, #gluon.auto_encoding>, tensor<128xi32, #gluon.auto_encoding>) -> tensor<128xi32, #gluon.auto_encoding>
    ttgl.atomic_cas(offset + ptr, old, new, sem="relaxed")
    ttgl.atomic_cas(offset + ptr, old, new)


@gluon.jit
def amd_mfma_layout_kernel():
    ttgl.full([128, 32], 0, ttgl.float32, layout=amd_layouts.AMDMFMALayout(version=3, instr_shape=[32, 32],
                                                                           transposed=True, warps_per_cta=[4, 1]))

    ttgl.full([128, 32], 0, ttgl.float32,
              layout=amd_layouts.AMDMFMALayout(version=3, instr_shape=[32, 32], tiles_per_warp=[4, 1], transposed=True,
                                               warps_per_cta=[4, 1]))

    ttgl.full([128, 32], 0, ttgl.float32,
              layout=amd_layouts.AMDMFMALayout(version=3, instr_shape=[32, 32], transposed=True, warps_per_cta=[4, 1],
                                               ctas_per_cga=[1, 1], tiles_per_warp=[1, 1], cta_split_num=[1, 1],
                                               cta_order=[1, 0]))

    ttgl.full([128, 32], 0, ttgl.float64,
              layout=amd_layouts.AMDMFMALayout(version=3, instr_shape=[16, 16], transposed=True, warps_per_cta=[4, 1],
                                               elem_type=ttgl.float64, tiles_per_warp=[1, 1], ctas_per_cga=[1, 1],
                                               cta_split_num=[1, 1], cta_order=[1, 0]))

    ttgl.full([128, 32], 0, ttgl.int32,
              layout=amd_layouts.AMDMFMALayout(version=3, instr_shape=[16, 16], transposed=True, warps_per_cta=[4, 1],
                                               elem_type=ttgl.int32, tiles_per_warp=[1, 1], ctas_per_cga=[1, 1],
                                               cta_split_num=[1, 1]))


@pytest.mark.parametrize("target", [HIP_TARGET_CDNA3, HIP_TARGET_CDNA4])
def test_amd_mfma_layout(target):

    module = run_parser(amd_mfma_layout_kernel, target=target)
    expecttest.assert_expected_inline(
        anonymize_ir(module.str_nodebug()), """\
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>
#mma1 = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], tilesPerWarp = [4, 1], instrShape = [32, 32], isTransposed = true}>
#mma2 = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true, elementType = f64}>
#mma3 = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true, elementType = i32}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "...", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @amd_mfma_layout_kernel() attributes {noinline = false} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x32xf32, #mma>
    %cst_1 = arith.constant 0.000000e+00 : f32
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x32xf32, #mma1>
    %cst_3 = arith.constant 0.000000e+00 : f32
    %cst_4 = arith.constant dense<0.000000e+00> : tensor<128x32xf32, #mma>
    %cst_5 = arith.constant 0.000000e+00 : f64
    %cst_6 = arith.constant dense<0.000000e+00> : tensor<128x32xf64, #mma2>
    %c0_i32 = arith.constant 0 : i32
    %cst_7 = arith.constant dense<0> : tensor<128x32xi32, #mma3>
    tt.return
  }
}
""")


@gluon.jit
def add_int(a, b):
    return a + b


@gluon.jit
def infer_layout_for_amd_mfma_kernel():
    layout: ttgl.constexpr = amd_layouts.AMDMFMALayout(version=3, instr_shape=[32, 32], transposed=True,
                                                       warps_per_cta=[4,
                                                                      1], elem_type=ttgl.int32, tiles_per_warp=[1, 1],
                                                       ctas_per_cga=[1, 1], cta_split_num=[1, 1], cta_order=[1, 0])
    a = ttgl.full([128, 32], 1, ttgl.int32, layout)
    b = ttgl.reduce(a, 1, add_int)
    ttgl.static_assert(b.type.layout == ttgl.SliceLayout(1, layout))


@pytest.mark.parametrize("target", [HIP_TARGET_CDNA3, HIP_TARGET_CDNA4])
def test_infer_layout_for_amd_mfma(target):
    module = run_parser(infer_layout_for_amd_mfma_kernel, target=target)

    expecttest.assert_expected_inline(
        anonymize_ir(module.str_nodebug()), """\
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true, elementType = i32}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "...", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @infer_layout_for_amd_mfma_kernel() attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant dense<1> : tensor<128x32xi32, #mma>
    %0 = "tt.reduce"(%cst) <{axis = 1 : i32}> ({
    ^bb0(%arg0: i32, %arg1: i32):
      %1 = tt.call @test_frontend.add_int__i32_i32__(%arg0, %arg1) : (i32, i32) -> i32
      tt.reduce.return %1 : i32
    }) : (tensor<128x32xi32, #mma>) -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #mma}>>
    tt.return
  }
  tt.func private @test_frontend.add_int__i32_i32__(%arg0: i32, %arg1: i32) -> i32 attributes {noinline = false} {
    %0 = arith.addi %arg0, %arg1 : i32
    tt.return %0 : i32
  ^bb1:  // no predecessors
    %1 = ub.poison : i32
    tt.return %1 : i32
  }
}
""")


@pytest.mark.parametrize("target", [HIP_TARGET_CDNA4])
def test_amd_load_shared_relaxed(target):

    @gluon.jit
    def kernel():
        blocked: ttgl.constexpr = ttgl.BlockedLayout([1, 8], [32, 2], [4, 1], [1, 0])
        shared: ttgl.constexpr = ttgl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])

        smem = ttgl.allocate_shared_memory(ttgl.float16, [128, 16], shared)
        cdna4_async_copy.load_shared_relaxed(smem, blocked)

    mod = run_parser(kernel, target=target)
    expecttest.assert_expected_inline(
        anonymize_ir(mod.str_nodebug()), """\
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [32, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "...", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @kernel() attributes {noinline = false} {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<128x16xf16, #shared, #smem, mutable>
    %1 = ttg.local_load %0 {ttg.amdgpu.syncedViaAsyncWait = true} : !ttg.memdesc<128x16xf16, #shared, #smem, mutable> -> tensor<128x16xf16, #blocked>
    tt.return
  }
}
""")


@pytest.mark.parametrize("target", [HIP_TARGET_CDNA4])
def test_amd_load_shared_relaxed_in_loop(target):

    @gluon.jit
    def kernel():
        blocked: ttgl.constexpr = ttgl.BlockedLayout([1, 8], [32, 2], [4, 1], [1, 0])
        shared: ttgl.constexpr = ttgl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])

        smem = ttgl.allocate_shared_memory(ttgl.float16, [128, 16], shared)
        for i in range(10):
            cdna4_async_copy.load_shared_relaxed(smem, blocked)

    mod = run_parser(kernel, target=target)
    expecttest.assert_expected_inline(
        anonymize_ir(mod.str_nodebug()), """\
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [32, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "...", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @kernel() attributes {noinline = false} {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<128x16xf16, #shared, #smem, mutable>
    %c0_i32 = arith.constant 0 : i32
    %c10_i32 = arith.constant 10 : i32
    %c1_i32 = arith.constant 1 : i32
    %1 = arith.bitcast %c0_i32 : i32 to i32
    %2 = arith.bitcast %c10_i32 : i32 to i32
    %3 = arith.bitcast %c1_i32 : i32 to i32
    %4 = ub.poison : i32
    scf.for %arg0 = %1 to %2 step %3  : i32 {
      %5 = ttg.local_load %0 {ttg.amdgpu.syncedViaAsyncWait = true} : !ttg.memdesc<128x16xf16, #shared, #smem, mutable> -> tensor<128x16xf16, #blocked>
    }
    tt.return
  }
}
""")


@pytest.mark.parametrize("target", [HIP_TARGET_CDNA4])
def test_amd_global_load_to_shared(target):

    @gluon.jit
    def kernel(ptr):
        blocked: ttgl.constexpr = ttgl.BlockedLayout([1, 8], [32, 2], [4, 1], [1, 0])
        shared: ttgl.constexpr = ttgl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])

        smem = ttgl.allocate_shared_memory(ptr.dtype.element_ty, [128, 16], shared)
        offsets = ttgl.arange(0, 128, layout=ttgl.SliceLayout(1, blocked))[:, None] * 16 + \
                  ttgl.arange(0, 16, layout=ttgl.SliceLayout(0, blocked))[None, :]

        cdna4_async_copy.global_load_to_shared(smem, ptr + offsets)
        cdna4_async_copy.async_wait(0)

    ptr = MockTensor(ttgl.float16)
    mod = run_parser(kernel, *make_args(ptr), target=target)
    expecttest.assert_expected_inline(
        anonymize_ir(mod.str_nodebug()), """\
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [32, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "...", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @kernel(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<128x16xf16, #shared, #smem, mutable>
    %1 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %2 = tt.expand_dims %1 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
    %c16_i32 = arith.constant 16 : i32
    %c16_i32_0 = arith.constant 16 : i32
    %cst = arith.constant dense<16> : tensor<128x1xi32, #blocked>
    %3 = arith.muli %2, %cst : tensor<128x1xi32, #blocked>
    %4 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %5 = tt.expand_dims %4 {axis = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x16xi32, #blocked>
    %6 = tt.broadcast %3 : tensor<128x1xi32, #blocked> -> tensor<128x16xi32, #blocked>
    %7 = tt.broadcast %5 : tensor<1x16xi32, #blocked> -> tensor<128x16xi32, #blocked>
    %8 = arith.addi %6, %7 : tensor<128x16xi32, #blocked>
    %9 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x16x!tt.ptr<f16>, #blocked>
    %10 = tt.addptr %9, %8 : tensor<128x16x!tt.ptr<f16>, #blocked>, tensor<128x16xi32, #blocked>
    %11 = ttg.async_copy_global_to_local %10, %0 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    %12 = ttg.async_wait {num = 0 : i32}
    tt.return
  }
}
""")


@pytest.mark.parametrize("target", [HIP_TARGET_CDNA4])
def test_amd_global_load_to_shared_with_broadcast(target):

    @gluon.jit
    def kernel(ptr):
        blocked: ttgl.constexpr = ttgl.BlockedLayout([1, 8], [32, 2], [4, 1], [1, 0])
        shared: ttgl.constexpr = ttgl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])

        smem = ttgl.allocate_shared_memory(ptr.dtype.element_ty, [128, 16], shared)
        y_offset = ttgl.arange(0, 128, layout=ttgl.SliceLayout(1, blocked))
        x_offset = ttgl.arange(0, 16, layout=ttgl.SliceLayout(0, blocked))
        offsets = y_offset[:, None] * 16 + x_offset[None, :]

        mask = (y_offset < 64)[:, None]
        other = tl.cast(0.0, ptr.dtype.element_ty)

        cdna4_async_copy.global_load_to_shared(smem, ptr + offsets, mask, other)
        cdna4_async_copy.async_wait(0)

    ptr = MockTensor(ttgl.float16)
    mod = run_parser(kernel, *make_args(ptr), target=target)
    expecttest.assert_expected_inline(
        anonymize_ir(mod.str_nodebug()), """\
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [32, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "...", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @kernel(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<128x16xf16, #shared, #smem, mutable>
    %1 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %2 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %3 = tt.expand_dims %1 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
    %c16_i32 = arith.constant 16 : i32
    %c16_i32_0 = arith.constant 16 : i32
    %cst = arith.constant dense<16> : tensor<128x1xi32, #blocked>
    %4 = arith.muli %3, %cst : tensor<128x1xi32, #blocked>
    %5 = tt.expand_dims %2 {axis = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x16xi32, #blocked>
    %6 = tt.broadcast %4 : tensor<128x1xi32, #blocked> -> tensor<128x16xi32, #blocked>
    %7 = tt.broadcast %5 : tensor<1x16xi32, #blocked> -> tensor<128x16xi32, #blocked>
    %8 = arith.addi %6, %7 : tensor<128x16xi32, #blocked>
    %c64_i32 = arith.constant 64 : i32
    %cst_1 = arith.constant dense<64> : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %9 = arith.cmpi slt, %1, %cst_1 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %10 = tt.expand_dims %9 {axis = 1 : i32} : tensor<128xi1, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi1, #blocked>
    %cst_2 = arith.constant 0.000000e+00 : f32
    %11 = arith.truncf %cst_2 : f32 to f16
    %12 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x16x!tt.ptr<f16>, #blocked>
    %13 = tt.addptr %12, %8 : tensor<128x16x!tt.ptr<f16>, #blocked>, tensor<128x16xi32, #blocked>
    %14 = tt.broadcast %10 : tensor<128x1xi1, #blocked> -> tensor<128x16xi1, #blocked>
    %15 = tt.splat %11 : f16 -> tensor<128x16xf16, #blocked>
    %16 = ttg.async_copy_global_to_local %13, %0 mask %14 other %15 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    %17 = ttg.async_wait {num = 0 : i32}
    tt.return
  }
}
""")


@pytest.mark.parametrize("target", [HIP_TARGET_CDNA4])
def test_buffer_load_to_shared(target):

    @gluon.jit
    def kernel(ptr):
        blocked: ttgl.constexpr = ttgl.BlockedLayout([1], [64], [4], [0])
        shared: ttgl.constexpr = ttgl.SwizzledSharedLayout(1, 1, 1, order=[0])

        dest = ttgl.allocate_shared_memory(ptr.dtype.element_ty, [256], shared)
        offsets = ttgl.arange(0, 256, layout=blocked)

        cdna4_async_copy.buffer_load_to_shared(dest, ptr, offsets)

    ptr = MockTensor(ttgl.float32)
    mod = run_parser(kernel, *make_args(ptr), target=target)
    expecttest.assert_expected_inline(
        anonymize_ir(mod.str_nodebug()), """\
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "...", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<256xf32, #shared, #smem, mutable>
    %1 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
    %2 = amdgpu.buffer_load_to_local %arg0[%1] into %0 : <f32>[tensor<256xi32, #blocked>]  -> <256xf32, #shared, #smem, mutable>
    tt.return
  }
}
""")


@pytest.mark.parametrize("target", [HIP_TARGET_CDNA4])
def test_buffer_load_to_shared_with_broadcast(target):

    @gluon.jit
    def kernel(ptr):
        blocked1: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [1, 64], [4, 1], [1, 0])
        shared: ttgl.constexpr = ttgl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])

        dest = ttgl.allocate_shared_memory(ptr.dtype.element_ty, [4, 64], shared)

        y_index = ttgl.arange(0, 4, layout=ttgl.SliceLayout(1, blocked1))
        x_index = ttgl.arange(0, 64, layout=ttgl.SliceLayout(0, blocked1))
        offsets = y_index[:, None] * 64 + x_index[None, :]

        mask = (y_index < 2)[:, None]
        other = 0.0

        cdna4_async_copy.buffer_load_to_shared(dest, ptr, offsets, mask, other)

    ptr = MockTensor(ttgl.float32)
    mod = run_parser(kernel, *make_args(ptr), target=target)
    expecttest.assert_expected_inline(
        anonymize_ir(mod.str_nodebug()), """\
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "...", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<4x64xf32, #shared, #smem, mutable>
    %1 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %3 = tt.expand_dims %1 {axis = 1 : i32} : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<4x1xi32, #blocked>
    %c64_i32 = arith.constant 64 : i32
    %c64_i32_0 = arith.constant 64 : i32
    %cst = arith.constant dense<64> : tensor<4x1xi32, #blocked>
    %4 = arith.muli %3, %cst : tensor<4x1xi32, #blocked>
    %5 = tt.expand_dims %2 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %6 = tt.broadcast %4 : tensor<4x1xi32, #blocked> -> tensor<4x64xi32, #blocked>
    %7 = tt.broadcast %5 : tensor<1x64xi32, #blocked> -> tensor<4x64xi32, #blocked>
    %8 = arith.addi %6, %7 : tensor<4x64xi32, #blocked>
    %c2_i32 = arith.constant 2 : i32
    %cst_1 = arith.constant dense<2> : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %9 = arith.cmpi slt, %1, %cst_1 : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %10 = tt.expand_dims %9 {axis = 1 : i32} : tensor<4xi1, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<4x1xi1, #blocked>
    %cst_2 = arith.constant 0.000000e+00 : f32
    %11 = tt.broadcast %10 : tensor<4x1xi1, #blocked> -> tensor<4x64xi1, #blocked>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<4x64xf32, #blocked>
    %12 = amdgpu.buffer_load_to_local %arg0[%8] mask = %11 other = %cst_3 into %0 : <f32>[tensor<4x64xi32, #blocked>] tensor<4x64xf32, #blocked> -> <4x64xf32, #shared, #smem, mutable>
    tt.return
  }
}
""")


@pytest.mark.parametrize("target", [HIP_TARGET_CDNA4])
def test_buffer_load_to_shared_mask_other(target):

    @gluon.jit
    def kernel(ptr):
        blocked: ttgl.constexpr = ttgl.BlockedLayout([1], [64], [4], [0])
        shared: ttgl.constexpr = ttgl.SwizzledSharedLayout(1, 1, 1, order=[0])

        dest = ttgl.allocate_shared_memory(ptr.dtype.element_ty, [256], shared)
        offsets = ttgl.arange(0, 256, layout=blocked)

        mask = ttgl.full([256], 1, ttgl.int1, layout=blocked)
        other = ttgl.full([256], 0, ptr.dtype.element_ty, layout=blocked)
        cdna4_async_copy.buffer_load_to_shared(dest, ptr, offsets, mask, other)

    ptr = MockTensor(ttgl.float32)
    mod = run_parser(kernel, *make_args(ptr), target=target)
    expecttest.assert_expected_inline(
        anonymize_ir(mod.str_nodebug()), """\
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "...", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<256xf32, #shared, #smem, mutable>
    %1 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
    %true = arith.constant true
    %cst = arith.constant dense<true> : tensor<256xi1, #blocked>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<256xf32, #blocked>
    %2 = amdgpu.buffer_load_to_local %arg0[%1] mask = %cst other = %cst_1 into %0 : <f32>[tensor<256xi32, #blocked>] tensor<256xf32, #blocked> -> <256xf32, #shared, #smem, mutable>
    tt.return
  }
}
""")


@pytest.mark.parametrize("target", [HIP_TARGET_CDNA4])
def test_buffer_load_to_shared_cache_mods(target):

    @gluon.jit
    def kernel(ptr):
        blocked: ttgl.constexpr = ttgl.BlockedLayout([1], [64], [4], [0])
        shared: ttgl.constexpr = ttgl.SwizzledSharedLayout(1, 1, 1, order=[0])

        dest = ttgl.allocate_shared_memory(ptr.dtype.element_ty, [256], shared)
        offsets = ttgl.arange(0, 256, layout=blocked)

        cdna4_async_copy.buffer_load_to_shared(dest, ptr, offsets, cache_modifier=".ca")
        cdna4_async_copy.buffer_load_to_shared(dest, ptr, offsets, cache_modifier=".cg")
        cdna4_async_copy.buffer_load_to_shared(dest, ptr, offsets, cache_modifier=".cv")

    ptr = MockTensor(ttgl.float32)
    mod = run_parser(kernel, *make_args(ptr), target=target)
    expecttest.assert_expected_inline(
        anonymize_ir(mod.str_nodebug()), """\
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "...", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<256xf32, #shared, #smem, mutable>
    %1 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
    %2 = amdgpu.buffer_load_to_local %arg0[%1] cacheModifier = ca into %0 : <f32>[tensor<256xi32, #blocked>]  -> <256xf32, #shared, #smem, mutable>
    %3 = amdgpu.buffer_load_to_local %arg0[%1] cacheModifier = cg into %0 : <f32>[tensor<256xi32, #blocked>]  -> <256xf32, #shared, #smem, mutable>
    %4 = amdgpu.buffer_load_to_local %arg0[%1] cacheModifier = cv into %0 : <f32>[tensor<256xi32, #blocked>]  -> <256xf32, #shared, #smem, mutable>
    tt.return
  }
}
""")


@gluon.jit
def buffer_load_store_kernel(x, y):
    layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1, 1], threads_per_warp=[1, 64], warps_per_cta=[4, 1],
                                                order=[1, 0])

    offsets = ttgl.arange(0, 64 * 64).reshape(64, 64)
    offsets = ttgl.convert_layout(offsets, layout=layout)
    mask = ttgl.full((64, 64), 1, tl.int1, layout=layout)
    other = ttgl.full((64, 64), 1.0, tl.float32, layout=layout)
    a = ttgl.amd.cdna3.buffer_load(ptr=x, offsets=offsets, mask=mask, other=other, cache='.ca')
    ttgl.amd.cdna3.buffer_store(stored_value=a, ptr=y, offsets=offsets, mask=mask, cache='.cs')

    a = ttgl.amd.cdna4.buffer_load(ptr=x, offsets=offsets, mask=mask, other=other, cache='.ca')
    ttgl.amd.cdna4.buffer_store(stored_value=a, ptr=y, offsets=offsets, mask=mask, cache='.cs')


@pytest.mark.parametrize("target", [HIP_TARGET_CDNA3, HIP_TARGET_CDNA4])
def test_buffer_load_store(target):
    x = MockTensor(ttgl.float32)
    y = MockTensor(ttgl.float32)
    module = run_parser(buffer_load_store_kernel, *make_args(x, y), target=target)

    expecttest.assert_expected_inline(
        anonymize_ir(module.str_nodebug()), """\
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "...", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @buffer_load_store_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %0 = tt.make_range {end = 4096 : i32, start = 0 : i32} : tensor<4096xi32, #gluon.auto_encoding>
    %1 = tt.reshape %0 : tensor<4096xi32, #gluon.auto_encoding> -> tensor<64x64xi32, #gluon.auto_encoding>
    %2 = ttg.convert_layout %1 : tensor<64x64xi32, #gluon.auto_encoding> -> tensor<64x64xi32, #blocked>
    %true = arith.constant true
    %cst = arith.constant dense<true> : tensor<64x64xi1, #blocked>
    %cst_0 = arith.constant 1.000000e+00 : f32
    %cst_1 = arith.constant dense<1.000000e+00> : tensor<64x64xf32, #blocked>
    %3 = amdgpu.buffer_load %arg0[%2], %cst, %cst_1 cacheModifier = ca : tensor<64x64xf32, #blocked>
    amdgpu.buffer_store %3, %arg1[%2], %cst cacheModifier = cs : tensor<64x64xf32, #blocked>
    %4 = amdgpu.buffer_load %arg0[%2], %cst, %cst_1 cacheModifier = ca : tensor<64x64xf32, #blocked>
    amdgpu.buffer_store %4, %arg1[%2], %cst cacheModifier = cs : tensor<64x64xf32, #blocked>
    tt.return
  }
}
""")


@gluon.jit
def buffer_load_store_with_broadcast_kernel(x, y):
    layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1, 1], threads_per_warp=[1, 64], warps_per_cta=[4, 1],
                                                order=[1, 0])

    offsets = ttgl.arange(0, 64 * 64).reshape(64, 64)
    offsets = ttgl.convert_layout(offsets, layout=layout)
    other = ttgl.full((64, 64), 1.0, tl.float32, layout=layout)

    mask = ttgl.full((64, 1), 1, tl.int1, layout=layout)
    a = ttgl.amd.cdna3.buffer_load(ptr=x, offsets=offsets, mask=mask, other=other, cache='.ca')
    ttgl.amd.cdna3.buffer_store(stored_value=a, ptr=y, offsets=offsets, mask=mask, cache='.cs')

    mask = ttgl.full((1, 64), 1, tl.int1, layout=layout)
    a = ttgl.amd.cdna3.buffer_load(ptr=x, offsets=offsets, mask=mask, other=other, cache='.ca')
    ttgl.amd.cdna3.buffer_store(stored_value=a, ptr=y, offsets=offsets, mask=mask, cache='.cs')

    other = 1.0
    a = ttgl.amd.cdna3.buffer_load(ptr=x, offsets=offsets, mask=mask, other=other, cache='.ca')
    ttgl.amd.cdna3.buffer_store(stored_value=a, ptr=y, offsets=offsets, mask=mask, cache='.cs')


@pytest.mark.parametrize("target", [HIP_TARGET_CDNA3, HIP_TARGET_CDNA4])
def test_buffer_load_store_with_broadcast(target):
    x = MockTensor(ttgl.float32)
    y = MockTensor(ttgl.float32)
    module = run_parser(buffer_load_store_with_broadcast_kernel, *make_args(x, y), target=target)

    expecttest.assert_expected_inline(
        anonymize_ir(module.str_nodebug()), """\
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "...", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @buffer_load_store_with_broadcast_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %0 = tt.make_range {end = 4096 : i32, start = 0 : i32} : tensor<4096xi32, #gluon.auto_encoding>
    %1 = tt.reshape %0 : tensor<4096xi32, #gluon.auto_encoding> -> tensor<64x64xi32, #gluon.auto_encoding>
    %2 = ttg.convert_layout %1 : tensor<64x64xi32, #gluon.auto_encoding> -> tensor<64x64xi32, #blocked>
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant dense<1.000000e+00> : tensor<64x64xf32, #blocked>
    %true = arith.constant true
    %cst_1 = arith.constant dense<true> : tensor<64x1xi1, #blocked>
    %3 = tt.broadcast %cst_1 : tensor<64x1xi1, #blocked> -> tensor<64x64xi1, #blocked>
    %4 = amdgpu.buffer_load %arg0[%2], %3, %cst_0 cacheModifier = ca : tensor<64x64xf32, #blocked>
    %5 = tt.broadcast %cst_1 : tensor<64x1xi1, #blocked> -> tensor<64x64xi1, #blocked>
    amdgpu.buffer_store %4, %arg1[%2], %5 cacheModifier = cs : tensor<64x64xf32, #blocked>
    %true_2 = arith.constant true
    %cst_3 = arith.constant dense<true> : tensor<1x64xi1, #blocked>
    %6 = tt.broadcast %cst_3 : tensor<1x64xi1, #blocked> -> tensor<64x64xi1, #blocked>
    %7 = amdgpu.buffer_load %arg0[%2], %6, %cst_0 cacheModifier = ca : tensor<64x64xf32, #blocked>
    %8 = tt.broadcast %cst_3 : tensor<1x64xi1, #blocked> -> tensor<64x64xi1, #blocked>
    amdgpu.buffer_store %7, %arg1[%2], %8 cacheModifier = cs : tensor<64x64xf32, #blocked>
    %cst_4 = arith.constant 1.000000e+00 : f32
    %9 = tt.broadcast %cst_3 : tensor<1x64xi1, #blocked> -> tensor<64x64xi1, #blocked>
    %cst_5 = arith.constant dense<1.000000e+00> : tensor<64x64xf32, #blocked>
    %10 = amdgpu.buffer_load %arg0[%2], %9, %cst_5 cacheModifier = ca : tensor<64x64xf32, #blocked>
    %11 = tt.broadcast %cst_3 : tensor<1x64xi1, #blocked> -> tensor<64x64xi1, #blocked>
    amdgpu.buffer_store %10, %arg1[%2], %11 cacheModifier = cs : tensor<64x64xf32, #blocked>
    tt.return
  }
}
""")


@pytest.mark.parametrize("target", [HIP_TARGET_CDNA3, HIP_TARGET_CDNA4])
def test_amd_mfma(target):

    @gluon.jit
    def kernel():
        mfma_layout: ttgl.constexpr = ttgl.amd.AMDMFMALayout(version=3, instr_shape=[32, 32], transposed=True,
                                                             warps_per_cta=[4, 1])

        a = ttgl.full([64, 32], 1.0, ttgl.float32, layout=ttgl.DotOperandLayout(operand_index=0, parent=mfma_layout,
                                                                                k_width=8))
        b = ttgl.full([32, 64], 2.0, ttgl.float32, layout=ttgl.DotOperandLayout(operand_index=1, parent=mfma_layout,
                                                                                k_width=8))

        acc = ttgl.zeros([64, 64], ttgl.float32, mfma_layout)
        acc = ttgl.amd.cdna3.mfma(a, b, acc)
        ttgl.static_assert(isinstance(acc, ttgl.tensor))
        ttgl.static_assert(acc.type.layout == mfma_layout)

    module = run_parser(kernel, target=target)

    expecttest.assert_expected_inline(
        anonymize_ir(module.str_nodebug()), """\
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "...", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @kernel() attributes {noinline = false} {
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant dense<1.000000e+00> : tensor<64x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    %cst_1 = arith.constant 2.000000e+00 : f32
    %cst_2 = arith.constant dense<2.000000e+00> : tensor<32x64xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    %0 = tt.call @"triton.experimental.gluon.language._standard.zeros____(0, 0)cconstexpr_64__(0, 1)cconstexpr_64__(1,)cconstexpr_fp32__(2,)cconstexpr_AMDMFMALayout(version=3, instr_shape=(32 ,32), transposed=True, warps_per_cta=(4 ,1), elem_type=triton_d_language_d_float32, tiles_per_warp=_1, 1_, ctas_per_cga=_1, 1_, cta_split_num=_1, 1_, cta_order=_1, 0_)_"() : () -> tensor<64x64xf32, #mma>
    %cst_3 = arith.constant 0.000000e+00 : f32
    %1 = tt.dot %cst_0, %cst_2, %0 : tensor<64x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<32x64xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<64x64xf32, #mma>
    tt.return
  }
  tt.func private @"triton.experimental.gluon.language._standard.zeros____(0, 0)cconstexpr_64__(0, 1)cconstexpr_64__(1,)cconstexpr_fp32__(2,)cconstexpr_AMDMFMALayout(version=3, instr_shape=(32 ,32), transposed=True, warps_per_cta=(4 ,1), elem_type=triton_d_language_d_float32, tiles_per_warp=_1, 1_, ctas_per_cga=_1, 1_, cta_split_num=_1, 1_, cta_order=_1, 0_)_"() -> tensor<64x64xf32, #mma> attributes {noinline = false} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    tt.return %cst_0 : tensor<64x64xf32, #mma>
  ^bb1:  // no predecessors
    %0 = ub.poison : tensor<64x64xf32, #mma>
    tt.return %0 : tensor<64x64xf32, #mma>
  }
}
""")


@pytest.mark.parametrize("target", [HIP_TARGET_CDNA4])
def test_amd_mfma_scaled(target):

    @gluon.jit
    def kernel():
        mfma_layout: ttgl.constexpr = ttgl.amd.AMDMFMALayout(version=4, warps_per_cta=[1, 1], tiles_per_warp=[1, 1],
                                                             instr_shape=[16, 16], transposed=True)
        scale_layout: ttgl.constexpr = ttgl.DistributedLinearLayout([],
                                                                    [[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2]],
                                                                    [], [], [16, 4])

        a = ttgl.full([16, 64], 0x11, ttgl.uint8, ttgl.DotOperandLayout(operand_index=0, parent=mfma_layout,
                                                                        k_width=16))
        b = ttgl.full([64, 16], 0x22, ttgl.uint8, ttgl.DotOperandLayout(operand_index=1, parent=mfma_layout,
                                                                        k_width=16))
        a_scale = ttgl.full([16, 4], 0x02, ttgl.uint8, scale_layout)
        b_scale = ttgl.full([16, 4], 0x01, ttgl.uint8, scale_layout)
        acc = ttgl.full([16, 16], 0, ttgl.float32, mfma_layout)
        ttgl.amd.cdna4.mfma_scaled(a, a_scale, 'e2m1', b, b_scale, 'e2m1', acc)

    module = run_parser(kernel, *make_args(num_warps=1), target=target)
    expecttest.assert_expected_inline(
        anonymize_ir(module.str_nodebug()), """\
#linear = #ttg.linear<{register = [], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2]], warp = [], block = []}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "...", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @kernel() attributes {noinline = false} {
    %c17_i8 = arith.constant 17 : i8
    %cst = arith.constant dense<17> : tensor<16x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>
    %c34_i8 = arith.constant 34 : i8
    %cst_0 = arith.constant dense<34> : tensor<64x16xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>
    %c2_i8 = arith.constant 2 : i8
    %cst_1 = arith.constant dense<2> : tensor<16x4xi8, #linear>
    %c1_i8 = arith.constant 1 : i8
    %cst_2 = arith.constant dense<1> : tensor<16x4xi8, #linear>
    %cst_3 = arith.constant 0.000000e+00 : f32
    %cst_4 = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    %cst_5 = arith.constant 0.000000e+00 : f32
    %0 = tt.dot_scaled %cst scale %cst_1, %cst_0 scale %cst_2, %cst_4 lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<16x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>, tensor<16x4xi8, #linear> * tensor<64x16xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>, tensor<16x4xi8, #linear> -> tensor<16x16xf32, #mma>
    tt.return
  }
}
""")


@gluon.jit
def padded_shared_layout_kernel():
    padded_shared_layout: ttgl.constexpr = ttgl.PaddedSharedLayout(interval_padding_pairs=[[2, 1], [4, 2], [8, 4]],
                                                                   order=[1, 0], ctas_per_cga=[1, 1],
                                                                   cta_split_num=[1, 1], cta_order=[1, 0])

    ttgl.allocate_shared_memory(ttgl.int32, [64, 64], padded_shared_layout)


@pytest.mark.parametrize("target", [HIP_TARGET_CDNA3, HIP_TARGET_CDNA4])
def test_padded_shared_layout(target):
    # This test is used to test the construction of PaddedSharedEncodingAttr in the gluon.
    module = run_parser(padded_shared_layout_kernel, target=target)
    expecttest.assert_expected_inline(
        anonymize_ir(module.str_nodebug()), """\
#shared = #ttg.padded_shared<[2:+1, 4:+2, 8:+4] {order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "...", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @padded_shared_layout_kernel() attributes {noinline = false} {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<64x64xi32, #shared, #smem, mutable>
    tt.return
  }
}
""")


@gluon.jit
def infer_layout_for_padded_shared_kernel():
    layout: ttgl.constexpr = ttgl.PaddedSharedLayout(interval_padding_pairs=[[2, 1], [4, 2], [8, 4]], order=[2, 0, 1])
    smem = ttgl.allocate_shared_memory(ttgl.int32, [32, 4, 32], layout)

    reshaped = smem.permute((1, 0, 2))
    """
    permute is [1 0 2], which means
    old 1 to new 0
    old 0 to new 1
    old 2 to new 2
    so inverseMapping[0] = 1, inverseMapping[1] = 0, inverseMapping[2] = 2

    order in srcEnc is [2, 0, 1]
    thus the order in dstEnc are:
    newOrder[0] = inverseMapping[srcEncOrder[0]] = 2
    newOrder[1] = inverseMapping[srcEncOrder[1]] = 1
    newOrder[2] = inverseMapping[srcEncOrder[2]] = 0
    """
    ttgl.static_assert(
        reshaped.layout == ttgl.PaddedSharedLayout(interval_padding_pairs=[(2, 1), (4, 2), (8, 4)], order=[2, 1, 0]))


@pytest.mark.parametrize("target", ALL_TARGETS)
def test_infer_layout_for_padded_shared(target):
    # This test is used to test the conversion to gluon object PaddedSharedLayout from PaddedSharedEncodingAttr.
    # This conversion is in layoutToGluon and ttgl.permute will finally use it.
    module = run_parser(infer_layout_for_padded_shared_kernel, target=target)
    expecttest.assert_expected_inline(
        anonymize_ir(module.str_nodebug()), """\
#shared = #ttg.padded_shared<[2:+1, 4:+2, 8:+4] {order = [2, 0, 1]}>
#shared1 = #ttg.padded_shared<[2:+1, 4:+2, 8:+4] {order = [2, 1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "...", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @infer_layout_for_padded_shared_kernel() attributes {noinline = false} {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<32x4x32xi32, #shared, #smem, mutable>
    %1 = ttg.memdesc_trans %0 {order = array<i32: 1, 0, 2>} : !ttg.memdesc<32x4x32xi32, #shared, #smem, mutable> -> !ttg.memdesc<4x32x32xi32, #shared1, #smem, mutable>
    tt.return
  }
}
""")


@filecheck_test
@gluon.jit
def test_layout_zeros():
    # CHECK: #blocked = #ttg.blocked
    # CHECK: arith.constant dense<0.000000e+00> : tensor<128xf32, #blocked>
    ttgl.zeros([128], ttgl.float32, layout=ttgl.BlockedLayout([1], [32], [4], [0]))
