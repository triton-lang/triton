import expecttest

from triton import knobs
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl


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
        h.asm["ttgir"], """\
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


@gluon.jit
def shared_memory_kernel(XBLOCK: ttgl.constexpr, YBLOCK: ttgl.constexpr, layout_a: ttgl.constexpr,
                         layout_b: ttgl.constexpr, smem_layout: ttgl.constexpr):
    a = ttgl.full([XBLOCK, YBLOCK], 0, ttgl.int32, layout_a)
    mem = ttgl.allocate_shared(ttgl.int32, a.shape, smem_layout, a)
    b = mem.load(layout_b)  # noqa: F841
    mem.store(a)


def test_shared_memory(fresh_knobs):
    knobs.compilation.disable_line_info = True

    layout_a = ttgl.BlockedLayout(size_per_thread=[1, 1], threads_per_warp=[1, 32], warps_per_cta=[4, 1], order=[1, 0])
    layout_b = ttgl.BlockedLayout(size_per_thread=[1, 4], threads_per_warp=[1, 32], warps_per_cta=[4, 1], order=[1, 0])
    smem_layout = ttgl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=32, rank=2)
    h = shared_memory_kernel.warmup(8, 32, layout_a, layout_b, smem_layout, num_warps=layout_a.warps_per_cta[0],
                                    grid=(1, ))
    expecttest.assert_expected_inline(
        h.asm["ttgir"], """\
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func public @shared_memory_kernel() attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32 loc(#loc)
    %cst = arith.constant dense<0> : tensor<8x32xi32, #blocked> loc(#loc)
    %0 = ttg.local_alloc %cst : (tensor<8x32xi32, #blocked>) -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable> loc(#loc)
    %1 = ttg.local_load %0 : !ttg.memdesc<8x32xi32, #shared, #smem, mutable> -> tensor<8x32xi32, #blocked1> loc(#loc)
    ttg.local_store %cst, %0 : tensor<8x32xi32, #blocked> -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable> loc(#loc)
    tt.return loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
""")
