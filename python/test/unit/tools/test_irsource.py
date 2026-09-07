import pathlib
import pytest
import torch
import triton
from triton.compiler import IRSource, make_backend
from triton._C.libtriton import ir
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl

target = triton.runtime.driver.active.get_current_target()
backend = make_backend(target)


def test_mlir_attribute_parsing(tmp_path: pathlib.Path) -> None:
    '''
    Tests that MLIR attributes are parsed correctly from input ttir/ttgir.

    Checks for the following:
    1. Name and type signature are parsed correctly
    2. _get_num_warps_from_ir_str() works
    3. tt.nv_tma_desc attribute is parsed correctly
    '''

    sample_ttgir = r"""
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 2], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [16, 8]}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                %arg3: i32 {tt.divisibility = 16 : i32},
                                %arg4: i32 {tt.divisibility = 16 : i32},
                                %arg5: i32 {tt.divisibility = 16 : i32},
                                %arg6: i32 {tt.divisibility = 16 : i32},
                                %arg7: i32 {tt.divisibility = 16 : i32},
                                %arg8: i32 {tt.divisibility = 16 : i32, tt.nv_tma_desc = 0 : i32},
                                %desc: !tt.ptr<i8, "descriptor"> {tt.nv_tma_desc = 1 : i32}) attributes {noinline = false} {
    tt.return
  }
}
"""
    temp_file = tmp_path / "test_mlir_attribute_parsing0.ttgir"
    temp_file.write_text(sample_ttgir)
    context = ir.context()
    src = IRSource(str(temp_file), context, backend)

    # check name and type signature
    # should match ty_to_cpp(...)
    assert  src.signature == \
                {0: "*f32", 1: "*f32", 2: "*f32", 3: "i32", \
                        4: "i32", 5: "i32", 6: "i32", 7: "i32", 8: "nvTmaDesc", 9: "nvTmaDesc"}
    assert src.name == "@matmul_kernel"

    # check num warps
    assert src.parse_options()['num_warps'] == 8

    sample_ttgir_vector_add = r"""
    #blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
    module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
       tt.func public @add_kernel(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32},
       %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32},
       %arg2: !tt.ptr<i32> {tt.divisibility = 16 : i32},
       %arg3: i32 {tt.divisibility = 16 : i32})
        attributes {noinline = false} {
         %c1024_i32 = arith.constant 1024 : i32
         %0 = tt.get_program_id x : i32
         %1 = arith.muli %0, %c1024_i32 : i32
         %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
         %3 = tt.splat %1 : i32 -> tensor<1024xi32, #blocked>
         %4 = arith.addi %3, %2 : tensor<1024xi32, #blocked>
         %5 = tt.splat %arg3 : i32 -> tensor<1024xi32, #blocked>
         %6 = arith.cmpi slt, %4, %5 : tensor<1024xi32, #blocked>
         %7 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>, #blocked>
         %8 = tt.addptr %7, %4 : tensor<1024x!tt.ptr<i32>, #blocked>, tensor<1024xi32, #blocked>
         %9 = tt.load %8, %6 : tensor<1024x!tt.ptr<i32>, #blocked>
         %10 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>, #blocked>
         %11 = tt.addptr %10, %4 : tensor<1024x!tt.ptr<i32>, #blocked>, tensor<1024xi32, #blocked>
         %12 = tt.load %11, %6 : tensor<1024x!tt.ptr<i32>, #blocked>
         %13 = arith.addi %9, %12 : tensor<1024xi32, #blocked>
         %14 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>, #blocked>
         %15 = tt.addptr %14, %4 : tensor<1024x!tt.ptr<i32>, #blocked>, tensor<1024xi32, #blocked>
         tt.store %15, %13, %6 : tensor<1024x!tt.ptr<i32>, #blocked>
         tt.return
       }
     }
    """
    temp_file = tmp_path / "test_mlir_attribute_parsing1.ttgir"
    temp_file.write_text(sample_ttgir_vector_add)
    context = ir.context()
    src = IRSource(str(temp_file), context, backend)

    # now test compilation
    triton.compile(str(temp_file), target=target)

    # Identical IR with a different extension must use its own pipeline and cache entry.
    glir_file = temp_file.with_suffix(".glir")
    glir_file.write_text(sample_ttgir_vector_add)
    assert "glir" in triton.compile(str(glir_file), target=target).asm


@pytest.mark.parametrize("num_warps", [4, 8])
@pytest.mark.parametrize("num_ctas, coalesced", [(1, False), (2, False), (1, True)])
def test_gluon_ir_file(tmp_path, device, num_warps, num_ctas, coalesced):
    if num_ctas > 1 and (target.backend != "cuda" or target.arch < 90):
        pytest.skip("requires CUDA cluster support")
    if coalesced and target.backend != "cuda":
        pytest.skip("coalesced layouts require CUDA")

    @gluon.jit
    def kernel(In, Out, Layout: ttgl.constexpr, Increment: ttgl.constexpr):
        offsets = ttgl.arange(0, 1024, layout=ttgl.AutoLayout())
        in_ptrs = ttgl.set_auto_layout(In + offsets, Layout)
        out_ptrs = ttgl.set_auto_layout(Out + offsets, Layout)
        ttgl.store(out_ptrs, ttgl.load(in_ptrs) + Increment)

    if coalesced:
        layout = ttgl.CoalescedLayout()
    else:
        cga_layout = [[1]] if num_ctas == 2 else []
        layout = ttgl.BlockedLayout([1], [target.warp_size], [num_warps], [0], cga_layout=cga_layout)
    x = torch.arange(1024, dtype=torch.int32, device=device)
    y = torch.empty_like(x)
    compiled = kernel[(1, )](x, y, layout, 1, num_warps=num_warps, num_ctas=num_ctas)
    assert "#gluon.auto_encoding" in compiled.asm["glir"]
    assert "gluon" not in compiled.asm["ttgir"]

    path = tmp_path / "kernel.glir"
    path.write_text(compiled.asm["glir"])
    from_file = triton.compile(str(path), target=target)
    assert from_file.metadata.num_warps == num_warps
    assert from_file.metadata.num_ctas == num_ctas
    y.fill_(-1)
    from_file[(1, 1, 1)](x, y)
    torch.testing.assert_close(y, x + 1)

    # A GLIR override must be parsed before resolving Gluon layouts.
    y.fill_(-1)
    kernel[(1, )](x, y, layout, 2, num_warps=num_warps, num_ctas=num_ctas, ir_override=str(path))
    torch.testing.assert_close(y, x + 1)
