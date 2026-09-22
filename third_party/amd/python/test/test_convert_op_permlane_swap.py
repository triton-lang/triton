import pytest
import torch
import pathlib

import triton

from triton._internal_testing import is_hip_cdna4, is_hip_gfx1250, to_triton, numpy_random

num_ctas_list = [1]

GPU_DIALECT = "ttg"


class LinearLayout:

    def __init__(self, register, lane, warp, block):
        self.register = register
        self.lane = lane
        self.warp = warp
        self.block = block

    def __str__(self):
        return f"#{GPU_DIALECT}.linear<{{register={self.register}, lane={self.lane}, warp={self.warp}, block={self.block}}}>"


class BlockedLayout:

    def __init__(self, size_per_thread, threads_per_warp, warps_per_cta, order):
        self.sz_per_thread = size_per_thread
        self.threads_per_warp = threads_per_warp
        self.warps_per_cta = warps_per_cta
        self.order = order

    def __str__(self):
        return f"#{GPU_DIALECT}.blocked<{{sizePerThread={self.sz_per_thread}, threadsPerWarp={self.threads_per_warp}, warpsPerCTA={self.warps_per_cta}, order={self.order}}}>"


def get_src_layouts():
    threads = 64 if is_hip_cdna4() else 32
    return [BlockedLayout([1, 1], [1, threads], [1, 1], [0, 1])]


dst_layouts_cdna4 = [
    LinearLayout([[0, 32]], [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [1, 0]], [], []),
    LinearLayout([[1, 0], [0, 32]], [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [2, 0]], [], []),
    LinearLayout([[1, 0], [2, 0], [0, 32]], [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [4, 0]], [], []),
    LinearLayout([[0, 16]], [[0, 1], [0, 2], [0, 4], [0, 8], [0, 32], [1, 0]], [], []),
    LinearLayout([[1, 0], [0, 16]], [[0, 1], [0, 2], [0, 4], [0, 8], [0, 32], [2, 0]], [], []),
    LinearLayout([[1, 0], [2, 0], [0, 16]], [[0, 1], [0, 2], [0, 4], [0, 8], [0, 32], [4, 0]], [], [])
]

dst_layouts_gfx1250 = [
    LinearLayout([[0, 16]], [[0, 1], [0, 2], [0, 4], [0, 8], [1, 0]], [], []),
    LinearLayout([[1, 0], [0, 16]], [[0, 1], [0, 2], [0, 4], [0, 8], [2, 0]], [], []),
    LinearLayout([[1, 0], [2, 0], [0, 16]], [[0, 1], [0, 2], [0, 4], [0, 8], [4, 0]], [], []),
]


@pytest.mark.parametrize(
    "M, dst_layout, gpu_type",
    [[2, dst_layouts_cdna4[0], 'cdna4'], [4, dst_layouts_cdna4[1], 'cdna4'], [8, dst_layouts_cdna4[2], 'cdna4'],
     [2, dst_layouts_cdna4[3], 'cdna4'], [4, dst_layouts_cdna4[4], 'cdna4'], [8, dst_layouts_cdna4[5], 'cdna4'],
     [2, dst_layouts_gfx1250[0], 'gfx1250'], [4, dst_layouts_gfx1250[1], 'gfx1250'],
     [8, dst_layouts_gfx1250[2], 'gfx1250']])
@pytest.mark.parametrize("src_layout", get_src_layouts())
@pytest.mark.parametrize("N", [64])
@pytest.mark.parametrize("dtype", ['float8e5', 'float16', 'float32', 'int64'])
def test_convert_permlane_swap(M, N, src_layout, dst_layout, gpu_type, dtype, device, tmp_path: pathlib.Path):
    if not (is_hip_cdna4() or is_hip_gfx1250()):
        pytest.skip("Permlane swap specific tests only run on CDNA4 or GFX1250")
    if gpu_type == 'cdna4' and not is_hip_cdna4():
        pytest.skip("CDNA4 specific test")
    if gpu_type == 'gfx1250' and not is_hip_gfx1250():
        pytest.skip("GFX1250 specific test")
    if dtype == "float8e5":
        mlir_dtype = "f8E5M2"
    elif dtype == "float16":
        mlir_dtype = "f16"
    elif dtype == "float32":
        mlir_dtype = "f32"
    elif dtype == "int64":
        mlir_dtype = "i64"

    # Set threads-per-warp based on GPU type
    threads_per_warp = 64 if gpu_type == 'cdna4' else 32

    ir = f"""
    #src = {src_layout}
    #dst = {dst_layout}
    module attributes {{"ttg.num-warps" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = {threads_per_warp} : i32}} {{
  tt.func public @kernel(%arg0: !tt.ptr<{mlir_dtype}> {{tt.divisibility = 16 : i32}}, %arg1: !tt.ptr<{mlir_dtype}> {{tt.divisibility = 16 : i32}}) {{
    %0 = tt.make_range {{end = {M} : i32, start = 0 : i32}} : tensor<{M}xi32, #ttg.slice<{{dim = 1, parent = #src}}>>
    %1 = tt.make_range {{end = {N} : i32, start = 0 : i32}} : tensor<{N}xi32, #ttg.slice<{{dim = 0, parent = #src}}>>
    %2 = tt.expand_dims %0 {{axis = 1 : i32}} : tensor<{M}xi32, #ttg.slice<{{dim = 1, parent = #src}}>> -> tensor<{M}x1xi32, #src>
    %3 = tt.expand_dims %1 {{axis = 0 : i32}} : tensor<{N}xi32, #ttg.slice<{{dim = 0, parent = #src}}>> -> tensor<1x{N}xi32, #src>
    %4 = tt.broadcast %2 : tensor<{M}x1xi32, #src> -> tensor<{M}x{N}xi32, #src>
    %5 = tt.broadcast %3 : tensor<1x{N}xi32, #src> -> tensor<{M}x{N}xi32, #src>
    %cst = arith.constant dense<{N}> : tensor<{M}x{N}xi32, #src>
    %6 = arith.muli %4, %cst : tensor<{M}x{N}xi32, #src>
    %7 = arith.addi %6, %5 : tensor<{M}x{N}xi32, #src>
    %8 = tt.splat %arg0 : !tt.ptr<{mlir_dtype}> -> tensor<{M}x{N}x!tt.ptr<{mlir_dtype}>, #src>
    %9 = tt.addptr %8, %7 : tensor<{M}x{N}x!tt.ptr<{mlir_dtype}>, #src>, tensor<{M}x{N}xi32, #src>
    %10 = tt.load %9 : tensor<{M}x{N}x!tt.ptr<{mlir_dtype}>, #src>
    %11 = ttg.convert_layout %10 : tensor<{M}x{N}x{mlir_dtype}, #src> ->tensor<{M}x{N}x{mlir_dtype}, #dst>
    %12 = tt.splat %arg1 : !tt.ptr<{mlir_dtype}> -> tensor<{M}x{N}x!tt.ptr<{mlir_dtype}>, #dst>
    %13 = ttg.convert_layout %7 : tensor<{M}x{N}xi32, #src> -> tensor<{M}x{N}xi32, #dst>
    %14 = tt.addptr %12, %13 : tensor<{M}x{N}x!tt.ptr<{mlir_dtype}>, #dst>, tensor<{M}x{N}xi32, #dst>
    tt.store %14, %11 : tensor<{M}x{N}x!tt.ptr<{mlir_dtype}>, #dst>
    tt.return
  }}
}}
"""

    x = to_triton(numpy_random((M, N), dtype_str=dtype), device=device)
    z = torch.empty_like(x, device=device)

    temp_file = tmp_path / "test_convert_permlane_swap.ttgir"
    temp_file.write_text(ir)
    kernel = triton.compile(str(temp_file))

    kernel[(1, 1, 1)](x.data_ptr(), z.data_ptr())

    amdgcn = kernel.asm['amdgcn']

    if gpu_type == 'cdna4':
        assert "v_permlane32_swap" in amdgcn
    else:
        assert "v_permlane16_swap" in amdgcn

    torch.testing.assert_close(z, x, rtol=0, atol=0)
