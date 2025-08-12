import pytest
import torch
import pathlib

import triton

from triton._internal_testing import is_hip

num_ctas_list = [1]

GPU_DIALECT = "ttg"

if is_hip():
    THREADS_PER_WARP = triton.runtime.driver.active.get_current_target().warp_size
else:
    THREADS_PER_WARP = 32


class LinearLayout:

    def __init__(self, register, lane, warp, block):
        self.register = register
        self.lane = lane
        self.warp = warp
        self.block = block

    def __str__(self):
        return f"#{GPU_DIALECT}.linear<{{register={self.register}, lane={self.lane}, warp={self.warp}, block={self.block}}}>"


class BlockedLayout:

    def __init__(self, size_per_thread, threads_per_warp, warps_per_cta, order, ctas_per_cga, cta_split_num, cta_order):
        self.sz_per_thread = size_per_thread
        self.threads_per_warp = threads_per_warp
        self.warps_per_cta = warps_per_cta
        self.order = order
        self.ctas_per_cga = ctas_per_cga
        self.cta_split_num = cta_split_num
        self.cta_order = cta_order

    def __str__(self):
        return f"#{GPU_DIALECT}.blocked<{{sizePerThread={self.sz_per_thread}, threadsPerWarp={self.threads_per_warp}, warpsPerCTA={self.warps_per_cta}, order={self.order}, CTAsPerCGA={self.ctas_per_cga}, CTASplitNum={self.cta_split_num}, CTAOrder={self.cta_order}}}>"


# -----------------------
# test extract slice
# -----------------------

regs2x2 = [[1, 0], [0, 1]]
lanes8x8 = [[2, 0], [4, 0], [8, 0], [0, 2], [0, 4], [0, 8]]
warps2x2 = [[16, 0], [0, 16]]
cta_layout = [[1, 1], [1, 1], [0, 1]]
redundant_ll = LinearLayout([[0, 0]] + regs2x2, lanes8x8, warps2x2, block=[])
non_redundant_ll = LinearLayout(regs2x2, lanes8x8, warps2x2, block=[])

# list of pairs defining ExtractSliceOp input and output layouts
extract_layout = [
    (BlockedLayout([1, 8], [16, 4], [4, 1], [1, 0], *cta_layout), ) * 2,
    (BlockedLayout([2, 2], [64, 1], [2, 2], [1, 0], *cta_layout), ) * 2,
    (BlockedLayout([2, 2], [16, 4], [4, 1], [0, 1], *cta_layout), ) * 2,
    (BlockedLayout([1, 8], [16, 4], [4, 1], [1, 0], *cta_layout), ) * 2,
    (BlockedLayout([1, 8], [16, 4], [4, 1], [0, 1], *cta_layout), ) * 2,
    (redundant_ll, non_redundant_ll),
    (non_redundant_ll, redundant_ll),
]
blocked_layout = [
    BlockedLayout([1, 8], [16, 4], [4, 1], [1, 0], *cta_layout),
    BlockedLayout([2, 2], [16, 4], [2, 2], [1, 0], *cta_layout),
    BlockedLayout([2, 2], [16, 4], [2, 2], [0, 1], *cta_layout),
    BlockedLayout([1, 8], [16, 4], [4, 1], [1, 0], *cta_layout),
    BlockedLayout([1, 8], [16, 4], [4, 1], [0, 1], *cta_layout),
]


@pytest.mark.parametrize(
    "M, N, M_tile_size, N_tile_size, M_tile_offset, N_tile_offset",
    [[256, 256, 256, 32, 0, 32], [128, 128, 128, 64, 0, 64], [1, 512, 1, 256, 0, 256], [512, 1, 256, 1, 256, 0]])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("extract_layout", extract_layout)
@pytest.mark.parametrize("blocked_layout", blocked_layout)
def test_extract_slice(dtype, M, N, M_tile_size, N_tile_size, M_tile_offset, N_tile_offset, blocked_layout,
                       extract_layout, device, tmp_path: pathlib.Path):
    if not is_hip():
        pytest.skip("extract_slice is AMD specific instruction.")

    ir = f"""
    #blocked = {blocked_layout}
    #src_extract_layout = {extract_layout[0]}
    #dst_extract_layout = {extract_layout[1]}
    module attributes {{"ttg.num-ctas" = 1, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = {str(64)} : i32}} {{
    tt.func public @kernel(%arg0: !tt.ptr<f16> {{tt.divisibility = 16 : i32}}, %arg1: !tt.ptr<f16> {{tt.divisibility = 16 : i32}}) {{
        %cst = arith.constant dense<{N}> : tensor<{M}x1xi32, #blocked>
        %cst_n = arith.constant dense<{N_tile_size}> : tensor<{M_tile_size}x1xi32, #blocked>
        %0 = tt.make_range {{end = {M} : i32, start = 0 : i32}} : tensor<{M}xi32, #ttg.slice<{{dim = 1, parent = #blocked}}>>
        %42 = tt.make_range {{end = {M_tile_size} : i32, start = 0 : i32}} : tensor<{M_tile_size}xi32, #ttg.slice<{{dim = 1, parent = #blocked}}>>
        %1 = tt.make_range {{end = {N} : i32, start = 0 : i32}} : tensor<{N}xi32, #ttg.slice<{{dim = 0, parent = #blocked}}>>
        %2 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<{M}x{N}x!tt.ptr<f16>, #blocked>
        %4 = tt.expand_dims %0 {{axis = 1 : i32}} : tensor<{M}xi32, #ttg.slice<{{dim = 1, parent = #blocked}}>> -> tensor<{M}x1xi32, #blocked>
        %43 = tt.expand_dims %42 {{axis = 1 : i32}} : tensor<{M_tile_size}xi32, #ttg.slice<{{dim = 1, parent = #blocked}}>> -> tensor<{M_tile_size}x1xi32, #blocked>
        %5 = arith.muli %4, %cst : tensor<{M}x1xi32, #blocked>
        %44 = arith.muli %43, %cst_n : tensor<{M_tile_size}x1xi32, #blocked>
        %6 = tt.expand_dims %1 {{axis = 0 : i32}} : tensor<{N}xi32, #ttg.slice<{{dim = 0, parent = #blocked}}>> -> tensor<1x{N}xi32, #blocked>
        %7 = tt.broadcast %6 : tensor<1x{N}xi32, #blocked> -> tensor<{M}x{N}xi32, #blocked>
        %8 = tt.broadcast %5 : tensor<{M}x1xi32, #blocked> -> tensor<{M}x{N}xi32, #blocked>
        %9 = arith.addi %8, %7 : tensor<{M}x{N}xi32, #blocked>
        %33 = tt.make_range {{end = {N_tile_size} : i32, start = 0 : i32}} : tensor<{N_tile_size}xi32, #ttg.slice<{{dim = 0, parent = #blocked}}>>
        %34 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<{M_tile_size}x{N_tile_size}x!tt.ptr<f16>, #blocked>
        %37 = tt.expand_dims %33 {{axis = 0 : i32}} : tensor<{N_tile_size}xi32, #ttg.slice<{{dim = 0, parent = #blocked}}>> -> tensor<1x{N_tile_size}xi32, #blocked>
        %38 = tt.broadcast %37 : tensor<1x{N_tile_size}xi32, #blocked> -> tensor<{M_tile_size}x{N_tile_size}xi32, #blocked>
        %39 = tt.broadcast %44 : tensor<{M_tile_size}x1xi32, #blocked> -> tensor<{M_tile_size}x{N_tile_size}xi32, #blocked>
        %40 = arith.addi %38, %39 : tensor<{M_tile_size}x{N_tile_size}xi32, #blocked>
        %10 = tt.addptr %2, %9 : tensor<{M}x{N}x!tt.ptr<f16>, #blocked>, tensor<{M}x{N}xi32, #blocked>
        %11 = tt.load %10 {{cache = 1 : i32, evict = 1 : i32, isVolatile = false}} : tensor<{M}x{N}x!tt.ptr<f16>, #blocked>
        %12 = ttg.convert_layout %11 : tensor<{M}x{N}xf16, #blocked> -> tensor<{M}x{N}xf16, #src_extract_layout>
        %13 = amdgpu.extract_slice %12 [{M_tile_offset}, {N_tile_offset}] : tensor<{M}x{N}xf16, #src_extract_layout> to tensor<{M_tile_size}x{N_tile_size}xf16, #dst_extract_layout>
        %14 = ttg.convert_layout %13 : tensor<{M_tile_size}x{N_tile_size}xf16, #dst_extract_layout> -> tensor<{M_tile_size}x{N_tile_size}xf16, #blocked>
        %15 = tt.addptr %34, %40 : tensor<{M_tile_size}x{N_tile_size}x!tt.ptr<f16>, #blocked>, tensor<{M_tile_size}x{N_tile_size}xi32, #blocked>
        tt.store %15, %14 : tensor<{M_tile_size}x{N_tile_size}x!tt.ptr<f16>, #blocked>
        tt.return
    }}
    }}
    """
    x = torch.randn((M, N), device=device, dtype=dtype)

    temp_file = tmp_path / "test_extract_slice.ttgir"
    temp_file.write_text(ir)
    kernel = triton.compile(str(temp_file))

    extract_slice = torch.empty((M_tile_size, N_tile_size), device=device, dtype=dtype)

    kernel[(1, 1, 1)](x.data_ptr(), extract_slice)
    test_result = torch.equal(x[M_tile_offset:M_tile_size + M_tile_offset, N_tile_offset:N_tile_offset + N_tile_size],
                              extract_slice)

    assert test_result


# -----------------------
# test concat op
# -----------------------

cta_layout = [[1, 1], [1, 1], [0, 1]]
blocked_32x32 = BlockedLayout([2, 2], [8, 8], [2, 2], [0, 1], *cta_layout)
broadcasted_32x32 = LinearLayout(register=[[0, 0], [1, 0], [0, 1]], lane=[[2, 0], [4, 0], [8, 0], [0, 2], [0, 4],
                                                                          [0, 8]], warp=[[16, 0], [0, 16]], block=[])

src_layout = [
    LinearLayout(register=[[0, 1], [0, 2], [0, 8], [0, 16], [0, 64], [64, 0]], lane=[[1, 0], [2, 0], [4, 0], [8, 0],
                                                                                     [16, 0], [0, 4]], warp=[[0, 32],
                                                                                                             [32, 0]],
                 block=[]),
    LinearLayout(register=[[1, 0], [2, 0], [4, 0]], lane=[[0, 1], [0, 2], [0, 4], [0, 8], [8, 0], [16, 0]],
                 warp=[[0, 16], [0, 32]], block=[]),
]

dst_layout = [
    LinearLayout(register=[[0, 1], [0, 2], [0, 8], [0, 16], [0, 64], [0, 128], [64, 0], [128, 0]],
                 lane=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp=[[0, 32], [32, 0]], block=[]),
    LinearLayout(register=[[1, 0], [2, 0], [4, 0], [32, 0], [0, 32]], lane=[[0, 1], [0, 2], [0, 4], [0, 8], [8, 0],
                                                                            [16, 0]], warp=[[0, 16], [0, 64]],
                 block=[]),
]


@pytest.mark.parametrize(
    "src_layout, dst_layout, M, N, M_tile_size, N_tile_size",
    [[src_layout[0], dst_layout[0], 128, 128, 256, 256], [src_layout[1], dst_layout[1], 32, 32, 64, 64],
     [broadcasted_32x32, blocked_32x32, 32, 32, 64, 64], [blocked_32x32, broadcasted_32x32, 32, 32, 64, 64]])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_concat_op(dtype, M, N, M_tile_size, N_tile_size, src_layout, dst_layout, device, tmp_path: pathlib.Path):
    if not is_hip():
        pytest.skip("concat op is AMD specific instruction.")

    ir = f"""
    #blocked = #ttg.blocked<{{sizePerThread=[1, 8], threadsPerWarp=[16, 4], warpsPerCTA=[4, 1], order=[1, 0], CTAsPerCGA=[1, 1], CTASplitNum=[1, 1], CTAOrder=[0, 1]}}>
    #src_layout = {src_layout}
    #dst_layout = {dst_layout}

    module attributes {{"ttg.num-ctas" = 1, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = {str(64)} : i32}} {{
    tt.func public @kernel(%arg0: !tt.ptr<f16> {{tt.divisibility = 16 : i32}}, %arg1: !tt.ptr<f16> {{tt.divisibility = 16 : i32}}, %arg2: !tt.ptr<f16> {{tt.divisibility = 16 : i32}}, %arg3: !tt.ptr<f16> {{tt.divisibility = 16 : i32}}, %arg4: !tt.ptr<f16> {{tt.divisibility = 16 : i32}}) {{
        %cst = arith.constant dense<{N}> : tensor<{M}x1xi32, #blocked>
        %cst_n = arith.constant dense<{N_tile_size}> : tensor<{M_tile_size}x1xi32, #blocked>
        %0 = tt.make_range {{end = {M} : i32, start = 0 : i32}} : tensor<{M}xi32, #ttg.slice<{{dim = 1, parent = #blocked}}>>
        %42 = tt.make_range {{end = {M_tile_size} : i32, start = 0 : i32}} : tensor<{M_tile_size}xi32, #ttg.slice<{{dim = 1, parent = #blocked}}>>
        %1 = tt.make_range {{end = {M} : i32, start = 0 : i32}} : tensor<{M}xi32, #ttg.slice<{{dim = 0, parent = #blocked}}>>
        %2 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<{M}x{N}x!tt.ptr<f16>, #blocked>
        %100 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<{M}x{N}x!tt.ptr<f16>, #blocked>
        %101 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<{M}x{N}x!tt.ptr<f16>, #blocked>
        %102 = tt.splat %arg3 : !tt.ptr<f16> -> tensor<{M}x{N}x!tt.ptr<f16>, #blocked>
        %4 = tt.expand_dims %0 {{axis = 1 : i32}} : tensor<{M}xi32, #ttg.slice<{{dim = 1, parent = #blocked}}>> -> tensor<{M}x1xi32, #blocked>
        %43 = tt.expand_dims %42 {{axis = 1 : i32}} : tensor<{M_tile_size}xi32, #ttg.slice<{{dim = 1, parent = #blocked}}>> -> tensor<{M_tile_size}x1xi32, #blocked>
        %5 = arith.muli %4, %cst : tensor<{M}x1xi32, #blocked>
        %44 = arith.muli %43, %cst_n : tensor<{M_tile_size}x1xi32, #blocked>
        %6 = tt.expand_dims %1 {{axis = 0 : i32}} : tensor<{M}xi32, #ttg.slice<{{dim = 0, parent = #blocked}}>> -> tensor<1x{M}xi32, #blocked>
        %7 = tt.broadcast %6 : tensor<1x{M}xi32, #blocked> -> tensor<{M}x{N}xi32, #blocked>
        %8 = tt.broadcast %5 : tensor<{M}x1xi32, #blocked> -> tensor<{M}x{N}xi32, #blocked>
        %9 = arith.addi %8, %7 : tensor<{M}x{N}xi32, #blocked>
        %33 = tt.make_range {{end = {N_tile_size} : i32, start = 0 : i32}} : tensor<{N_tile_size}xi32, #ttg.slice<{{dim = 0, parent = #blocked}}>>
        %34 = tt.splat %arg4 : !tt.ptr<f16> -> tensor<{M_tile_size}x{N_tile_size}x!tt.ptr<f16>, #blocked>
        %37 = tt.expand_dims %33 {{axis = 0 : i32}} : tensor<{N_tile_size}xi32, #ttg.slice<{{dim = 0, parent = #blocked}}>> -> tensor<1x{N_tile_size}xi32, #blocked>
        %38 = tt.broadcast %37 : tensor<1x{N_tile_size}xi32, #blocked> -> tensor<{M_tile_size}x{N_tile_size}xi32, #blocked>
        %39 = tt.broadcast %44 : tensor<{M_tile_size}x1xi32, #blocked> -> tensor<{M_tile_size}x{N_tile_size}xi32, #blocked>
        %40 = arith.addi %38, %39 : tensor<{M_tile_size}x{N_tile_size}xi32, #blocked>
        %10 = tt.addptr %2, %9 : tensor<{M}x{N}x!tt.ptr<f16>, #blocked>, tensor<{M}x{N}xi32, #blocked>
        %200 = tt.addptr %100, %9 : tensor<{M}x{N}x!tt.ptr<f16>, #blocked>, tensor<{M}x{N}xi32, #blocked>
        %201 = tt.addptr %101, %9 : tensor<{M}x{N}x!tt.ptr<f16>, #blocked>, tensor<{M}x{N}xi32, #blocked>
        %202 = tt.addptr %102, %9 : tensor<{M}x{N}x!tt.ptr<f16>, #blocked>, tensor<{M}x{N}xi32, #blocked>
        %11 = tt.load %10 {{cache = 1 : i32, evict = 1 : i32, isVolatile = false}} : tensor<{M}x{N}x!tt.ptr<f16>, #blocked>
        %300 = tt.load %200 {{cache = 1 : i32, evict = 1 : i32, isVolatile = false}} : tensor<{M}x{N}x!tt.ptr<f16>, #blocked>
        %301 = tt.load %201 {{cache = 1 : i32, evict = 1 : i32, isVolatile = false}} : tensor<{M}x{N}x!tt.ptr<f16>, #blocked>
        %302 = tt.load %202 {{cache = 1 : i32, evict = 1 : i32, isVolatile = false}} : tensor<{M}x{N}x!tt.ptr<f16>, #blocked>

        %12 = ttg.convert_layout %11 : tensor<{M}x{N}xf16, #blocked> -> tensor<{M}x{N}xf16, #src_layout>
        %400 = ttg.convert_layout %300 : tensor<{M}x{N}xf16, #blocked> -> tensor<{M}x{N}xf16, #src_layout>
        %401 = ttg.convert_layout %301 : tensor<{M}x{N}xf16, #blocked> -> tensor<{M}x{N}xf16, #src_layout>
        %402 = ttg.convert_layout %302 : tensor<{M}x{N}xf16, #blocked> -> tensor<{M}x{N}xf16, #src_layout>

        %13 = amdgpu.concat %12, %400, %401, %402 : tensor<{M}x{N}xf16, #src_layout>, tensor<{M}x{N}xf16, #src_layout>, tensor<{M}x{N}xf16, #src_layout>, tensor<{M}x{N}xf16, #src_layout> -> tensor<{M_tile_size}x{N_tile_size}xf16, #dst_layout>
        %14 = ttg.convert_layout %13 : tensor<{M_tile_size}x{N_tile_size}xf16, #dst_layout> -> tensor<{M_tile_size}x{N_tile_size}xf16, #blocked>
        %15 = tt.addptr %34, %40 : tensor<{M_tile_size}x{N_tile_size}x!tt.ptr<f16>, #blocked>, tensor<{M_tile_size}x{N_tile_size}xi32, #blocked>
        tt.store %15, %14 : tensor<{M_tile_size}x{N_tile_size}x!tt.ptr<f16>, #blocked>
        tt.return
    }}
    }}
    """
    x1 = torch.randn((M, N), device=device, dtype=dtype)
    x2 = torch.randn((M, N), device=device, dtype=dtype)
    x3 = torch.randn((M, N), device=device, dtype=dtype)
    x4 = torch.randn((M, N), device=device, dtype=dtype)

    temp_file = tmp_path / "test_concat_op.ttgir"
    temp_file.write_text(ir)
    kernel = triton.compile(str(temp_file))

    concat = torch.empty((M_tile_size, N_tile_size), device=device, dtype=dtype)
    kernel[(1, 1, 1)](x1.data_ptr(), x2.data_ptr(), x3.data_ptr(), x4.data_ptr(), concat)

    top = torch.cat([x1, x2], dim=1)
    bottom = torch.cat([x3, x4], dim=1)
    result = torch.cat([top, bottom], dim=0)

    test_result = torch.equal(result, concat)
    assert test_result
