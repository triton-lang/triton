import torch
import pytest

import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl
from triton._internal_testing import is_cuda, is_hip, is_hopper_or_newer


def _is_layout_applicable(layout) -> bool:
    if isinstance(layout, ttgl.SliceLayout):
        return _is_layout_applicable(layout.parent)
    elif is_cuda():
        mma_layout = layout.parent if isinstance(layout, ttgl.DotOperandLayout) else layout
        if not isinstance(mma_layout, ttgl.NVMMADistributedLayout):
            return False
        if mma_layout.version[0] >= 3 and not is_hopper_or_newer():
            return False
        return True
    elif is_hip():
        # TODO: Add other amd layouts
        return isinstance(layout, ttgl.amd.AMDMFMALayout)
    else:
        return True


def _filter_layouts(layouts):
    return [l for l in layouts if _is_layout_applicable(l)]


THREADS_PER_WARP = triton.runtime.driver.active.get_current_target().warp_size


@pytest.mark.parametrize("M, N", [(32, 16), (32, 32), (32, 64), (64, 32)])
@pytest.mark.parametrize(
    "src_layout",
    _filter_layouts([
        ttgl.BlockedLayout([1, 4], [4, THREADS_PER_WARP // 4], [4, 1], [0, 1]),
        ttgl.BlockedLayout([1, 4], [8, THREADS_PER_WARP // 8], [4, 1], [0, 1]),
        ttgl.BlockedLayout([4, 1], [4, THREADS_PER_WARP // 4], [1, 4], [0, 1]),
        ttgl.BlockedLayout([2, 2], [4, THREADS_PER_WARP // 4], [2, 2], [0, 1]),
        ttgl.BlockedLayout([2, 2], [8, THREADS_PER_WARP // 8], [2, 2], [0, 1]),
        ttgl.BlockedLayout([1, 4], [4, THREADS_PER_WARP // 4], [4, 1], [1, 0]),
        ttgl.BlockedLayout([1, 4], [8, THREADS_PER_WARP // 8], [4, 1], [1, 0]),
        ttgl.BlockedLayout([4, 1], [4, THREADS_PER_WARP // 4], [1, 4], [1, 0]),
        ttgl.BlockedLayout([2, 2], [4, THREADS_PER_WARP // 4], [2, 2], [1, 0]),
        ttgl.BlockedLayout([2, 2], [8, THREADS_PER_WARP // 8], [2, 2], [1, 0]),
        ttgl.BlockedLayout([1, 2], [1, THREADS_PER_WARP], [1, 4], [1, 0]),
    ]))
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("sanitize_overflow", [False, True])
def test_scan_layouts(M, N, src_layout, axis, sanitize_overflow, device):

    @gluon.jit
    def _combine(a, b):
        return a + b

    @gluon.jit
    def kernel(x_ptr, z_ptr, M: ttgl.constexpr, N: ttgl.constexpr, layout: ttgl.constexpr, axis: ttgl.constexpr):
        x_offs_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, layout))[:, None]
        x_offs_n = ttgl.arange(0, N, layout=ttgl.SliceLayout(0, layout))[None, :]
        x = ttgl.load(x_ptr + x_offs_m * N + x_offs_n)
        y = ttgl.associative_scan(x, axis=axis, combine_fn=_combine)
        ttgl.store(z_ptr + x_offs_m * N + x_offs_n, y)

    torch.manual_seed(0)

    x = torch.randint(-100, 100, (M, N), dtype=torch.int32, device=device)
    z = torch.zeros((M, N), dtype=torch.int32, device=device)
    z_tri = torch.empty_like(z)

    kernel[(1, 1, 1)](x, z_tri, M, N, src_layout, axis, num_warps=4, sanitize_overflow=sanitize_overflow,
                      debug=sanitize_overflow)

    z_ref = torch.cumsum(x, dim=axis, dtype=torch.int32)
    torch.testing.assert_close(z_tri, z_ref)


@pytest.mark.parametrize("M, N", [[128, 16], [32, 128], [32, 32], [16, 16]])
@pytest.mark.parametrize(
    "src_layout",
    _filter_layouts([
        # FIXME: Do not enable these tests until the SLPVectorizor problem with nvptx target has been resolved
        # SliceLayout(dim=1, parent=BlockedLayout([1, 4, 1], [1, 8, THREADS_PER_WARP // 8], [1, 1, 4], [2, 0, 1], [1, 1, 1], [1, 1, 1], [0, 1, 2])),
        # SliceLayout(dim=0, parent=BlockedLayout([1, 4, 1], [1, 8, THREADS_PER_WARP // 8], [1, 4, 1], [2, 1, 0], [1, 1, 1], [1, 1, 1], [0, 1, 2])),
        ttgl.BlockedLayout([1, 4], [8, THREADS_PER_WARP // 8], [4, 1], [1, 0], [1, 1], [1, 1], [0, 1]),
        ttgl.BlockedLayout([1, 4], [8, THREADS_PER_WARP // 8], [4, 1], [0, 1], [1, 1], [1, 1], [0, 1]),
        ttgl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[2, 4], ctas_per_cga=[1, 1], cta_split_num=[1, 1],
                                    cta_order=[0, 1], instr_shape=[16, 8]),
        ttgl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], ctas_per_cga=[1, 1], cta_split_num=[1, 1],
                                    cta_order=[1, 0], instr_shape=[16, 16, 16]),
        ttgl.amd.AMDMFMALayout(version=2, warps_per_cta=[4, 1], tiles_per_warp=[1, 1], instr_shape=[32, 32],
                               transposed=False),
        ttgl.amd.AMDMFMALayout(version=2, warps_per_cta=[1, 4], tiles_per_warp=[1, 1], instr_shape=[32, 32],
                               transposed=False),
        ttgl.amd.AMDMFMALayout(version=2, warps_per_cta=[4, 1], tiles_per_warp=[1, 1], instr_shape=[32, 32],
                               transposed=True),
        ttgl.amd.AMDMFMALayout(version=2, warps_per_cta=[1, 4], tiles_per_warp=[1, 1], instr_shape=[32, 32],
                               transposed=True),
        # TODO: AMDWMMA layouts
        # WmmaLayout(version=1, warps_per_cta=[4, 1]),
        # WmmaLayout(version=1, warps_per_cta=[1, 4]),
        ttgl.DotOperandLayout(
            parent=ttgl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[2, 4], ctas_per_cga=[1, 1],  #
                                               cta_split_num=[1, 1], cta_order=[0, 1], instr_shape=[16, 8]),  #
            operand_index=1, k_width=8),
        ttgl.DotOperandLayout(
            parent=ttgl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[8, 1], ctas_per_cga=[1, 1],  #
                                               cta_split_num=[1, 1], cta_order=[1, 0], instr_shape=[16, 32, 16]),  #
            operand_index=0, k_width=2),
        ttgl.SliceLayout(
            dim=0,
            parent=ttgl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[4, 1, 1], ctas_per_cga=[1, 1, 1],  #
                                               cta_split_num=[1, 1, 1], cta_order=[2, 1, 0], instr_shape=[1, 16,
                                                                                                          8])),  #
        ttgl.SliceLayout(
            dim=1, parent=ttgl.DotOperandLayout(
                parent=ttgl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[4, 1, 1], ctas_per_cga=[1, 1, 1],  #
                                                   cta_split_num=[1, 1, 1], cta_order=[2, 1, 0], instr_shape=[1, 16,
                                                                                                              8]),  #
                operand_index=1, k_width=2)),
        "linear_layout",
    ]))
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("epilogue_kind", ['reduce1d', 'reduce2d', 'expand_reduce2d'])
@pytest.mark.parametrize("dtype_str, sanitize_overflow", [("int32", False), ("int32", True), ("float32", False),
                                                          ("float16", False)])
@pytest.mark.parametrize("reduce_op", ["sum", "max"])
def test_reduce_layouts(M, N, src_layout, axis, epilogue_kind, dtype_str, sanitize_overflow, reduce_op, device):
    if src_layout == "linear_layout":
        src_layout = ttgl.DistributedLinearLayout(reg_bases=[[0, 16], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0]],  #
                                                  lane_bases=[[0, 0], [0, 1], [0, 2], [0, 4], [0, 8]],  #
                                                  warp_bases=[[32, 0], [0, 32]], block_bases=[], shape=[M, N])
        if THREADS_PER_WARP != (1 << len(src_layout.lane_bases)):
            pytest.skip(f"Skipping. This LinearLayout assumes {1 << len(src_layout.lane_bases)} threads per warp")
        elif M < 64 or N < 64:
            pytest.skip(f"Skipping. This LinearLayout assumes M >= 64 and N >= 64, got M={M}, N={N}")
    if isinstance(src_layout,
                  (ttgl.amd.AMDMFMALayout, ttgl.NVMMADistributedLayout)) and (M < src_layout.instr_shape[0]
                                                                              or N < src_layout.instr_shape[1]):
        pytest.skip("Skipping because tensor shape is smaller than M(f)maLayout instr_shape")

    @gluon.jit
    def _add(a, b):
        return a + b

    @gluon.jit
    def _max(a, b):
        return ttgl.maximum(a, b)

    combine_fn = _add if reduce_op == "sum" else _max

    @gluon.jit
    def kernel(x_ptr, z_ptr, M: ttgl.constexpr, N: ttgl.constexpr, layout: ttgl.constexpr, axis: ttgl.constexpr,
               epilogue_kind: ttgl.constexpr):
        x_offs_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, layout))[:, None]
        x_offs_n = ttgl.arange(0, N, layout=ttgl.SliceLayout(0, layout))[None, :]
        x = ttgl.load(x_ptr + x_offs_m * N + x_offs_n)
        y = ttgl.reduce(x, axis=axis, combine_fn=combine_fn)
        if epilogue_kind == "reduce1d":
            if axis == 0:
                z_offs = ttgl.arange(0, N, layout=ttgl.SliceLayout(0, layout))
            else:
                z_offs = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, layout))
            ttgl.store(z_ptr + z_offs, y)
        elif epilogue_kind == "reduce2d":
            y = ttgl.reduce(y, axis=0, combine_fn=combine_fn)
            ttgl.store(z_ptr, y)
        elif epilogue_kind == "expand_reduce2d":
            y = ttgl.expand_dims(y, axis=axis)
            y = ttgl.reduce(y, axis=1 - axis, combine_fn=combine_fn)
            z_offs = ttgl.arange(0, 1, layout=ttgl.SliceLayout(1 - axis, layout))
            ttgl.store(z_ptr + z_offs, y)

    torch.manual_seed(0)

    torch_dtype = getattr(torch, dtype_str)
    x = torch.randint(-10, 10, (M, N), dtype=torch.int32, device=device).to(torch_dtype)
    out_shape = (1, 1) if "reduce2d" in epilogue_kind else (1, N) if axis == 0 else (M, 1)
    z = torch.empty(out_shape, dtype=torch_dtype, device=device)

    num_warps = int(torch.prod(torch.tensor(ttgl._layouts.warps_per_cta(src_layout, (M, N)))))
    kernel[(1, 1, 1)](x, z, M, N, src_layout, axis, num_warps=num_warps, epilogue_kind=epilogue_kind,
                      sanitize_overflow=sanitize_overflow, debug=sanitize_overflow)

    reduce_fn = torch.sum if reduce_op == "sum" else torch.amax
    z_ref = reduce_fn(x, dim=axis, keepdim=True)
    if epilogue_kind in ("expand_reduce2d", "reduce2d"):
        z_ref = reduce_fn(z_ref, dim=1 - axis, keepdim=True)
    torch.testing.assert_close(z, z_ref.to(torch_dtype))
