import torch
import pytest

import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl
from triton._internal_testing import is_cuda, is_hip, is_hopper_or_newer, get_hip_lds_size


def _is_layout_applicable(layout) -> bool:
    if isinstance(layout, (ttgl.BlockedLayout, ttgl.SwizzledSharedLayout, ttgl.DistributedLinearLayout)):
        return True
    elif isinstance(layout, ttgl.SliceLayout):
        return _is_layout_applicable(layout.parent)
    elif is_cuda():
        if isinstance(layout, ttgl.NVMMASharedLayout):
            return True
        mma_layout = layout.parent if isinstance(layout, ttgl.DotOperandLayout) else layout
        if not isinstance(mma_layout, ttgl.NVMMADistributedLayout):
            return False
        if mma_layout.version[0] >= 3 and not is_hopper_or_newer():
            return False
        return True
    elif is_hip():
        if isinstance(layout, ttgl.PaddedSharedLayout):
            return True
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


@pytest.mark.parametrize("M", [32, 64, 128, 256])
@pytest.mark.parametrize(
    "src_layout",
    _filter_layouts([
        ttgl.BlockedLayout([1, 4], [1, THREADS_PER_WARP], [4, 1], [1, 0], [1, 1], [1, 1], [0, 1]),
        ttgl.BlockedLayout([1, 4], [1, THREADS_PER_WARP], [2, 2], [1, 0], [1, 1], [1, 1], [0, 1]),
        ttgl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[4, 1], ctas_per_cga=[1, 1], cta_split_num=[1, 1],
                                    cta_order=[0, 1], instr_shape=[16, 8]),
    ]))
def test_store_layouts(M, src_layout, device):

    @gluon.jit
    def kernel(x_ptr, y_ptr, M: ttgl.constexpr, layout: ttgl.constexpr):
        offs = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, layout))
        x = ttgl.load(x_ptr + offs)
        x_2d = ttgl.expand_dims(x, axis=1)
        offs_2d = ttgl.expand_dims(offs, axis=1)
        ttgl.store(y_ptr + offs_2d, x_2d)

    torch.manual_seed(17)
    x = torch.randint(0, 4, (M, 1), dtype=torch.float32, device=device)
    y = torch.zeros((M, 1), dtype=torch.float32, device=device)
    kernel[(1, )](x, y, M, src_layout, num_warps=4)
    torch.testing.assert_close(y, x)


_1d_layouts = _filter_layouts([
    ttgl.BlockedLayout([1, 4], [1, THREADS_PER_WARP], [4, 1], [1, 0], [1, 1], [1, 1], [0, 1]),
    ttgl.BlockedLayout([1, 4], [1, THREADS_PER_WARP], [2, 2], [1, 0], [1, 1], [1, 1], [0, 1]),
    ttgl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], ctas_per_cga=[1, 1], cta_split_num=[1, 1],
                                cta_order=[1, 0], instr_shape=[16, 32, 16]),
    ttgl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[4, 1], ctas_per_cga=[1, 1], cta_split_num=[1, 1],
                                cta_order=[0, 1], instr_shape=[16, 8]),
    ttgl.DotOperandLayout(
        parent=ttgl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], ctas_per_cga=[1, 1],
                                           cta_split_num=[1, 1], cta_order=[1, 0], instr_shape=[16, 32, 16]),
        operand_index=0, k_width=2),
    ttgl.DotOperandLayout(
        parent=ttgl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[2, 2], ctas_per_cga=[1, 1],
                                           cta_split_num=[1, 1], cta_order=[0, 1], instr_shape=[16, 8]),
        operand_index=0, k_width=2),
])


@pytest.mark.parametrize("M, bins", [[2048, 2], [8, 512], [32, 32]])
@pytest.mark.parametrize("src_layout", [ttgl.BlockedLayout([1], [THREADS_PER_WARP], [4], [0]), "linear_layout"])
@pytest.mark.parametrize("dst_layout", [ttgl.BlockedLayout([1], [THREADS_PER_WARP], [4], [0])])
def test_histogram(M, bins, src_layout, dst_layout, device):

    @gluon.jit
    def kernel(x_ptr, z_ptr, M: ttgl.constexpr, B: ttgl.constexpr, src_layout: ttgl.constexpr,
               dst_layout: ttgl.constexpr):
        offs = ttgl.arange(0, M, layout=src_layout)
        x = ttgl.load(x_ptr + offs)
        h = ttgl.histogram(x, B, layout=dst_layout)
        z_offs = ttgl.arange(0, B, layout=dst_layout)
        ttgl.store(z_ptr + z_offs, h)

    if src_layout == "linear_layout":
        if M == 32:
            src_layout = ttgl.DistributedLinearLayout(
                reg_bases=[],
                lane_bases=[[0], [16], [4], [2], [1]] + [[0]] * (THREADS_PER_WARP >> 6),
                warp_bases=[[0], [8]],
                block_bases=[],
                shape=(M, ),
            )
        else:
            pytest.skip("Linear layout is specialized for 32 elements")

    torch.manual_seed(0)
    x = torch.randint(0, bins, (M, ), dtype=torch.int32, device=device)
    z = torch.zeros((bins, ), dtype=torch.int32, device=device)
    z_torch = torch.histc(x.float(), bins=bins, min=0, max=bins - 1).to(torch.int32)
    kernel[(1, )](x, z, M, bins, src_layout, dst_layout, num_warps=4)
    torch.testing.assert_close(z, z_torch, atol=0, rtol=0)


@pytest.mark.parametrize("M", [64, 128, 256])
@pytest.mark.parametrize("src_layout", _1d_layouts)
@pytest.mark.parametrize("dst_layout", _1d_layouts)
@pytest.mark.parametrize("src_dim", [0, 1])
@pytest.mark.parametrize("dst_dim", [0, 1])
@pytest.mark.parametrize("is_bool", [True, False])
def test_convert1d_layouts(M, src_layout, dst_layout, src_dim, dst_dim, is_bool, device):

    @gluon.jit
    def kernel(x_ptr, y_ptr, M: ttgl.constexpr, src_layout: ttgl.constexpr, dst_layout: ttgl.constexpr,
               src_dim: ttgl.constexpr, dst_dim: ttgl.constexpr):
        offs_src = ttgl.arange(0, M, layout=ttgl.SliceLayout(src_dim, src_layout))
        x = ttgl.load(x_ptr + offs_src)
        y = ttgl.convert_layout(x, layout=ttgl.SliceLayout(dst_dim, dst_layout))
        offs_dst = ttgl.arange(0, M, layout=ttgl.SliceLayout(dst_dim, dst_layout))
        ttgl.store(y_ptr + offs_dst, y)

    torch.manual_seed(17)
    x = torch.randint(0, 4, (M, ), dtype=torch.int32, device=device)
    x = x.to(torch.bool) if is_bool else x
    y = torch.zeros((M, ), dtype=torch.int32, device=device)
    kernel[(1, )](x, y, M, src_layout, dst_layout, src_dim, dst_dim, num_warps=4)
    torch.testing.assert_close(y, x.to(torch.int32))


_2d_layouts = _filter_layouts([
    ttgl.BlockedLayout([1, 1], [THREADS_PER_WARP, 1], [2, 2], [0, 1]),
    ttgl.BlockedLayout([1, 16], [8, THREADS_PER_WARP // 8], [4, 1], [1, 0]),
    ttgl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], ctas_per_cga=[1, 1], cta_split_num=[1, 1],
                                cta_order=[1, 0], instr_shape=[16, 32, 16]),
    ttgl.DotOperandLayout(
        parent=ttgl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], ctas_per_cga=[1, 1],
                                           cta_split_num=[1, 1], cta_order=[1, 0], instr_shape=[16, 32, 16]),
        operand_index=0, k_width=2),
    ttgl.DotOperandLayout(
        parent=ttgl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], ctas_per_cga=[1, 1],
                                           cta_split_num=[1, 1], cta_order=[1, 0], instr_shape=[16, 32, 16]),
        operand_index=0, k_width=1),
    ttgl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[4, 1], ctas_per_cga=[1, 1], cta_split_num=[1, 1],
                                cta_order=[1, 0], instr_shape=[16, 8]),
    ttgl.DotOperandLayout(
        parent=ttgl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[4, 1], ctas_per_cga=[1, 1],
                                           cta_split_num=[1, 1], cta_order=[1, 0], instr_shape=[16, 8]),
        operand_index=1, k_width=2),
    ttgl.DotOperandLayout(
        parent=ttgl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[2, 2], ctas_per_cga=[1, 1],
                                           cta_split_num=[1, 1], cta_order=[1, 0], instr_shape=[16, 8]),
        operand_index=0, k_width=2),
    ttgl.DotOperandLayout(
        parent=ttgl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[4, 1], ctas_per_cga=[1, 1],
                                           cta_split_num=[1, 1], cta_order=[1, 0], instr_shape=[16, 8]),
        operand_index=0, k_width=8),
    ttgl.SliceLayout(
        dim=1, parent=ttgl.DotOperandLayout(
            parent=ttgl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1, 1], ctas_per_cga=[1, 1, 1],
                                               cta_split_num=[1, 1, 1], cta_order=[2, 1, 0], instr_shape=[16, 32, 16]),
            operand_index=0, k_width=2)),
    ttgl.SliceLayout(
        dim=1, parent=ttgl.DotOperandLayout(
            parent=ttgl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[4, 1, 1], ctas_per_cga=[1, 1, 1],
                                               cta_split_num=[1, 1, 1], cta_order=[2, 1, 0], instr_shape=[1, 16, 8]),
            operand_index=1, k_width=2)),
])

_intermediate_layouts = _filter_layouts([
    None,
    ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0, 1]),
    ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0]),
    ttgl.SwizzledSharedLayout(vec=4, per_phase=2, max_phase=4, order=[1, 0]),
    ttgl.SwizzledSharedLayout(vec=2, per_phase=2, max_phase=4, order=[1, 0]),
    ttgl.PaddedSharedLayout(interval_padding_pairs=[[32, 8]], order=[1, 0], ctas_per_cga=[1, 1], cta_split_num=[1, 1],
                            cta_order=[0, 1]),
    ttgl.PaddedSharedLayout(interval_padding_pairs=[[64, 4], [128, 8]], order=[1, 0], ctas_per_cga=[1, 1],
                            cta_split_num=[1, 1], cta_order=[0, 1]),
])


@pytest.mark.parametrize("M, N", [[64, 1], [64, 64], [64, 128], [1, 64]])
@pytest.mark.parametrize("dtype", ["float16"])
@pytest.mark.parametrize("src_layout", _2d_layouts)
@pytest.mark.parametrize("interm_layout", _intermediate_layouts)
@pytest.mark.parametrize("dst_layout", _2d_layouts)
def test_convert2d_layouts(M, N, src_layout, interm_layout, dst_layout, dtype, device):
    if str(src_layout) == str(dst_layout):
        pytest.skip("Source and destination layouts are the same")

    def compute_scratch_buffer_shape(src_layout, dst_layout, shape):

        def compute_rep_shape(layout):
            if type(layout) is ttgl.BlockedLayout:
                warp_shape = torch.tensor(layout.size_per_thread) * torch.tensor(layout.threads_per_warp)
                rep_shape = warp_shape * torch.tensor(layout.warps_per_cta)
                return rep_shape
            else:
                assert False, "TODO: support compute_rep_shape for layout " + str(type(layout))

        src_rep_shape = compute_rep_shape(src_layout)
        dst_rep_shape = compute_rep_shape(dst_layout)
        full_scratch_shape = torch.maximum(src_rep_shape, dst_rep_shape)
        return torch.minimum(full_scratch_shape, torch.tensor(shape))

    if is_hip():
        try:
            scratch_shape = compute_scratch_buffer_shape(src_layout, dst_layout, (M, N))
        except AssertionError:
            pytest.skip("Can't compute scratch buffer size")
        lds_size = get_hip_lds_size()
        # consider int32 dtype in scratch buffer size,
        # because it is the largest dtype used in convert_layout in this test
        int32_size = 4
        # skip even if scratch buffer equal to lds_size, because real scratch buffer is typically larger due to padding
        if scratch_shape[0] * scratch_shape[1] * int32_size >= lds_size:
            pytest.skip("Scratch buffer is too large")

    @gluon.jit
    def kernel(x_ptr, y_ptr, M: ttgl.constexpr, N: ttgl.constexpr, src_layout: ttgl.constexpr,
               dst_layout: ttgl.constexpr, interm_layout: ttgl.constexpr):
        # Create offsets for src layout
        offs_m_src = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, src_layout))[:, None]
        offs_n_src = ttgl.arange(0, N, layout=ttgl.SliceLayout(0, src_layout))[None, :]

        # Load data
        x = ttgl.load(x_ptr + offs_m_src * N + offs_n_src)

        # Convert layout (with or without intermediate shared memory)
        if interm_layout is None:
            y = ttgl.convert_layout(x, layout=dst_layout)
        else:
            # Store to shared memory and load back before converting
            shared_desc = ttgl.allocate_shared_memory(x.dtype, (M, N), interm_layout, value=x)
            x_shared = shared_desc.load(src_layout)
            y = ttgl.convert_layout(x_shared, layout=dst_layout)

        # Create offsets for dst layout and store
        offs_m_dst = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, dst_layout))[:, None]
        offs_n_dst = ttgl.arange(0, N, layout=ttgl.SliceLayout(0, dst_layout))[None, :]
        ttgl.store(y_ptr + offs_m_dst * N + offs_n_dst, y)

    torch.manual_seed(0)
    torch_dtype = getattr(torch, dtype)
    x = torch.randn((M, N), dtype=torch_dtype, device=device)
    y = torch.zeros_like(x)
    kernel[(1, )](x, y, M, N, src_layout, dst_layout, interm_layout)

    torch.testing.assert_close(y, x, rtol=0, atol=0)


# MMA layout pairs for MMA-to-MMA conversion tests
_mma_pairs = [
    # MMA v2.0 layouts
    [
        ttgl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[1, 4], ctas_per_cga=[1, 1], cta_split_num=[1, 1],
                                    cta_order=[0, 1], instr_shape=[16, 8]),
        ttgl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[4, 1], ctas_per_cga=[1, 1], cta_split_num=[1, 1],
                                    cta_order=[0, 1], instr_shape=[16, 8]),
    ],
    [
        ttgl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[2, 8], ctas_per_cga=[1, 1], cta_split_num=[1, 1],
                                    cta_order=[0, 1], instr_shape=[16, 8]),
        ttgl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[8, 2], ctas_per_cga=[1, 1], cta_split_num=[1, 1],
                                    cta_order=[0, 1], instr_shape=[16, 8]),
    ],
    # MMA v2.1 layouts
    [
        ttgl.NVMMADistributedLayout(version=[2, 1], warps_per_cta=[1, 4], ctas_per_cga=[1, 1], cta_split_num=[1, 1],
                                    cta_order=[0, 1], instr_shape=[16, 8]),
        ttgl.NVMMADistributedLayout(version=[2, 1], warps_per_cta=[4, 1], ctas_per_cga=[1, 1], cta_split_num=[1, 1],
                                    cta_order=[0, 1], instr_shape=[16, 8]),
    ],
    [
        ttgl.NVMMADistributedLayout(version=[2, 1], warps_per_cta=[2, 8], ctas_per_cga=[1, 1], cta_split_num=[1, 1],
                                    cta_order=[0, 1], instr_shape=[16, 8]),
        ttgl.NVMMADistributedLayout(version=[2, 1], warps_per_cta=[8, 2], ctas_per_cga=[1, 1], cta_split_num=[1, 1],
                                    cta_order=[0, 1], instr_shape=[16, 8]),
    ],
    # MMA v3.0 layouts
    [
        ttgl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], ctas_per_cga=[1, 1], cta_split_num=[1, 1],
                                    cta_order=[0, 1], instr_shape=[16, 32, 32]),
        ttgl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], ctas_per_cga=[1, 1], cta_split_num=[1, 1],
                                    cta_order=[0, 1], instr_shape=[16, 64, 32]),
    ],
    [
        ttgl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[1, 4], ctas_per_cga=[1, 1], cta_split_num=[1, 1],
                                    cta_order=[0, 1], instr_shape=[16, 32, 32]),
        ttgl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], ctas_per_cga=[1, 1], cta_split_num=[1, 1],
                                    cta_order=[0, 1], instr_shape=[16, 64, 32]),
    ],
    [
        ttgl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[2, 8], ctas_per_cga=[1, 1], cta_split_num=[1, 1],
                                    cta_order=[0, 1], instr_shape=[16, 64, 32]),
        ttgl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[8, 2], ctas_per_cga=[1, 1], cta_split_num=[1, 1],
                                    cta_order=[0, 1], instr_shape=[16, 32, 32]),
    ],
    [
        ttgl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], ctas_per_cga=[1, 1], cta_split_num=[1, 1],
                                    cta_order=[0, 1], instr_shape=[16, 128, 16]),
        ttgl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], ctas_per_cga=[1, 1], cta_split_num=[1, 1],
                                    cta_order=[0, 1], instr_shape=[16, 64, 16]),
    ],
    # AMD MFMA layouts
    [
        ttgl.amd.AMDMFMALayout(version=2, warps_per_cta=[2, 2], tiles_per_warp=[1, 1], instr_shape=[32, 32],
                               transposed=False),
        ttgl.amd.AMDMFMALayout(version=2, warps_per_cta=[4, 1], tiles_per_warp=[1, 1], instr_shape=[32, 32],
                               transposed=False),
    ],
    [
        ttgl.amd.AMDMFMALayout(version=2, warps_per_cta=[4, 1], tiles_per_warp=[1, 1], instr_shape=[32, 32],
                               transposed=False),
        ttgl.amd.AMDMFMALayout(version=2, warps_per_cta=[2, 2], tiles_per_warp=[1, 1], instr_shape=[32, 32],
                               transposed=False),
    ],
    [
        ttgl.amd.AMDMFMALayout(version=2, warps_per_cta=[2, 2], tiles_per_warp=[1, 1], instr_shape=[32, 32],
                               transposed=False),
        ttgl.amd.AMDMFMALayout(version=2, warps_per_cta=[4, 1], tiles_per_warp=[1, 1], instr_shape=[32, 32],
                               transposed=True),
    ],
    [
        ttgl.amd.AMDMFMALayout(version=2, warps_per_cta=[4, 1], tiles_per_warp=[1, 1], instr_shape=[32, 32],
                               transposed=False),
        ttgl.amd.AMDMFMALayout(version=2, warps_per_cta=[2, 2], tiles_per_warp=[1, 1], instr_shape=[32, 32],
                               transposed=True),
    ],
    [
        ttgl.amd.AMDMFMALayout(version=2, warps_per_cta=[4, 4], tiles_per_warp=[1, 1], instr_shape=[16, 16],
                               transposed=False),
        ttgl.amd.AMDMFMALayout(version=2, warps_per_cta=[16, 1], tiles_per_warp=[1, 1], instr_shape=[16, 16],
                               transposed=False),
    ],
    [
        ttgl.amd.AMDMFMALayout(version=2, warps_per_cta=[16, 1], tiles_per_warp=[1, 1], instr_shape=[16, 16],
                               transposed=False),
        ttgl.amd.AMDMFMALayout(version=2, warps_per_cta=[4, 4], tiles_per_warp=[1, 1], instr_shape=[16, 16],
                               transposed=False),
    ],
    [
        ttgl.amd.AMDMFMALayout(version=2, warps_per_cta=[4, 4], tiles_per_warp=[1, 1], instr_shape=[16, 16],
                               transposed=False),
        ttgl.amd.AMDMFMALayout(version=2, warps_per_cta=[16, 1], tiles_per_warp=[1, 1], instr_shape=[16, 16],
                               transposed=True),
    ],
    [
        ttgl.amd.AMDMFMALayout(version=2, warps_per_cta=[16, 1], tiles_per_warp=[1, 1], instr_shape=[16, 16],
                               transposed=False),
        ttgl.amd.AMDMFMALayout(version=2, warps_per_cta=[4, 4], tiles_per_warp=[1, 1], instr_shape=[16, 16],
                               transposed=True),
    ],
    # TODO: AMD WMMA layouts
    #[
    #    WmmaLayout(1, [4, 4]),
    #    WmmaLayout(1, [16, 1]),
    #],
    #[
    #    WmmaLayout(1, [16, 1]),
    #    WmmaLayout(1, [4, 4]),
    #],
    #[
    #    WmmaLayout(2, [4, 4]),
    #    WmmaLayout(2, [16, 1]),
    #],
    #[
    #    WmmaLayout(2, [16, 1]),
    #    WmmaLayout(2, [4, 4]),
    #],
]


@pytest.mark.parametrize("M, N", [[16, 16], [64, 1], [1, 64], [64, 64], [128, 128], [256, 256]])
@pytest.mark.parametrize("dtype", ["float16"])
@pytest.mark.parametrize("mma_pair",
                         [pair for pair in _mma_pairs if all(_is_layout_applicable(layout) for layout in pair)])
def test_convert_mma2mma_layouts(M, N, mma_pair, dtype, device):
    src_layout, dst_layout = mma_pair

    @gluon.jit
    def kernel(x_ptr, y_ptr, M: ttgl.constexpr, N: ttgl.constexpr, src_layout: ttgl.constexpr,
               dst_layout: ttgl.constexpr):
        # Create offsets for src layout
        offs_m_src = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, src_layout))[:, None]
        offs_n_src = ttgl.arange(0, N, layout=ttgl.SliceLayout(0, src_layout))[None, :]

        # Load data and convert layout
        x = ttgl.load(x_ptr + offs_m_src * N + offs_n_src)
        y = ttgl.convert_layout(x, layout=dst_layout)

        # Create offsets for dst layout and store
        offs_m_dst = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, dst_layout))[:, None]
        offs_n_dst = ttgl.arange(0, N, layout=ttgl.SliceLayout(0, dst_layout))[None, :]
        ttgl.store(y_ptr + offs_m_dst * N + offs_n_dst, y)

    torch.manual_seed(0)
    torch_dtype = getattr(torch, dtype)
    x = torch.randn((M, N), dtype=torch_dtype, device=device)

    # Calculate num_warps based on layout
    num_warps = int(torch.prod(torch.tensor(ttgl._layouts.warps_per_cta(src_layout, (M, N)))))
    y = torch.zeros_like(x)
    kernel[(1, )](x, y, M, N, src_layout, dst_layout, num_warps=num_warps)
    torch.testing.assert_close(y, x, rtol=0, atol=0)

    y = torch.zeros_like(x)
    kernel[(1, )](x, y, M, N, dst_layout, src_layout, num_warps=num_warps)
    torch.testing.assert_close(y, x, rtol=0, atol=0)


_warp_local_layouts = _filter_layouts([
    ttgl.BlockedLayout([1, 1], [THREADS_PER_WARP, 1], [1, 1], [1, 0]),
    ttgl.BlockedLayout([1, 1], [THREADS_PER_WARP // 2, 2], [1, 1], [1, 0]),
    ttgl.BlockedLayout([1, 1], [THREADS_PER_WARP // 4, 4], [1, 1], [1, 0]),
    ttgl.BlockedLayout([1, 1], [THREADS_PER_WARP // 8, 8], [1, 1], [1, 0]),
    ttgl.BlockedLayout([1, 1], [THREADS_PER_WARP // 16, 16], [1, 1], [1, 0]),
    ttgl.BlockedLayout([1, 1], [THREADS_PER_WARP // 32, 32], [1, 1], [1, 0]),
    ttgl.BlockedLayout([32, 1], [1, THREADS_PER_WARP], [1, 1], [1, 0]),
    ttgl.BlockedLayout([16, 1], [2, THREADS_PER_WARP // 2], [1, 1], [1, 0]),
    ttgl.BlockedLayout([1, 4], [THREADS_PER_WARP, 1], [1, 1], [1, 0]),
    ttgl.BlockedLayout([1, 4], [THREADS_PER_WARP // 2, 2], [1, 1], [1, 0]),
    ttgl.BlockedLayout([1, 4], [THREADS_PER_WARP // 4, 4], [1, 1], [1, 0]),
    ttgl.BlockedLayout([1, 4], [THREADS_PER_WARP // 8, 8], [1, 1], [1, 0]),
    ttgl.BlockedLayout([1, 4], [THREADS_PER_WARP // 16, 16], [1, 1], [1, 0]),
    ttgl.BlockedLayout([1, 4], [THREADS_PER_WARP // 32, 32], [1, 1], [1, 0]),
])


@pytest.mark.parametrize("M, N", [[32, 32], [64, 64]])
@pytest.mark.parametrize("dtype", ["float16"])
@pytest.mark.parametrize("src_layout", _warp_local_layouts)
@pytest.mark.parametrize("dst_layout", _warp_local_layouts)
def test_convert_warp_local_layouts(M, N, src_layout, dst_layout, dtype, device):
    if str(src_layout) == str(dst_layout):
        pytest.skip("Source and destination layouts are the same")

    # Test layout pairs that are likely to codegen warp shuffles.
    a, b = list(torch.tensor(src_layout.threads_per_warp) // torch.tensor(dst_layout.threads_per_warp))
    c = a if a != 0 else b
    if c > 2:
        pytest.skip("Layout pair too complex for warp-local conversion")

    @gluon.jit
    def kernel(x_ptr, y_ptr, M: ttgl.constexpr, N: ttgl.constexpr, src_layout: ttgl.constexpr,
               dst_layout: ttgl.constexpr):
        # Create offsets for src layout
        offs_m_src = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, src_layout))[:, None]
        offs_n_src = ttgl.arange(0, N, layout=ttgl.SliceLayout(0, src_layout))[None, :]

        # Load data and convert layout
        x = ttgl.load(x_ptr + offs_m_src * N + offs_n_src)
        y = ttgl.convert_layout(x, layout=dst_layout)

        # Create offsets for dst layout and store
        offs_m_dst = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, dst_layout))[:, None]
        offs_n_dst = ttgl.arange(0, N, layout=ttgl.SliceLayout(0, dst_layout))[None, :]
        ttgl.store(y_ptr + offs_m_dst * N + offs_n_dst, y)

    torch.manual_seed(0)
    torch_dtype = getattr(torch, dtype)
    x = torch.randn((M, N), dtype=torch_dtype, device=device)
    y = torch.zeros_like(x)

    num_warps = int(torch.prod(torch.tensor(ttgl._layouts.warps_per_cta(src_layout, (M, N)))))
    kernel[(1, )](x, y, M, N, src_layout, dst_layout, num_warps=num_warps)

    torch.testing.assert_close(y, x, rtol=0, atol=0)


_ld_st_dot_layouts = _filter_layouts([
    ttgl.DotOperandLayout(
        parent=ttgl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[4, 1], ctas_per_cga=[1, 1],
                                           cta_split_num=[1, 1], cta_order=[1, 0], instr_shape=[16, 8]),
        operand_index=0, k_width=4),
    ttgl.DotOperandLayout(
        parent=ttgl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[4, 1], ctas_per_cga=[1, 1],
                                           cta_split_num=[1, 1], cta_order=[0, 1], instr_shape=[16, 8]),
        operand_index=1, k_width=4),
    ttgl.DotOperandLayout(
        parent=ttgl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[4, 1], ctas_per_cga=[1, 1],
                                           cta_split_num=[1, 1], cta_order=[0, 1], instr_shape=[16, 8]),
        operand_index=0, k_width=2),
    ttgl.DotOperandLayout(
        parent=ttgl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[4, 1], ctas_per_cga=[1, 1],
                                           cta_split_num=[1, 1], cta_order=[1, 0], instr_shape=[16, 8]),
        operand_index=1, k_width=2),
])

_ld_st_mma_layouts = _filter_layouts([
    ttgl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[1, 4], ctas_per_cga=[1, 1], cta_split_num=[1, 1],
                                cta_order=[0, 1], instr_shape=[16, 8]),
    ttgl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], ctas_per_cga=[1, 1], cta_split_num=[1, 1],
                                cta_order=[0, 1], instr_shape=[16, 128, 16]),
    ttgl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 2], ctas_per_cga=[1, 1], cta_split_num=[1, 1],
                                cta_order=[0, 1], instr_shape=[16, 128, 16]),
    ttgl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 2], ctas_per_cga=[1, 1], cta_split_num=[1, 1],
                                cta_order=[0, 1], instr_shape=[16, 64, 16]),
    ttgl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[8, 1], ctas_per_cga=[1, 1], cta_split_num=[1, 1],
                                cta_order=[0, 1], instr_shape=[16, 128, 16]),
    ttgl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[8, 4], ctas_per_cga=[1, 1], cta_split_num=[1, 1],
                                cta_order=[0, 1], instr_shape=[16, 64, 16]),
])

_ld_st_shared_layouts = _filter_layouts([
    ttgl.NVMMASharedLayout(swizzle_byte_width=0, transposed=False, element_bitwidth=16, rank=2),
    ttgl.NVMMASharedLayout(swizzle_byte_width=64, transposed=False, element_bitwidth=16, rank=2),
    ttgl.NVMMASharedLayout(swizzle_byte_width=64, transposed=True, element_bitwidth=16, rank=2),
    ttgl.NVMMASharedLayout(swizzle_byte_width=128, transposed=False, element_bitwidth=16, rank=2),
    ttgl.SwizzledSharedLayout(vec=8, per_phase=1, max_phase=1, order=[1, 0]),
    ttgl.SwizzledSharedLayout(vec=4, per_phase=2, max_phase=4, order=[0, 1]),
    ttgl.SwizzledSharedLayout(vec=8, per_phase=1, max_phase=8, order=[1, 0]),
    ttgl.SwizzledSharedLayout(vec=16, per_phase=1, max_phase=16, order=[1, 0]),
])


@pytest.mark.parametrize("shape, dtype", [
    ((16, 32), "float8_e5m2"),
    ((16, 32), "float16"),
    ((16, 32), "float32"),
    ((128, 128), "float16"),
])
@pytest.mark.parametrize("dist_layout", _ld_st_dot_layouts + _ld_st_mma_layouts)
@pytest.mark.parametrize("shared_layout", _ld_st_shared_layouts)
def test_local_load_store_2d_layouts(shape, dtype, dist_layout, shared_layout, device):
    if isinstance(shared_layout, ttgl.NVMMASharedLayout):
        contig_dim = 0 if shared_layout.transposed else 1
        if shape[contig_dim] < (8 * shared_layout.swizzle_byte_width) / shared_layout.element_bitwidth:
            pytest.skip("contig_dim too small for swizzle_byte_width in NVMMASharedLayout")

    # A simple blocked layout
    num_warps = int(torch.prod(torch.tensor(ttgl._layouts.warps_per_cta(dist_layout, shape))))
    blocked_layout = ttgl.BlockedLayout(size_per_thread=[1, 1], threads_per_warp=[4, THREADS_PER_WARP // 4],
                                        warps_per_cta=[1, num_warps], order=[0, 1])

    @gluon.jit
    def kernel(x_ptr, y_ptr, shape_tuple: ttgl.constexpr, src_layout: ttgl.constexpr, dst_layout: ttgl.constexpr,
               shared_layout: ttgl.constexpr):
        M: ttgl.constexpr = shape_tuple[0]
        N: ttgl.constexpr = shape_tuple[1]
        offs_m_src = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, src_layout))[:, None]
        offs_n_src = ttgl.arange(0, N, layout=ttgl.SliceLayout(0, src_layout))[None, :]

        x = ttgl.load(x_ptr + offs_m_src * N + offs_n_src)

        shared_desc = ttgl.allocate_shared_memory(x.dtype, shape_tuple, shared_layout, value=x)
        y = shared_desc.load(dst_layout)

        offs_m_dst = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, dst_layout))[:, None]
        offs_n_dst = ttgl.arange(0, N, layout=ttgl.SliceLayout(0, dst_layout))[None, :]
        ttgl.store(y_ptr + offs_m_dst * N + offs_n_dst, y)

    torch.manual_seed(0)
    torch_dtype = getattr(torch, dtype)

    if "float8" in dtype:
        x = torch.randn(shape, device=device, dtype=torch.float16).to(torch_dtype)
    else:
        x = torch.randn(shape, device=device, dtype=torch_dtype)

    y = torch.zeros_like(x)
    kernel[(1, )](x, y, shape, blocked_layout, dist_layout, shared_layout, num_warps=num_warps)
    torch.testing.assert_close(y, x)

    y = torch.zeros_like(x)
    obj = kernel[(1, )](x, y, shape, dist_layout, blocked_layout, shared_layout, num_warps=num_warps)
    torch.testing.assert_close(y, x)
    if (isinstance(shared_layout, ttgl.NVMMASharedLayout) and dist_layout in _ld_st_mma_layouts
            and dist_layout.version[0] >= 3 and dtype == "float16"):
        assert "stmatrix" in obj.asm["ptx"]


_ld_st_3d_layouts = _filter_layouts([
    ttgl.BlockedLayout([4, 4, 1], [1, 8, THREADS_PER_WARP // 8], [2, 2, 1], [2, 1, 0], [1, 1, 1], [1, 1, 1], [0, 1, 2]),
    ttgl.BlockedLayout([1, 1, 4], [8, THREADS_PER_WARP // 8, 1], [2, 1, 2], [1, 2, 0], [1, 1, 1], [1, 1, 1], [0, 1, 2]),
    ttgl.DotOperandLayout(
        parent=ttgl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[4, 1, 1], ctas_per_cga=[1, 1, 1],
                                           cta_split_num=[1, 1, 1], cta_order=[2, 1, 0], instr_shape=[1, 16, 8]),
        operand_index=0, k_width=1),
])

_ld_st_3d_shared_layouts = _filter_layouts([
    ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[2, 1, 0]),
    ttgl.SwizzledSharedLayout(vec=4, per_phase=2, max_phase=4, order=[1, 2, 0]),
    ttgl.SwizzledSharedLayout(vec=8, per_phase=2, max_phase=4, order=[0, 2, 1]),
    ttgl.SwizzledSharedLayout(vec=4, per_phase=2, max_phase=1, order=[2, 0, 1]),
])


@pytest.mark.parametrize("shape, dtype", [
    ((8, 16, 32), "float32"),
])
@pytest.mark.parametrize("dist_layout", _ld_st_3d_layouts)
@pytest.mark.parametrize("shared_layout", _ld_st_3d_shared_layouts)
def test_local_load_store_3d_layouts(shape, dtype, dist_layout, shared_layout, device):
    # A simple blocked layout
    num_warps = int(torch.prod(torch.tensor(ttgl._layouts.warps_per_cta(dist_layout, shape))))
    blocked_layout = ttgl.BlockedLayout(
        size_per_thread=[1, 1, 1],
        threads_per_warp=[1, 4, THREADS_PER_WARP // 4],
        warps_per_cta=[1, 1, num_warps],
        order=[2, 1, 0],
    )

    @gluon.jit
    def kernel(x_ptr, y_ptr, shape_tuple: ttgl.constexpr, src_layout: ttgl.constexpr, dst_layout: ttgl.constexpr,
               shared_layout: ttgl.constexpr):
        M: ttgl.constexpr = shape_tuple[0]
        N: ttgl.constexpr = shape_tuple[1]
        K: ttgl.constexpr = shape_tuple[2]
        offs_m_src = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, parent=ttgl.SliceLayout(2, src_layout)))[:, None,
                                                                                                           None]
        offs_n_src = ttgl.arange(0, N, layout=ttgl.SliceLayout(0, parent=ttgl.SliceLayout(2, src_layout)))[None, :,
                                                                                                           None]
        offs_k_src = ttgl.arange(0, K, layout=ttgl.SliceLayout(0, parent=ttgl.SliceLayout(1, src_layout)))[None,
                                                                                                           None, :]

        x = ttgl.load(x_ptr + offs_m_src * N * K + offs_n_src * K + offs_k_src)

        shared_desc = ttgl.allocate_shared_memory(x.dtype, shape_tuple, shared_layout, value=x)
        y = shared_desc.load(dst_layout)

        offs_m_dst = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, parent=ttgl.SliceLayout(2, dst_layout)))[:, None,
                                                                                                           None]
        offs_n_dst = ttgl.arange(0, N, layout=ttgl.SliceLayout(0, parent=ttgl.SliceLayout(2, dst_layout)))[None, :,
                                                                                                           None]
        offs_k_dst = ttgl.arange(0, K, layout=ttgl.SliceLayout(0, parent=ttgl.SliceLayout(1, dst_layout)))[None,
                                                                                                           None, :]
        ttgl.store(y_ptr + offs_m_dst * N * K + offs_n_dst * K + offs_k_dst, y)

    torch.manual_seed(0)
    torch_dtype = getattr(torch, dtype)
    x = torch.randn(shape, device=device, dtype=torch_dtype)

    y = torch.zeros_like(x)
    kernel[(1, )](x, y, shape, blocked_layout, dist_layout, shared_layout, num_warps=num_warps)
    torch.testing.assert_close(y, x)

    y = torch.zeros_like(x)
    kernel[(1, )](x, y, shape, dist_layout, blocked_layout, shared_layout, num_warps=num_warps)
    torch.testing.assert_close(y, x)
