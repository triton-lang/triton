import json
import statistics

import numpy as np
import pytest
import torch
from triton._internal_testing import is_blackwell
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    TensorMemoryScalesLayout,
    allocate_tensor_memory,
    mbarrier,
    tcgen05_commit,
    tcgen05_copy,
)


@gluon.jit
def _copy_scales(smem, tmem, bar):
    tcgen05_copy(smem, tmem)
    tcgen05_commit(bar)
    mbarrier.wait(bar, phase=0)


@gluon.jit
def _idle():
    pass


@gluon.jit
def _scale_copy(x, out, ROWS: gl.constexpr, SHARED: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [32, 1], [4, 1], [1, 0])
    rows = gl.arange(0, ROWS, gl.SliceLayout(1, layout))
    cols = gl.arange(0, 16, gl.SliceLayout(0, layout))
    value = gl.load(x + rows[:, None] * 16 + cols[None, :])
    smem = gl.allocate_shared_memory(gl.float8e4nv, (ROWS, 16), SHARED, value)
    tmem = allocate_tensor_memory(gl.float8e4nv, (ROWS, 16), TensorMemoryScalesLayout())
    physical = tmem.reinterpret(shape=(128, ROWS // 2), layout=TensorMemoryLayout((128, ROWS // 2), col_stride=1))
    bar = mbarrier.allocate_mbarrier()
    mbarrier.init(bar, count=1)
    gl.warp_specialize([(_idle, ()), (_copy_scales, (smem, tmem, bar))], [1])
    mbarrier.invalidate(bar)
    out_layout: gl.constexpr = physical.get_reg_layout()
    out_rows = gl.arange(0, 128, gl.SliceLayout(1, out_layout))
    out_cols = gl.arange(0, ROWS // 2, gl.SliceLayout(0, out_layout))
    gl.store(out + out_rows[:, None] * (ROWS // 2) + out_cols[None, :], physical.load())


@gluon.jit
def _repeated_scale_copy(x, out, ROWS: gl.constexpr, COPIES: gl.constexpr, SHARED: gl.constexpr):
    # Distinct inputs and observable outputs keep every unrolled copy live.
    for i in gl.static_range(COPIES):
        _scale_copy(x + i * ROWS * 16, out + i * ROWS * 64, ROWS, SHARED)


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell tensor memory")
@pytest.mark.parametrize("rows", [128, 256])
@pytest.mark.parametrize("copies", [1, 2, 4])
def test_fpsan_scale_copy_performance(rows, copies, fresh_knobs):
    raw = np.random.RandomState(0).randint(1, 64, (copies, rows, 16), dtype=np.uint8)
    x = torch.tensor(raw, device="cuda").view(torch.float8_e4m3fn)
    out = torch.empty((copies, 128, rows // 2), device="cuda", dtype=torch.float8_e4m3fn)
    # The scale-copy physical representation repeats each 32-row group four times.
    expected = raw.reshape(copies, rows // 32, 32, 4, 4).transpose(0, 2, 3, 1, 4).reshape(copies, 32, -1)
    expected = torch.tensor(np.tile(expected, (1, 4, 1)), device="cuda")
    bases = [[0, 1], [0, 2], [32, 0], [64, 0], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0]]
    bases += [[1 << bit, 0] for bit in range(7, (rows - 1).bit_length())]
    bases += [[0, 4], [0, 8]]
    shared = gl.SharedLinearLayout(bases, [])

    def launch():
        return _repeated_scale_copy[(1, )](x, out, rows, copies, shared, num_warps=4)

    # Check the ordinary hardware operation too, outside the compilation timing.
    fresh_knobs.compilation.instrumentation_mode = ""
    launch()
    torch.testing.assert_close(out.view(torch.uint8), expected, rtol=0, atol=0)

    compilations = []

    def listener(*, src, metadata, metadata_group, times, cache_hit):
        if getattr(src, "fn", None) is _repeated_scale_copy:
            assert not cache_hit
            compilations.append(times)

    fresh_knobs.compilation.instrumentation_mode = "fpsan"
    fresh_knobs.compilation.listener = listener
    fresh_knobs.compilation.always_compile = True
    ptx_bytes = []
    for _ in range(3):
        _repeated_scale_copy.device_caches.clear()
        compiled = launch()
        ptx_bytes.append(len(compiled.asm["ptx"].encode()))
        torch.testing.assert_close(out.view(torch.uint8), expected, rtol=0, atol=0)
    assert len(compilations) == 3
    seconds = [times.total / 1_000_000 for times in compilations]
    stages = [{name: value / 1_000_000 for name, value in times.lowering_stages} for times in compilations]
    print("FPSAN_SCALE_COPY " + json.dumps(
        dict(rows=rows, copies=copies, seconds=seconds, median_seconds=statistics.median(seconds), stages=stages,
             ptx_bytes=ptx_bytes)))
