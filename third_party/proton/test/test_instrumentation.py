import torch
import pathlib

import triton
import triton.language as tl
import triton.profiler.language as pl
import triton.profiler as proton
import pytest


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def test_record(tmp_path: pathlib.Path):
    if is_hip():
        pytest.skip("HIP backend does not support record")

    @triton.jit
    def add_kernel(
        x_ptr,
        y_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        pl.enter_scope("load0")
        y = tl.load(y_ptr + offsets, mask=mask)
        pl.exit_scope("load0")
        output = x + y
        tl.store(output_ptr + offsets, output, mask=mask)

    torch.manual_seed(0)
    size = 256
    x = torch.rand(size, device='cuda')
    y = torch.rand(size, device='cuda')
    temp_file = tmp_path / "test_hook_instrumentation.hatchet"

    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = (1, 1, 1)
    proton.start(str(temp_file.with_suffix("")), backend="instrumentation")
    pgm = add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    proton.finalize()
    ttir = pgm.asm['ttir']
    assert "proton.record start" in ttir
    assert "proton.record end" in ttir


def test_jit(tmp_path):
    if is_hip():
        pytest.skip("HIP backend does not support record")

    @triton.jit
    def foo(x, size: tl.constexpr, y):
        offs = tl.arange(0, size)
        pl.enter_scope("store0")
        tl.store(y + offs, tl.load(x + offs))
        pl.exit_scope("store0")

    x = torch.tensor([2], device="cuda", dtype=torch.float32)
    y = torch.zeros_like(x)
    temp_file = tmp_path / "test_hook_instrumentation.hatchet"
    proton.start(str(temp_file.with_suffix("")), backend="instrumentation")
    foo[(1, )](x, 1, y, num_warps=4)
    device = triton.runtime.driver.active.get_current_device()
    assert len(foo.device_caches[device][0]) == 1, "Kernel should be cached"
    proton.finalize()
    foo[(1, )](x, 1, y, num_warps=4)
    assert len(foo.device_caches[device][0]) == 2, "Instrumented and uninstrumented kernels both should be cached"
