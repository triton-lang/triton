import torch
import pathlib
import pytest

import triton
import triton.language as tl
import triton.profiler.language as pl
import triton.profiler as proton


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


@pytest.mark.parametrize("mode",
                         ["default", "default:metric_type=cycle", "default:metric_type=cycle:buffer_size=4096", "mma"])
def test_mode_str(mode, tmp_path: pathlib.Path):
    temp_file = tmp_path / "test_mode_str.hatchet"
    proton.start(str(temp_file.with_suffix("")), backend="instrumentation", mode=mode)
    proton.finalize()


@pytest.mark.parametrize("mode", [
    proton.mode.Default(),
    proton.mode.Default(metric_type="cycle"),
    proton.mode.Default(metric_type="cycle", buffer_size=4096),
    proton.mode.MMA()
])
def test_mode_obj(mode, tmp_path: pathlib.Path):
    temp_file = tmp_path / "test_mode_simple.hatchet"
    proton.start(str(temp_file.with_suffix("")), backend="instrumentation", mode=mode)
    proton.finalize()


@pytest.mark.parametrize("method", ["operator", "context_manager"])
def test_record(method, tmp_path: pathlib.Path):
    if is_hip():
        pytest.skip("HIP backend does not support record")

    from contextlib import contextmanager

    @contextmanager
    def instrumentation(file_path):
        proton.hooks.InstrumentationHook.enable_host_buffer = True
        proton.start(str(file_path.with_suffix("")), backend="instrumentation")
        try:
            yield
        finally:
            proton.hooks.InstrumentationHook.enable_host_buffer = False
            proton.finalize()

    @triton.jit
    def add_kernel(
        x_ptr,
        y_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
        METHOD: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        if METHOD == "operator":
            pl.enter_scope("load0")
            y = tl.load(y_ptr + offsets, mask=mask)
            pl.exit_scope("load0")
        else:
            with pl.scope("load0"):
                y = tl.load(y_ptr + offsets, mask=mask)
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
    with instrumentation(temp_file):
        pgm = add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024, METHOD=method)
        #FIXME(fywkevin): have a dedicated place to put those decoding related constants
        payload_offset = int.from_bytes(proton.hooks.InstrumentationHook.host_buffer[12:16].numpy().tobytes(), 'little')
        host_buffer = proton.hooks.InstrumentationHook.host_buffer[payload_offset:]
        preamble = host_buffer[0:4]
        assert int.from_bytes(preamble.numpy().tobytes(), 'little') == 0xdeadbeef
        header_size = 16
        metadata_size = header_size + pgm.metadata.num_warps * 4
        start_tag = host_buffer[metadata_size:metadata_size + 4]
        start_clock = host_buffer[metadata_size + 4:metadata_size + 8]
        end_tag = host_buffer[metadata_size + 8:metadata_size + 12]
        end_clock = host_buffer[metadata_size + 12:metadata_size + 16]
        assert int.from_bytes(start_tag.numpy().tobytes(), 'little') == 0
        assert int.from_bytes(end_tag.numpy().tobytes(), 'little') == 0x80000000
        start_clock_val = int.from_bytes(start_clock.numpy().tobytes(), 'little')
        end_clock_val = int.from_bytes(end_clock.numpy().tobytes(), 'little')
        assert end_clock_val > start_clock_val

    # instrumentation context has finalized, now validate assembly
    ttir = pgm.asm['ttir']
    assert "proton.record start" in ttir
    assert "proton.record end" in ttir


def test_jit(tmp_path):
    if is_hip():
        pytest.skip("HIP backend does not support record")

    @triton.jit
    def foo(x, size: tl.constexpr, y):
        offs = tl.arange(0, size)
        tl.store(y + offs, tl.load(x + offs))

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
