import torch
import pathlib
import pytest
import json

import triton
import triton.language as tl
import triton.profiler.language as pl
import triton.profiler as proton

from typing import NamedTuple


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


def test_jit(tmp_path):

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


@pytest.mark.parametrize("method", ["operator", "context_manager"])
def test_record(method, tmp_path: pathlib.Path):

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
    temp_file = tmp_path / "test_record.hatchet"
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


@pytest.mark.parametrize("hook", ["launch", None])
def test_tree(tmp_path: pathlib.Path, hook):

    def metadata_fn(grid: tuple, metadata: NamedTuple, args: dict):
        BLOCK_SIZE = args["BLOCK_SIZE"]
        return {"name": f"add_{BLOCK_SIZE}"}

    @triton.jit(launch_metadata=metadata_fn)
    def add_kernel(
        x_ptr,
        y_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        with pl.scope("kernel"):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            with pl.scope("load_ops"):
                with pl.scope("load_x"):
                    x = tl.load(x_ptr + offsets, mask=mask)
                with pl.scope("load_y"):
                    y = tl.load(y_ptr + offsets, mask=mask)
            output = x + y
            tl.store(output_ptr + offsets, output, mask=mask)

    torch.manual_seed(0)
    size = 256
    x = torch.rand(size, device='cuda')
    y = torch.rand(size, device='cuda')
    temp_file = tmp_path / "test_tree.hatchet"
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = (1, 1, 1)
    proton.start(str(temp_file.with_suffix("")), backend="instrumentation", hook=hook)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024, num_warps=1)
    proton.finalize()

    with open(temp_file, "rb") as f:
        data = json.load(f)
        if hook:
            assert "add_1024" == data[0]["children"][0]["frame"]["name"]
        kernel_frame = data[0]["children"][0]["children"][0]
        load_ops = kernel_frame["children"][0]
        assert "load_ops" in load_ops["frame"]["name"]
        assert ("load_x" in load_ops["children"][0]["frame"]["name"]
                or "load_x" in load_ops["children"][1]["frame"]["name"])
        assert ("load_y" in load_ops["children"][0]["frame"]["name"]
                or "load_y" in load_ops["children"][1]["frame"]["name"])
        assert load_ops["children"][0]["metrics"]["cycles"] > 0
        assert load_ops["children"][0]["metrics"]["normalized_cycles"] > 0
        assert load_ops["children"][1]["metrics"]["cycles"] > 0
        assert load_ops["children"][1]["metrics"]["normalized_cycles"] > 0


def test_trace(tmp_path: pathlib.Path):

    @triton.jit
    def add_kernel(
        x_ptr,
        y_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        with pl.scope("kernel"):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            with pl.scope("load_ops"):
                x = tl.load(x_ptr + offsets, mask=mask)
                y = tl.load(y_ptr + offsets, mask=mask)
            output = x + y
            tl.store(output_ptr + offsets, output, mask=mask)

    @triton.jit
    def sub_kernel(
        x_ptr,
        y_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        with pl.scope("kernel"):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            with pl.scope("load_ops"):
                x = tl.load(x_ptr + offsets, mask=mask)
                y = tl.load(y_ptr + offsets, mask=mask)
            output = x - y
            tl.store(output_ptr + offsets, output, mask=mask)

    torch.manual_seed(0)
    size = 256
    x = torch.rand(size, device='cuda')
    y = torch.rand(size, device='cuda')
    temp_file = tmp_path / "test_trace.chrome_trace"
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = (1, 1, 1)
    proton.start(str(temp_file.with_suffix("")), backend="instrumentation", data="trace")
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024, num_warps=1)
    sub_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024, num_warps=1)
    proton.finalize()

    with open(temp_file, "rb") as f:
        data = json.load(f)
        events = data["traceEvents"]
        assert events[0]["name"] == "kernel"
        assert events[0]["cat"] == "add_kernel"
        assert events[1]["name"] == "load_ops"
        assert events[1]["cat"] == "add_kernel"
        assert events[2]["name"] == "kernel"
        assert events[2]["cat"] == "sub_kernel"
        assert events[3]["name"] == "load_ops"
        assert events[3]["cat"] == "sub_kernel"


def test_multi_session(tmp_path: pathlib.Path):

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
        with pl.scope("load_x"):
            x = tl.load(x_ptr + offsets, mask=mask)
        with pl.scope("load_y"):
            y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
        tl.store(output_ptr + offsets, output, mask=mask)

    torch.manual_seed(0)
    size = 256
    x = torch.rand(size, device='cuda')
    y = torch.rand(size, device='cuda')
    temp_file_inst = tmp_path / "test_tree_inst.hatchet"
    temp_file_driver = tmp_path / "test_tree_driver.hatchet"
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = (1, 1, 1)
    proton.start(str(temp_file_inst.with_suffix("")), backend="instrumentation")
    proton.start(str(temp_file_driver.with_suffix("")))
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024, num_warps=1)
    proton.finalize()

    with open(temp_file_inst, "rb") as f:
        data = json.load(f)
        kernel_frame = data[0]["children"][0]
        assert "add_kernel" == kernel_frame["frame"]["name"]
        assert "cycles" in kernel_frame["children"][0]["metrics"]

    with open(temp_file_driver, "rb") as f:
        data = json.load(f)
        kernel_frame = data[0]["children"][0]
        assert "add_kernel" == kernel_frame["frame"]["name"]
        assert "time (ns)" in kernel_frame["metrics"]


def test_autotune(tmp_path: pathlib.Path):

    def metadata_fn(
        grid: tuple,
        metadata: NamedTuple,
        args: dict,
    ):
        BLOCK_SIZE = args["BLOCK_SIZE"]
        return {
            "name": f"add_{BLOCK_SIZE}",
        }

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 256}, num_warps=1),
            triton.Config({"BLOCK_SIZE": 512}, num_warps=1),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=1),
        ],
        key=["n_elements"],
    )
    @triton.jit(launch_metadata=metadata_fn)
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
        with pl.scope("load_x"):
            x = tl.load(x_ptr + offsets, mask=mask)
        with pl.scope("load_y"):
            y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
        tl.store(output_ptr + offsets, output, mask=mask)

    torch.manual_seed(0)
    size = 2048
    x = torch.rand(size, device='cuda')
    y = torch.rand(size, device='cuda')
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = (1, 1, 1)
    temp_file = tmp_path / "test_autotune.hatchet"
    proton.start(str(temp_file.with_suffix("")), backend="instrumentation", hook="launch")
    add_kernel[grid](x, y, output, n_elements)
    proton.finalize()

    # Check all names exist in the output
    with open(temp_file, "rb") as f:
        data = json.load(f)
        names = [frame["frame"]["name"] for frame in data[0]["children"]]
        assert "add_256" in names
        assert "add_512" in names
        assert "add_1024" in names
