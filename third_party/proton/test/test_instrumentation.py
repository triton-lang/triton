import json
import pathlib

from typing import NamedTuple, Tuple

import pytest
import torch

import triton
import triton.language as tl
import triton.language.semantic
import triton.profiler as proton
import triton.profiler.language as pl
from triton._internal_testing import (
    is_cuda,
    is_hip,
    is_hip_cdna2,
    is_hip_cdna4,
    supports_tma,
    supports_ws,
)
from triton.tools.tensor_descriptor import TensorDescriptor

pl.enable_semantic("triton")

# Skip all tests if the AMD GPU version is not supported
pytestmark = pytest.mark.skipif(is_hip_cdna2(), reason="old AMD GPUs are not supported")

HAS_WARP_SPECIALIZE = supports_ws() and supports_tma()


@pytest.mark.parametrize(
    "mode",
    [
        "default",
        "default:metric_type=cycle",
        "default:metric_type=cycle:buffer_size=4096",
        "mma",
    ],
)
def test_mode_str(mode, tmp_path: pathlib.Path):
    temp_file = tmp_path / "test_mode_str.hatchet"
    proton.start(str(temp_file.with_suffix("")), backend="instrumentation", mode=mode)
    proton.finalize()


@pytest.mark.parametrize(
    "mode",
    [
        proton.mode.Default(),
        proton.mode.Default(metric_type="cycle"),
        proton.mode.Default(metric_type="cycle", buffer_size=4096),
        proton.mode.MMA(),
    ],
)
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
    assert (len(foo.device_caches[device][0]) == 2), "Instrumented and uninstrumented kernels both should be cached"


@pytest.mark.parametrize("method", ["operator", "context_manager"])
def test_record(method, fresh_knobs, tmp_path: pathlib.Path):
    fresh_knobs.compilation.disable_line_info = False

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

    size = 256
    x = torch.rand(size, device="cuda")
    y = torch.rand(size, device="cuda")
    temp_file = tmp_path / "test_record.hatchet"
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = (1, 1, 1)
    with instrumentation(temp_file):
        pgm = add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024, METHOD=method)
        # FIXME(fywkevin): have a dedicated place to put those decoding related constants
        payload_offset = int.from_bytes(
            proton.hooks.InstrumentationHook.host_buffer[12:16].numpy().tobytes(),
            "little",
        )
        host_buffer = proton.hooks.InstrumentationHook.host_buffer[payload_offset:]
        preamble = host_buffer[0:4]
        assert int.from_bytes(preamble.numpy().tobytes(), "little") == 0xDEADBEEF
        header_size = 40
        metadata_size = header_size + pgm.metadata.num_warps * 4
        start_tag = host_buffer[metadata_size:metadata_size + 4]
        start_clock = host_buffer[metadata_size + 4:metadata_size + 8]
        end_tag = host_buffer[metadata_size + 8:metadata_size + 12]
        end_clock = host_buffer[metadata_size + 12:metadata_size + 16]
        assert int.from_bytes(start_tag.numpy().tobytes(), "little") & 0xFFFFF800 == 0
        assert (int.from_bytes(end_tag.numpy().tobytes(), "little") & 0xFFFFF800 == 0x80000000)
        start_clock_val = int.from_bytes(start_tag.numpy().tobytes(), "little") & 0x7FF << 32 | int.from_bytes(
            start_clock.numpy().tobytes(), "little")
        end_clock_val = int.from_bytes(end_tag.numpy().tobytes(), "little") & 0x7FF << 32 | int.from_bytes(
            end_clock.numpy().tobytes(), "little")
        assert end_clock_val > start_clock_val

    # instrumentation context has finalized, now validate assembly
    ttir = pgm.asm["ttir"]
    assert "proton.record start" in ttir
    assert "proton.record end" in ttir

    # check ttir line info
    start_loc = None
    end_loc = None
    for line in ttir.split("\n"):
        if "proton.record start" in line:
            start_loc = line.split("loc(")[1].split(")")[0]
        elif "proton.record end" in line:
            end_loc = line.split("loc(")[1].split(")")[0]
        elif start_loc and f"#loc{start_loc}" in line:
            assert "test_instrumentation.py" in line
        elif end_loc and f"#loc{end_loc}" in line:
            assert "test_instrumentation.py" in line

    assert start_loc is not None and end_loc is not None

    # check llir line info
    llir_lines = pgm.asm["llir"].splitlines()
    clock_instr = "clock" if is_cuda() else "memtime"
    clock_loc = None
    for line in llir_lines:
        if clock_instr not in line or "!dbg" not in line:
            continue
        suffix = line.split("!dbg ")[1]
        clock_loc = suffix.split(",")[0].split()[0]
        break
    assert clock_loc is not None
    loc_line = next(
        (line for line in llir_lines if clock_loc in line and "DILocation" in line),
        None,
    )
    assert loc_line is not None
    assert "line: " in loc_line and "line: 0" not in loc_line


def test_select_ids(tmp_path: pathlib.Path):
    from contextlib import contextmanager

    select_ids = [0, 2]
    mode = proton.mode.Default(
        sampling_strategy="selective",
        sampling_options=",".join(str(i) for i in select_ids),
        granularity="warp",
    )

    @contextmanager
    def instrumentation(file_path):
        proton.hooks.InstrumentationHook.enable_host_buffer = True
        proton.start(
            str(file_path.with_suffix("")),
            backend="instrumentation",
            mode=mode,
        )
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
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        with pl.scope("load_ops"):
            x = tl.load(x_ptr + offsets, mask=mask)
            y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
        tl.store(output_ptr + offsets, output, mask=mask)

    size = 256
    x = torch.rand(size, device="cuda")
    y = torch.rand(size, device="cuda")
    temp_file = tmp_path / "test_select_ids.hatchet"
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = (1, 1, 1)

    warp_indices = []

    with instrumentation(temp_file):
        add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024, num_warps=4)
        uid_num_offset = 36
        uid_vec_offset = 40
        uid_num = int.from_bytes(
            proton.hooks.InstrumentationHook.host_buffer[uid_num_offset:uid_num_offset + 4].numpy().tobytes(),
            "little",
        )
        assert uid_num == len(select_ids)
        for i in range(uid_num):
            offset = uid_vec_offset + i * 4
            warp_id = int.from_bytes(
                proton.hooks.InstrumentationHook.host_buffer[offset:offset + 4].numpy().tobytes(),
                "little",
            )
            warp_indices.append(warp_id)
        assert sorted(warp_indices) == select_ids


@pytest.mark.parametrize("hook", ["triton", None])
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

    size = 256
    x = torch.rand(size, device="cuda")
    y = torch.rand(size, device="cuda")
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

    size = 256
    x = torch.rand(size, device="cuda")
    y = torch.rand(size, device="cuda")
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

    size = 256
    x = torch.rand(size, device="cuda")
    y = torch.rand(size, device="cuda")
    temp_file_inst = tmp_path / "test_tree_inst.hatchet"
    temp_file_driver = tmp_path / "test_tree_driver.hatchet"
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = (1, 1, 1)
    session_id0 = proton.start(str(temp_file_inst.with_suffix("")), backend="instrumentation")
    session_id1 = proton.start(str(temp_file_driver.with_suffix("")))
    proton.deactivate(session_id0)
    proton.deactivate(session_id1)
    proton.activate()
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024, num_warps=1)
    proton.finalize()

    temp_file_restart = tmp_path / "test_tree_restart.hatchet"
    session_id0 = proton.start(str(temp_file_restart.with_suffix("")), backend="instrumentation")
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

    with open(temp_file_restart, "rb") as f:
        data = json.load(f)
        kernel_frame = data[0]["children"][0]
        assert "add_kernel" == kernel_frame["frame"]["name"]
        assert "cycles" in kernel_frame["children"][0]["metrics"]


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

    size = 2048
    x = torch.rand(size, device="cuda")
    y = torch.rand(size, device="cuda")
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = (1, 1, 1)
    temp_file = tmp_path / "test_autotune.hatchet"
    proton.start(str(temp_file.with_suffix("")), backend="instrumentation", hook="triton")
    add_kernel[grid](x, y, output, n_elements)
    proton.finalize()

    # Check all names exist in the output
    with open(temp_file, "rb") as f:
        data = json.load(f)
        names = [frame["frame"]["name"] for frame in data[0]["children"]]
        assert "add_256" in names
        assert "add_512" in names
        assert "add_1024" in names


def test_warp_spec(tmp_path: pathlib.Path):
    if not supports_tma() or not supports_ws():
        pytest.skip("target backend does not support warp specialization and TMA")

    @triton.jit
    def matmul_kernel_tma(a_desc, b_desc, c_desc,  #
                          M, N, K,  #
                          BLOCK_SIZE_M: tl.constexpr,  #
                          BLOCK_SIZE_N: tl.constexpr,  #
                          BLOCK_SIZE_K: tl.constexpr,  #
                          GROUP_SIZE_M: tl.constexpr,  #
                          FP8_OUTPUT: tl.constexpr,  #
                          WARP_SPECIALIZE: tl.constexpr,  #
                          ):
        dtype = tl.float8e4nv if FP8_OUTPUT else tl.float16
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        k_tiles = tl.cdiv(K, BLOCK_SIZE_K)

        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for k in tl.range(k_tiles, warp_specialize=WARP_SPECIALIZE):
            pl.enter_scope("loop")
            offs_k = k * BLOCK_SIZE_K
            a = a_desc.load([offs_am, offs_k])
            b = b_desc.load([offs_bn, offs_k])
            accumulator = tl.dot(a, b.T, accumulator)
            pl.exit_scope("loop")

        c = accumulator.to(dtype)

        offs_cm = pid_m * BLOCK_SIZE_M
        offs_cn = pid_n * BLOCK_SIZE_N
        c_desc.store([offs_cm, offs_cn], c)

    def matmul_tma(a, b, warp_specialize: bool):
        # Check constraints.
        assert a.shape[1] == b.shape[1], "Incompatible dimensions"  # b is transposed
        assert a.dtype == b.dtype, "Incompatible dtypes"

        M, K = a.shape
        N, K = b.shape
        dtype = a.dtype

        c = torch.empty((M, N), device=a.device, dtype=dtype)

        a_desc = TensorDescriptor(a, a.shape, a.stride(), [128, 128])
        b_desc = TensorDescriptor(b, b.shape, b.stride(), [256, 128])
        c_desc = TensorDescriptor(c, c.shape, c.stride(), [128, 256])

        def grid(META):
            BLOCK_M = 128
            BLOCK_N = 256
            return (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), )

        matmul_kernel_tma[grid](
            a_desc,
            b_desc,
            c_desc,  #
            M,
            N,
            K,  #
            BLOCK_SIZE_M=128,  #
            BLOCK_SIZE_N=256,  #
            BLOCK_SIZE_K=128,  #
            GROUP_SIZE_M=8,  #
            FP8_OUTPUT=dtype == torch.float8_e4m3fn,  #
            WARP_SPECIALIZE=warp_specialize,  #
            num_stages=2,  #
            num_warps=8,
        )
        return c

    mode = proton.mode.Default(metric_type="cycle", optimizations="clock32")
    temp_file = tmp_path / "test_warpspec.hatchet"
    proton.start(str(temp_file.with_suffix("")), backend="instrumentation", mode=mode)
    M, N, K = 512, 512, 512
    a = torch.randn((M, K), device="cuda", dtype=torch.float16).to(torch.float8_e4m3fn)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16).to(torch.float8_e4m3fn)
    b = b.T.contiguous()

    matmul_tma(a, b, warp_specialize=HAS_WARP_SPECIALIZE)
    proton.finalize()

    with open(temp_file, "rb") as f:
        data = json.load(f)
        kernel = data[0]["children"][0]
        assert kernel["children"][0]["frame"]["name"] == "loop"
        assert kernel["children"][0]["metrics"]["cycles"] > 0
        assert kernel["frame"]["name"] == "matmul_kernel_tma"


def test_timeline(tmp_path: pathlib.Path):
    temp_file = tmp_path / "test_timeline.chrome_trace"
    mode = proton.mode.Default(metric_type="cycle", optimizations="time_shift")
    proton.start(
        str(temp_file.with_suffix("")),
        data="trace",
        backend="instrumentation",
        mode=mode,
    )

    @triton.jit
    def foo(x, y, size: tl.constexpr):
        pl.enter_scope("entire")
        offs = tl.arange(0, size)
        pl.enter_scope("load")
        x = tl.load(x + offs)
        x = x + 1
        pl.exit_scope("load")
        pl.enter_scope("store")
        tl.store(y + offs, x)
        pl.exit_scope("store")
        pl.exit_scope("entire")

    with proton.scope("init"):
        x = torch.ones((1024, ), device="cuda", dtype=torch.float32)
        y = torch.zeros_like(x)

    with proton.scope("test"):
        foo[(1, )](x, y, x.size()[0], num_warps=4)

    proton.finalize()

    with temp_file.open() as f:
        data = json.load(f)
        trace_events = data["traceEvents"]
        assert len(trace_events) == 12
        assert trace_events[-1]["tid"][0:4] == "warp"
        assert trace_events[-1]["args"]["call_stack"][-1] == "foo"
        assert trace_events[-1]["args"]["call_stack"][-2] == "test"


@pytest.mark.skipif(is_hip_cdna4(), reason="nondeterministic failure")
def test_globaltime(tmp_path: pathlib.Path):
    temp_file = tmp_path / "test_globaltime.chrome_trace"
    mode = proton.mode.Default(
        metric_type="cycle",
        optimizations="clock32,time_shift",
        sampling_strategy="selective",
        sampling_options="0",
    )
    proton.start(
        str(temp_file.with_suffix("")),
        data="trace",
        backend="instrumentation",
        mode=mode,
    )

    @triton.jit()
    def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pl.enter_scope("elementwise_add_kernel")
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
        tl.store(output_ptr + offsets, output, mask=mask)
        pl.exit_scope("elementwise_add_kernel")

    size = 1024 * 2000
    x = torch.rand(size, device="cuda")
    y = torch.rand(size, device="cuda")
    output = torch.empty_like(x)
    n_elements = output.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, BLOCK_SIZE), )
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE, num_warps=16)
    proton.finalize()

    with temp_file.open() as f:
        data = json.load(f)
        trace_events = data["traceEvents"]
        target = sorted(
            [event for event in trace_events if "Core0 " in event["pid"]],
            key=lambda x: x["ts"],
        )
        s = len(target)
        assert s > 1
        ts_diff = target[s - 1]["ts"] - target[0]["ts"]
        assert ts_diff >= target[0]["dur"]


@pytest.mark.skipif(is_hip(), reason="not stable overhead numbers on AMD GPUs")
def test_overhead(tmp_path: pathlib.Path):
    temp_file_cycles = tmp_path / "test_overhead.hatchet"
    temp_file_time = tmp_path / "test_overhead_time.hatchet"

    @triton.jit()
    def kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr, LOOP: tl.constexpr):
        pl.enter_scope("kernel")
        for _ in range(16):
            if LOOP:
                pl.enter_scope("loop")
            x = tl.load(x_ptr + tl.arange(0, BLOCK_SIZE))
            tl.store(y_ptr + tl.arange(0, BLOCK_SIZE), x + 1)
            if LOOP:
                pl.exit_scope("loop")
        pl.exit_scope("kernel")

    BLOCK_SIZE = 256
    x = torch.zeros(BLOCK_SIZE, device="cuda", dtype=torch.float32)
    y = torch.zeros_like(x)

    def bench():
        with proton.scope("single"):
            kernel[(1024, )](x, y, BLOCK_SIZE, False)
        with proton.scope("loop"):
            kernel[(1024, )](x, y, BLOCK_SIZE, True)

    # warmup
    bench()

    proton.start(str(temp_file_time.with_suffix("")), )

    with proton.scope("session0"):
        bench()

    proton.start(str(temp_file_cycles.with_suffix("")), backend="instrumentation",
                 mode=proton.mode.Default(metric_type="cycle", buffer_size=4096))

    with proton.scope("session1"):
        bench()
    proton.finalize()

    with temp_file_time.open("rb") as f:
        data = json.load(f)
    root = data[0]

    def session_kernel_time(session_name: str) -> Tuple[int, int]:
        session_node = next(child for child in root["children"] if child["frame"]["name"] == session_name)
        single_node = next(child for child in session_node["children"] if child["frame"]["name"] == "single")
        loop_node = next(child for child in session_node["children"] if child["frame"]["name"] == "loop")
        kernel_node = single_node["children"][0]
        single_time = kernel_node["metrics"]["time (ns)"]
        kernel_node = loop_node["children"][0]
        loop_time = kernel_node["metrics"]["time (ns)"]
        return single_time, loop_time

    session0_single_time, session0_loop_time = session_kernel_time("session0")
    session1_single_time, session1_loop_time = session_kernel_time("session1")
    single_threshold = 1.2 if is_cuda() else 1.5
    loop_threshold = 2.0 if is_cuda() else 3.0
    assert session1_single_time / session0_single_time < single_threshold, "Simple kernel overhead too high"
    assert session1_loop_time / session0_loop_time < loop_threshold, "Loop kernel overhead too high"


def test_gmem_buffer(tmp_path: pathlib.Path):

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

    size = 512
    x = torch.rand(size, device="cuda")
    y = torch.rand(size, device="cuda")
    temp_file = tmp_path / "test_gmem_buffer.chrome_trace"
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = (1, 1, 1)
    mode = proton.mode.Default(buffer_type="global")
    proton.start(
        str(temp_file.with_suffix("")),
        backend="instrumentation",
        data="trace",
        mode=mode,
    )
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024, num_warps=2)
    proton.finalize()

    with open(temp_file, "rb") as f:
        data = json.load(f)
        events = data["traceEvents"]

        # Assert we have exactly 4 events (2 warps × 2 scopes)
        assert len(events) == 4

        # Assert all events have the expected common fields
        for event in events:
            assert "ts" in event
            assert "dur" in event
            assert event["dur"] > 0

        # Assert we have 2 kernel events and 2 load_ops events
        kernel_events = [e for e in events if e["name"] == "kernel"]
        load_ops_events = [e for e in events if e["name"] == "load_ops"]
        assert len(kernel_events) == 2
        assert len(load_ops_events) == 2

        # Assert we have events from both warps
        warp0_events = [e for e in events if "warp 0" in e["tid"]]
        warp1_events = [e for e in events if "warp 1" in e["tid"]]
        assert len(warp0_events) == 2
        assert len(warp1_events) == 2


def test_event_args(tmp_path: pathlib.Path):

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
            x = tl.load(x_ptr + offsets, mask=mask)
            y = tl.load(y_ptr + offsets, mask=mask)
            output = x + y
            tl.store(output_ptr + offsets, output, mask=mask)

    size = 256
    x = torch.rand(size, device="cuda")
    y = torch.rand(size, device="cuda")
    temp_file = tmp_path / "test_block_metadata.chrome_trace"
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = (1, 1, 1)
    proton.start(str(temp_file.with_suffix("")), backend="instrumentation", data="trace")
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024, num_warps=2)
    proton.finalize()

    with open(temp_file, "rb") as f:
        data = json.load(f)
        events = data["traceEvents"]

        # Verify we have events
        assert len(events) > 0

        # Verify each event has the required metadata in args
        for event in events:
            assert "args" in event
            args = event["args"]

            assert "Init Time (ns)" in args
            assert "Post Final Time (ns)" in args
            assert "Finalization Time (ns)" in args

            # Verify timing values are reasonable
            init_time = args["Init Time (ns)"]
            post_final_time = args["Post Final Time (ns)"]
            finalization_time = args["Finalization Time (ns)"]

            assert init_time >= 0
            assert post_final_time >= 0
            assert finalization_time >= 0


def test_threaded_kernel_call(tmp_path: pathlib.Path):

    import threading

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
            x = tl.load(x_ptr + offsets, mask=mask)
            y = tl.load(y_ptr + offsets, mask=mask)
            output = x + y
            tl.store(output_ptr + offsets, output, mask=mask)

    size = 256
    x = torch.rand(size, device="cuda")
    y = torch.rand(size, device="cuda")
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = (1, 1, 1)

    temp_file = tmp_path / "test_threaded.chrome_trace"
    proton.start(
        str(temp_file.with_suffix("")),
        backend="instrumentation",
        data="trace",
    )

    exception_holder = []

    def run_kernel():
        try:
            add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
        except Exception as e:
            exception_holder.append(e)

    thread = threading.Thread(target=run_kernel)
    thread.start()
    thread.join()

    proton.finalize()

    assert len(exception_holder) == 0, f"Kernel raised exception: {exception_holder[0] if exception_holder else None}"

    with open(temp_file, "rb") as f:
        data = json.load(f)
        events = data["traceEvents"]
        assert len(events) > 0
        kernel_events = [e for e in events if e["name"] == "kernel"]
        assert len(kernel_events) > 0
