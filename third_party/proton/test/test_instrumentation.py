import torch
import pathlib
import pytest
import json

import triton
import triton.language as tl
import triton.language.semantic
import triton.profiler.language as pl
import triton.profiler as proton
from triton.tools.tensor_descriptor import TensorDescriptor

from typing import NamedTuple

pl.enable_semantic("triton")


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def supports_tma():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


def supports_ws():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


HAS_TENSOR_DESC = supports_tma() and hasattr(tl, "make_tensor_descriptor")
HAS_HOST_TENSOR_DESC = supports_tma() and hasattr(triton.tools.tensor_descriptor, "TensorDescriptor")
HAS_WARP_SPECIALIZE = supports_ws() and HAS_TENSOR_DESC


@pytest.mark.parametrize("mode",
                         ["default", "default:metric_type=cycle", "default:metric_type=cycle:buffer_size=4096", "mma"])
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
    x = torch.rand(size, device="cuda")
    y = torch.rand(size, device="cuda")
    temp_file = tmp_path / "test_record.hatchet"
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = (1, 1, 1)
    with instrumentation(temp_file):
        pgm = add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024, METHOD=method)
        # FIXME(fywkevin): have a dedicated place to put those decoding related constants
        payload_offset = int.from_bytes(proton.hooks.InstrumentationHook.host_buffer[12:16].numpy().tobytes(), "little")
        host_buffer = proton.hooks.InstrumentationHook.host_buffer[payload_offset:]
        preamble = host_buffer[0:4]
        assert int.from_bytes(preamble.numpy().tobytes(), "little") == 0xDEADBEEF
        header_size = 16
        metadata_size = header_size + pgm.metadata.num_warps * 4
        start_tag = host_buffer[metadata_size:metadata_size + 4]
        start_clock = host_buffer[metadata_size + 4:metadata_size + 8]
        end_tag = host_buffer[metadata_size + 8:metadata_size + 12]
        end_clock = host_buffer[metadata_size + 12:metadata_size + 16]
        assert int.from_bytes(start_tag.numpy().tobytes(), "little") & 0xFFFFF800 == 0
        assert int.from_bytes(end_tag.numpy().tobytes(), "little") & 0xFFFFF800 == 0x80000000
        start_clock_val = int.from_bytes(start_tag.numpy().tobytes(), "little") & 0x7FF << 32 | int.from_bytes(
            start_clock.numpy().tobytes(), "little")
        end_clock_val = int.from_bytes(end_tag.numpy().tobytes(), "little") & 0x7FF << 32 | int.from_bytes(
            end_clock.numpy().tobytes(), "little")
        assert end_clock_val > start_clock_val

    # instrumentation context has finalized, now validate assembly
    ttir = pgm.asm["ttir"]
    assert "proton.record start" in ttir
    assert "proton.record end" in ttir


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

    torch.manual_seed(0)
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

    torch.manual_seed(0)
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

    torch.manual_seed(0)
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


def test_sched_barrier(tmp_path: pathlib.Path):
    if is_cuda():
        pytest.skip("CUDA backend does not support instruction scheduling barriers")

    @triton.jit
    def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak,  #
                      stride_bk, stride_bn,  #
                      stride_cm, stride_cn, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                      BLOCK_SIZE_K: tl.constexpr,  #
                      GROUP_SIZE_M: tl.constexpr,  #
                      ):
        pl.enter_scope("warpgroup_1")
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        pl.exit_scope("warpgroup_1")
        pl.enter_scope("warpgroup_2")
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
            accumulator = tl.dot(a, b, accumulator)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk
        pl.exit_scope("warpgroup_2")

        pl.enter_scope("warpgroup_3")
        c = accumulator.to(tl.float16)
        pl.exit_scope("warpgroup_3")

        pl.enter_scope("warpgroup_4")
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)
        pl.exit_scope("warpgroup_4")

    torch.manual_seed(0)
    a = torch.randn((512, 512), device="cuda", dtype=torch.float16)
    b = torch.randn((512, 512), device="cuda", dtype=torch.float16)

    M, K = a.shape
    K, N = b.shape

    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 128, 256, 64
    GROUP_SIZE_M = 8

    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = lambda META: (triton.cdiv(M, 128) * triton.cdiv(N, 256), )

    temp_file = tmp_path / "test_sched_barrier.hatchet"
    mode = proton.mode.Default(metric_type="cycle", optimizations="sched_barriers")
    proton.start(str(temp_file.with_suffix("")), backend="instrumentation", mode=mode)

    grid = lambda META: (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N), )
    kernel = matmul_kernel[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M)
    proton.finalize()

    asm = kernel.asm["amdgcn"]

    # Make sure a sched barrier is inserted before every s_memtime call
    lines = asm.splitlines()
    for i, line in enumerate(lines):
        if "s_memtime" in line:
            if ".loc" in lines[i - 1]:
                assert "sched_barrier" in lines[i - 2]
            else:
                assert "sched_barrier" in lines[i - 1]


def test_warp_spec(tmp_path: pathlib.Path):
    if not HAS_WARP_SPECIALIZE:
        pytest.skip("target backend does not support warp specialization")

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
        pl.enter_scope("kernel")
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
        pl.exit_scope("kernel")

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
            a_desc, b_desc, c_desc,  #
            M, N, K,  #
            BLOCK_SIZE_M=128,  #
            BLOCK_SIZE_N=256,  #
            BLOCK_SIZE_K=128,  #
            GROUP_SIZE_M=8,  #
            FP8_OUTPUT=dtype == torch.float8_e4m3fn,  #
            WARP_SPECIALIZE=warp_specialize,  #
            num_stages=2,  #
            num_warps=8)
        return c

    mode = proton.mode.Default(metric_type="cycle", optimizations="clock32")
    temp_file = tmp_path / "test_warpspec.hatchet"
    proton.start(str(temp_file.with_suffix("")), backend="instrumentation", mode=mode)
    torch.manual_seed(0)
    M, N, K = 512, 512, 512
    a = torch.randn((M, K), device="cuda", dtype=torch.float16).to(torch.float8_e4m3fn)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16).to(torch.float8_e4m3fn)
    b = b.T.contiguous()

    matmul_tma(a, b, warp_specialize=HAS_WARP_SPECIALIZE)
    proton.finalize()

    with open(temp_file, "rb") as f:
        data = json.load(f)
        kernel_level = data[0]["children"][0]["children"][0]
        assert kernel_level["children"][0]["frame"]["name"] == 'loop'
        assert kernel_level["children"][0]["metrics"]['cycles'] > 0
        assert kernel_level["frame"]["name"] == "kernel"
        assert kernel_level["metrics"]["cycles"] > 0


def test_timeline(tmp_path: pathlib.Path):
    temp_file = tmp_path / "test_timeline.chrome_trace"
    mode = proton.mode.Default(metric_type="cycle", optimizations="time_shift")
    proton.start(str(temp_file.with_suffix("")), data="trace", backend="instrumentation", mode=mode)

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
