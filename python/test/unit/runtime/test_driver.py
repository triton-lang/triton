import os
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from multiprocessing.connection import Client
from types import SimpleNamespace

import cloudpickle
import pytest
import torch

import triton
import triton.language as tl
from triton._compile_warmup import (
    CompilationTrace,
    _require_complete_warmup,
    compile_warmup_only,
    pytest_collection_modifyitems,
    pytest_xdist_setupnodes,
    summarize_compile_trace,
)
from triton._compile_warmup_pool import ProcessPoolWarmupDispatcher, SharedWarmupCoordinator, _jit_dumps
from triton._internal_testing import is_compile_warmup, random_float, random_int
from triton import _test_runner
from triton.backends.driver import GPUDriver, expand_signature, wrap_handle_tensordesc_impl
from triton.backends.nvidia.compiler import CUDABackend
from triton.tools.mxfp import MXFP4Tensor, MXScaleTensor


def test_compile_warmup_only_intercepts_launches():
    launches = []

    class Kernel(triton.KernelInterface):

        def warmup(self, *args, grid, **kwargs):
            launches.append((args, grid, kwargs))
            return "compiled"

    kernel = Kernel()
    previous_getitem = triton.KernelInterface.__getitem__
    previous_assert_close = torch.testing.assert_close
    with compile_warmup_only():
        tensor = torch.empty(16, device="cuda")
        view = tensor[1:]
        result = kernel[(2, )](tensor, BLOCK_SIZE=16)
        torch.testing.assert_close(tensor, tensor)
        assert torch.allclose(tensor, tensor)
        assert (tensor == tensor).all()
        assert triton.testing.cublas() is None
        assert is_compile_warmup()
        assert view.data_ptr() == tensor.data_ptr() + tensor.element_size()
        assert CUDABackend.get_tensor_specialization(tensor, align=True) == "D"
        assert CUDABackend.get_tensor_specialization(view, align=True) == ""
        assert MXFP4Tensor(size=(16, ), device="cuda").random().to(torch.float32).shape == (16, )
        assert MXScaleTensor(torch.rand(16, device="cuda")).to(torch.float32).shape == (16, )

    assert result == "compiled"
    assert type(tensor).__name__ == "FakeTensor"
    assert launches == [((tensor, ), (2, ), {"BLOCK_SIZE": 16})]
    assert not is_compile_warmup()
    assert triton.KernelInterface.__getitem__ is previous_getitem
    assert torch.testing.assert_close is previous_assert_close


def test_compile_warmup_random_helpers_preserve_normal_randomness():
    for operation, original in (
        (lambda: random_int(0, 9), lambda: int(torch.randint(0, 9, size=()).item())),
        (random_float, lambda: float(torch.rand(()).item())),
    ):
        torch.manual_seed(123)
        actual = operation()
        torch.manual_seed(123)
        expected = original()
        torch.testing.assert_close(torch.as_tensor(actual), torch.as_tensor(expected))

    with compile_warmup_only():
        assert random_int(2, 9) == 2
        assert random_float() == 0.5


@pytest.mark.parametrize("fails", [False, True])
def test_compile_warmup_coordinator_deduplicates_and_propagates_errors(monkeypatch, fails):
    coordinator = SharedWarmupCoordinator(max_workers=1, trace_directory=None)
    submissions = []

    def submit(function, *args):
        submissions.append(args)
        future = Future()
        if fails:
            future.set_exception(RuntimeError("compiler failed"))
        else:
            future.set_result((os.getpid(), 0.01))
        return future

    monkeypatch.setattr(coordinator._dispatcher._executor, "submit", submit)
    for _ in range(2):
        connection = Client(coordinator.address, family="AF_UNIX")
        connection.send(("same-specialization", ()))
        connection.close()
    if fails:
        with pytest.raises(RuntimeError, match="compiler failed"):
            coordinator.close()
    else:
        coordinator.close()

    assert submissions == [()]


@pytest.mark.parametrize(
    ("num_gpus", "visible", "expected"),
    [
        (1, None, ["0", "0", "0"]),
        (2, None, ["0", "1", "0", "1"]),
        (2, "3,1", ["3", "1", "3", "1"]),
        (4, "5,2,7,3", ["5", "2", "7", "3", "5"]),
    ],
)
def test_compile_warmup_assigns_gpu_before_xdist_worker_initialization(monkeypatch, num_gpus, visible, expected):
    monkeypatch.setenv("TRITON_TEST_NUM_GPUS", str(num_gpus))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("TRITON_TEST_VISIBLE_GPUS", raising=False)
    if visible is not None:
        monkeypatch.setenv("TRITON_TEST_VISIBLE_GPUS", visible)
    specs = [SimpleNamespace(env={}) for _ in expected]

    pytest_xdist_setupnodes(None, specs)

    assert [spec.env["CUDA_VISIBLE_DEVICES"] for spec in specs] == expected


@pytest.mark.parametrize(("capability", "num_gpus", "kernel_workers"), [(8, 1, 6), (9, 1, 4), (10, 2, 6), (10, 4, 12)])
def test_unit_runner_balances_requested_gpu_shards(monkeypatch, capability, num_gpus, kernel_workers):
    concurrent = []
    deferred = []
    monkeypatch.setattr(_test_runner, "_validate_gpus", lambda count: None)
    monkeypatch.setattr(_test_runner, "_capability", lambda: capability)
    monkeypatch.setattr(_test_runner, "_concurrent", lambda commands: concurrent.extend(commands) or 0)
    monkeypatch.setattr(_test_runner, "_run", lambda command, **kwargs: deferred.append((command, kwargs)) or 0)
    options = SimpleNamespace(num_gpus=num_gpus, num_procs=24, debug_procs=4, kernel_procs=None)

    assert _test_runner._unit(options) == 0
    if capability >= 9:
        assert all(environment["TRITON_TEST_NUM_GPUS"] == str(num_gpus) for _, _, environment in concurrent)
        kernels = deferred[0]
    else:
        assert deferred[0][1]["num_gpus"] == num_gpus
        kernels = deferred[1]
    command, kwargs = kernels
    assert command[command.index("-n") + 1] == str(kernel_workers)
    assert kwargs["num_gpus"] == num_gpus
    attention, kwargs = deferred[1 if capability >= 9 else 2]
    assert attention[attention.index("-n") + 1] == str(num_gpus)
    assert kwargs["num_gpus"] == num_gpus


@pytest.mark.parametrize(("capability", "num_gpus"), [(8, 1), (10, 1), (10, 2), (10, 4), (10, 8)])
def test_gluon_runner_balances_requested_gpu_shards(monkeypatch, capability, num_gpus):
    concurrent = []
    deferred = []
    monkeypatch.setattr(_test_runner, "_validate_gpus", lambda count: None)
    monkeypatch.setattr(_test_runner, "_capability", lambda: capability)
    monkeypatch.setattr(_test_runner, "_concurrent", lambda commands: concurrent.extend(commands) or 0)
    monkeypatch.setattr(_test_runner, "_run", lambda command, **kwargs: deferred.append((command, kwargs)) or 0)
    options = SimpleNamespace(num_gpus=num_gpus, num_procs=64, gluon_procs=8, consan_procs=4, example_procs=4)

    assert _test_runner._gluon(options) == 0
    if capability < 9:
        assert len(deferred) == 2
        assert not concurrent
        return
    assert len(concurrent) == 2
    general, consan = concurrent
    assert general[0][general[0].index("-n") + 1] == str(8 * num_gpus)
    assert consan[0][consan[0].index("-n") + 1] == str(4 * num_gpus)
    assert all(environment["TRITON_TEST_NUM_GPUS"] == str(num_gpus) for _, _, environment in concurrent)
    examples, kwargs = deferred[0]
    assert examples[examples.index("-n") + 1] == str(4 * num_gpus)
    assert "--import-mode=importlib" not in general[0]
    assert "--import-mode=importlib" in examples
    assert kwargs["num_gpus"] == num_gpus


@pytest.mark.parametrize("num_gpus", [1, 2])
def test_gsan_runner_isolates_distributed_tests_when_multiple_gpus_are_visible(monkeypatch, num_gpus):
    commands = []
    monkeypatch.setattr(_test_runner, "_run", lambda command, **kwargs: commands.append((command, kwargs)) or 0)

    assert _test_runner._gsan(SimpleNamespace(num_gpus=num_gpus, num_procs=24)) == 0
    assert len(commands) == num_gpus
    assert commands[0][0][commands[0][0].index("-n") + 1] == str(8 * num_gpus)
    assert all(kwargs["timeout"] == 180 for _, kwargs in commands)
    assert all(kwargs["environment"]["TRITON_TEST_PROCESS_TIMEOUT"] == "90" for _, kwargs in commands)
    if num_gpus == 1:
        assert "not xdist_group" not in commands[0][0]
    else:
        assert "not xdist_group" in commands[0][0]
        assert commands[0][1]["environment"]["TRITON_TEST_NUM_GPUS"] == str(num_gpus)
        assert "xdist_group" in commands[1][0]
        assert commands[1][0][commands[1][0].index("-n") + 1] == "1"


def test_gsan_runner_terminates_stalled_pytest_processes():
    command = [sys.executable, "-c", "import time; time.sleep(60)"]
    assert _test_runner._run(command, timeout=0.05) == 1


def test_compile_warmup_selects_eligible_markers(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 0))
    enabled = SimpleNamespace(kwargs={})
    unsupported = SimpleNamespace(kwargs={"min_capability": 9})
    excluded = object()

    def item(markers):
        return SimpleNamespace(get_closest_marker=lambda name: markers.get(name))

    items = [
        item({}),
        item({"enable_warmup": enabled}),
        item({"enable_warmup": enabled, "disable_warmup": excluded}),
        item({"enable_warmup": unsupported}),
    ]
    deselected = []
    config = SimpleNamespace(getoption=lambda _: True,
                             hook=SimpleNamespace(pytest_deselected=lambda items: deselected.extend(items)))

    selected = items[1]
    pytest_collection_modifyitems(config, items)

    assert items == [selected]
    assert len(deselected) == 3


def test_compile_warmup_spreads_prioritized_tests_across_capture_workers(monkeypatch):
    monkeypatch.setenv("PYTEST_XDIST_WORKER_COUNT", "2")

    def item(priority):
        marker = SimpleNamespace(kwargs={"priority": priority})
        return SimpleNamespace(get_closest_marker=lambda name: marker if name == "enable_warmup" else None)

    low_a, high_a, low_b, high_b = items = [item(0), item(2), item(0), item(2)]
    config = SimpleNamespace(getoption=lambda _: True, hook=SimpleNamespace(pytest_deselected=lambda items: None))

    pytest_collection_modifyitems(config, items)

    assert items == [high_a, low_a, high_b, low_b]


def test_compile_warmup_serializes_patched_source_globals(monkeypatch):

    @triton.jit
    def helper(value):
        return value + 1

    @triton.jit
    def kernel(output, value: tl.constexpr):
        tl.store(output, value)

    name = "_compile_warmup_serialization_helper"
    monkeypatch.setitem(kernel.fn.__globals__, name, helper)
    kernel._unsafe_update_src(kernel.src.replace("tl.store(output, value)", f"tl.store(output, {name}(value))"))
    assert cloudpickle.loads(_jit_dumps(kernel)).cache_key == kernel.cache_key


def test_compile_warmup_reuses_equivalent_kernel_payloads(monkeypatch):
    dispatcher = ProcessPoolWarmupDispatcher(max_workers=1, trace_directory=None, phase="warmup")
    dispatcher.current_test = "test"
    serialized = []

    def serialize(value):
        if hasattr(value, "fn"):
            serialized.append(value)
        return _jit_dumps(value)

    def submit(*args):
        future = Future()
        future.set_result((os.getpid(), 0))
        return future

    monkeypatch.setattr("triton._compile_warmup_pool._jit_dumps", serialize)
    monkeypatch.setattr(dispatcher._executor, "submit", submit)

    def make_kernel():

        @triton.jit
        def kernel(output, value: tl.constexpr):
            tl.store(output, value)

        return kernel

    kernels = [make_kernel(), make_kernel(), make_kernel()]
    kernels[-1]._unsafe_update_src(kernels[-1].src.replace("tl.store(output, value)", "tl.store(output, value + 1)"))
    try:
        with compile_warmup_only(dispatcher):
            output = torch.empty(1, device="cuda")
            for kernel in kernels:
                kernel[(1, )](output, value=1)
    finally:
        dispatcher.finish()

    assert len(serialized) == 2


def _trace(directory, phase, test, digest, hit):
    source = SimpleNamespace(name="kernel", fn=SimpleNamespace(_fn_name="package.kernel"), hash=lambda: "source")
    CompilationTrace(str(directory), phase, test)(src=source, metadata={"hash": digest}, metadata_group={},
                                                  times=SimpleNamespace(total=125_000), cache_hit=hit)


@pytest.mark.parametrize(
    ("records", "message"),
    [
        (
            [("warmup-unit", "test.py::test", "warmed", False), ("unit", "test.py::test", "warmed", True),
             ("unit", "test.py::test", "different", False)],
            "not complete cache hits",
        ),
        (
            [("warmup-unit", "test.py::test", "used", False), ("warmup-unit", "test.py::test", "unused", False),
             ("unit", "test.py::test", "used", True)],
            "unused specializations",
        ),
        (
            [("warmup-unit", "test.py::present", "used", False), ("warmup-unit", "test.py::missing", "missing", False),
             ("unit", "test.py::present", "used", True)],
            "marked warmup tests were not executed",
        ),
        (
            [("warmup-unit", "unit.py::test", "shared", False), ("warmup-gluon", "gluon.py::test", "shared", False),
             ("unit", "unit.py::test", "shared", True)],
            "marked warmup tests were not executed",
        ),
    ],
)
def test_compilation_trace_rejects_incomplete_warmup(tmp_path, records, message):
    for record in records:
        _trace(tmp_path, *record)
    with pytest.raises(SystemExit, match=message):
        _require_complete_warmup(summarize_compile_trace(str(tmp_path)))


def test_compilation_trace_accepts_global_cross_suite_deduplication(tmp_path):
    _trace(tmp_path, "warmup-unit", "unit.py::test", "shared", False)
    attempted = tmp_path / "warmup-gluon-1.tests"
    attempted.write_text('{"phase": "warmup-gluon", "test": "gluon.py::test"}\n')
    _trace(tmp_path, "unit", "unit.py::test", "shared", True)
    _trace(tmp_path, "gluon", "gluon.py::test", "shared", True)
    with triton.knobs.cache.scope():
        triton.knobs.cache.dir = str(tmp_path / "isolated")
        _trace(tmp_path, "unit", "unit.py::isolated", "shared", False)

    report = summarize_compile_trace(str(tmp_path))

    assert report["phases"]["gluon"]["warmed_test_hits"] == 1
    assert report["phases"]["unit"]["warmed_misses"] == 0
    assert report["unused_warmup_hashes"] == 0
    _require_complete_warmup(report)


def test_compilation_trace_grades_attempted_warmup_tests(tmp_path):
    attempted = tmp_path / "warmup-unit-1.tests"
    attempted.write_text('{"phase": "warmup-unit", "test": "test_core.py::test_missing"}\n')
    _trace(tmp_path, "unit", "test_core.py::test_missing", "cold", False)

    report = summarize_compile_trace(str(tmp_path))

    assert report["warmup_tests"] == 1
    assert report["phases"]["unit"]["warmed_test_misses"] == 1


def test_is_lazy():
    from importlib import reload
    reload(sys.modules["triton.runtime.driver"])
    reload(sys.modules["triton.runtime"])
    assert triton.runtime.driver._active is None
    assert triton.runtime.driver._default is None
    assert isinstance(triton.runtime.driver.active, getattr(triton.backends.driver, "DriverBase"))
    assert isinstance(triton.runtime.driver.default, getattr(triton.backends.driver, "DriverBase"))
    utils = triton.runtime.driver.active.utils  # noqa: F841


def test_profile_scratch_stream_zero_uses_default_stream(monkeypatch):

    class Scratch:

        def __init__(self):
            self.recorded_streams = []

        def record_stream(self, stream):
            self.recorded_streams.append(stream)

    class DeviceInterface:

        def __init__(self):
            self.default_stream_arg = None
            self.stream_args = []

        def ExternalStream(self, stream, device):
            raise AssertionError("stream 0 must use the default stream")

        def stream(self, stream):
            self.stream_args.append(stream)
            return stream

        def default_stream(self, device):
            self.default_stream_arg = device
            return self

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    zeros_calls = []

    def zeros(size, dtype, device):
        zeros_calls.append((size, dtype, device))
        return Scratch()

    device_interface = DeviceInterface()
    driver = SimpleNamespace(get_active_torch_device=lambda: "cuda:0", get_device_interface=lambda: device_interface)
    monkeypatch.setattr(torch, "zeros", zeros)

    scratch = GPUDriver.allocate_default_profile_scratch(driver, 16, 8, 0)

    assert device_interface.default_stream_arg == "cuda:0"
    assert device_interface.stream_args == [device_interface]
    assert zeros_calls == [(16, torch.int8, "cuda:0")]
    assert scratch.recorded_streams == []


def test_kernel_in_thread(device):
    # Test calling in a new thread sets a valid device context
    buf = torch.zeros((38016 * 1024, ), dtype=torch.float32, device=device)

    @triton.jit
    def _kernel(P, BLOCK: tl.constexpr):
        pid = tl.program_id(0).to(tl.int64)
        offset = pid * BLOCK + tl.arange(0, BLOCK)

        p = tl.load(P + offset)
        tl.store(P + offset, p)

    def call_triton():
        N = buf.numel()
        grid = lambda meta: (triton.cdiv(N, meta["BLOCK"]), )
        _kernel[grid](buf, BLOCK=1024)
        getattr(torch, device).synchronize()

    call_triton()
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(call_triton)
        future.result()


def test_expand_signature_with_aggregate_tensordesc():
    signature = (
        "i32",
        ("tensordesc<fp16[16,32]>", "i64"),
        "tensordesc_im2col<fp32[1,16],input_rank=4,'layout'>",
    )
    expanded = expand_signature(signature, [], "nvTmaDesc")

    assert expanded[0] == "i32"
    assert expanded[1] == ("*fp16", *["i64"] * 4, *["i1"] * 2, *["i32"] * 2, *["i64"] * 3)
    # input_rank=4 drives the number of shape/stride entries for im2col.
    assert expanded[2:] == ["*fp32", *["i64"] * 8, *["i1"] * 2, *["i32"] * 4, *["i64"] * 4]

    expanded = expand_signature(signature, [{}, {}], "nvTmaDesc")
    assert expanded[0] == "i32"
    assert expanded[1] == ("nvTmaDesc", *["i32"] * 2, *["i64"] * 3)
    assert expanded[2:] == ["nvTmaDesc", *["i32"] * 4, *["i64"] * 4]


def test_wrap_tensordesc_handles_aggregate_arguments():
    signature = {0: ("tensordesc<fp16[16,16]>", "i32"), 1: "i64", 2: "tensordesc<fp16[16,16]>"}
    outer_meta = {"tag": "outer"}
    launcher_calls = []

    def launcher(*args):
        launcher_calls.append(args)
        return "ok"

    def make_descriptor(arg, meta, base_args):
        return [("desc", arg, meta, base_args[0]), ("shape", arg)]

    wrapped = wrap_handle_tensordesc_impl(launcher, signature, [None, outer_meta], make_descriptor)
    assert wrapped("meta0", "meta1", (("A", 7), 9, "B")) == "ok"

    assert len(launcher_calls) == 1
    assert launcher_calls[0][0] == "meta0"
    assert launcher_calls[0][1] == "meta1"
    assert launcher_calls[0][2] == [
        (("desc", "A", None, "meta0"), ("shape", "A"), 7),
        9,
        ("desc", "B", outer_meta, "meta0"),
        ("shape", "B"),
    ]


def test_wrap_tensordesc_is_noop_without_tensordesc():

    def launcher(*args):
        return args

    wrapped = wrap_handle_tensordesc_impl(launcher, {0: "i32", 1: ("i64", "constexpr")}, None, lambda *_: [])
    assert wrapped is launcher
