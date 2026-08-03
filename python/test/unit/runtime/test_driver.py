import os
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from multiprocessing.connection import Client
from pathlib import Path
from types import SimpleNamespace

import cloudpickle
import pytest
import torch

import triton
import triton.language as tl
from triton._compile_warmup import (
    CompilationTrace,
    _cache_phase_for_item,
    _main,
    _require_complete_warmup,
    compile_warmup_only,
    pytest_collection_modifyitems,
    pytest_runtest_call,
    summarize_compile_trace,
)
from triton._compile_warmup_pool import (
    ProcessPoolWarmupDispatcher,
    SharedWarmupCoordinator,
    _jit_callable_import_path,
    _jit_dumps,
)
from triton._internal_testing import assert_close, is_compile_warmup, rand, randint, random_float, random_int, randn, reference_tensor
from triton import _test_runner
from triton.backends.driver import GPUDriver, expand_signature, wrap_handle_tensordesc_impl
from triton.backends.nvidia.compiler import CUDABackend


@triton.jit
def _compile_warmup_importable_kernel(output):
    tl.store(output, 1)


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
        result = kernel[(2, )](tensor, BLOCK_SIZE=16)
        assert_close(tensor, tensor)
        assert is_compile_warmup()

    assert result == "compiled"
    assert type(tensor).__name__ == "FakeTensor"
    assert launches == [((tensor, ), (2, ), {"BLOCK_SIZE": 16})]
    assert triton.KernelInterface.__getitem__ is previous_getitem
    assert torch.testing.assert_close is previous_assert_close
    assert not is_compile_warmup()


def test_compile_warmup_preserves_tensor_view_alignment():
    with compile_warmup_only():
        tensor = torch.empty(16, device="cpu", dtype=torch.float32)
        view = tensor[1:]

        assert tensor.data_ptr() % 256 == 0
        assert view.data_ptr() == tensor.data_ptr() + tensor.element_size()
        assert CUDABackend.get_tensor_specialization(tensor, align=True) == "D"
        assert CUDABackend.get_tensor_specialization(view, align=True) == ""


def test_compile_warmup_random_helpers_preserve_normal_randomness():
    for operation, original in (
        (lambda: rand(4), lambda: torch.rand(4)),
        (lambda: randn(4), lambda: torch.randn(4)),
        (lambda: randint(0, 9, (4, )), lambda: torch.randint(0, 9, (4, ))),
        (lambda: random_int(0, 9), lambda: int(torch.randint(0, 9, size=()).item())),
        (random_float, lambda: float(torch.rand(()).item())),
    ):
        torch.manual_seed(123)
        actual = operation()
        torch.manual_seed(123)
        expected = original()
        if isinstance(actual, torch.Tensor):
            torch.testing.assert_close(actual, expected)
        else:
            assert actual == expected

    with compile_warmup_only():
        assert type(rand(4)).__name__ == "FakeTensor"
        assert type(randn(4)).__name__ == "FakeTensor"
        assert type(randint(0, 9, (4, ))).__name__ == "FakeTensor"
        assert random_int(2, 9) == 2
        assert random_float() == 0.5
        wrapped = SimpleNamespace(data=torch.empty(4), to=lambda dtype: torch.ones(4, dtype=dtype))
        assert reference_tensor(wrapped, torch.float32).shape == (4, )

    wrapped = SimpleNamespace(data=torch.empty(4), to=lambda dtype: torch.ones(4, dtype=dtype))
    torch.testing.assert_close(reference_tensor(wrapped, torch.float32), torch.ones(4))


def test_compile_warmup_process_pool_requires_workers():
    with pytest.raises(ValueError, match="max_workers must be >= 1"):
        ProcessPoolWarmupDispatcher(max_workers=0, trace_directory=None, phase="warmup-test")


def test_compile_warmup_coordinator_deduplicates_capture_workers(monkeypatch):
    coordinator = SharedWarmupCoordinator(max_workers=1, trace_directory=None)
    submissions = []

    def submit(function, *args):
        submissions.append(args)
        future = Future()
        future.set_result((os.getpid(), 0.01))
        return future

    monkeypatch.setattr(coordinator._dispatcher._executor, "submit", submit)
    for _ in range(2):
        connection = Client(coordinator.address, family="AF_UNIX")
        connection.send(("same-specialization", ()))
        connection.close()
    coordinator.close()

    assert submissions == [()]


def test_compile_warmup_coordinator_propagates_compilation_errors(monkeypatch):
    coordinator = SharedWarmupCoordinator(max_workers=1, trace_directory=None)

    def submit(function, *args):
        future = Future()
        future.set_exception(RuntimeError("compiler failed"))
        return future

    monkeypatch.setattr(coordinator._dispatcher._executor, "submit", submit)
    connection = Client(coordinator.address, family="AF_UNIX")
    connection.send(("specialization", ()))
    connection.close()

    with pytest.raises(RuntimeError, match="compiler failed"):
        coordinator.close()


def test_test_runner_preserves_default_import_mode_unless_matching_unified_warmup():
    assert "--import-mode=importlib" not in _test_runner._pytest("test.py", workers=1)
    assert "--import-mode=importlib" in _test_runner._pytest("test.py", workers=1, import_mode="importlib")


@pytest.mark.parametrize("num_gpus", [1, 2, 4, 8])
@pytest.mark.parametrize("capability", [9, 10])
def test_gluon_runner_balances_requested_gpu_shards(monkeypatch, num_gpus, capability):
    concurrent = []
    deferred = []
    monkeypatch.setattr(_test_runner, "_validate_gpus", lambda count: None)
    monkeypatch.setattr(_test_runner, "_capability", lambda: capability)
    monkeypatch.setattr(_test_runner, "_concurrent", lambda commands: concurrent.extend(commands) or 0)
    monkeypatch.setattr(_test_runner, "_run", lambda command, **kwargs: deferred.append((command, kwargs)) or 0)
    options = SimpleNamespace(num_gpus=num_gpus, num_procs=64, gluon_procs=8, consan_procs=4, example_procs=4)

    assert _test_runner._gluon(options) == 0
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


def test_gluon_runner_skips_consan_pool_on_ampere(monkeypatch):
    commands = []
    monkeypatch.setattr(_test_runner, "_validate_gpus", lambda count: None)
    monkeypatch.setattr(_test_runner, "_capability", lambda: 8)
    monkeypatch.setattr(_test_runner, "_run", lambda command, **kwargs: commands.append(command) or 0)
    options = SimpleNamespace(num_gpus=1, num_procs=8, gluon_procs=8, consan_procs=4, example_procs=4)

    assert _test_runner._gluon(options) == 0
    assert len(commands) == 2
    assert "python/test/gluon/test_consan.py" not in commands[0]


@pytest.mark.parametrize("num_gpus", [1, 2, 4])
def test_gsan_runner_isolates_distributed_tests_when_multiple_gpus_are_visible(monkeypatch, num_gpus):
    commands = []
    monkeypatch.setattr(_test_runner, "_run", lambda command, **kwargs: commands.append((command, kwargs)) or 0)

    assert _test_runner._gsan(SimpleNamespace(num_gpus=num_gpus, num_procs=24)) == 0
    assert all(kwargs["environment"]["TRITON_CI_CACHE_PHASE"] == "gsan" for _, kwargs in commands)
    assert all(kwargs["environment"]["TRITON_DISABLE_LINE_INFO"] == "0" for _, kwargs in commands)
    assert all("--dist=loadgroup" in command for command, _ in commands)
    assert commands[0][0][commands[0][0].index("-n") + 1] == "24"

    symmetric_memory = "python/test/gsan/test_symmetric_memory.py"
    if num_gpus == 1:
        assert len(commands) == 1
        assert f"--ignore={symmetric_memory}" not in commands[0][0]
    else:
        assert len(commands) == 2
        assert f"--ignore={symmetric_memory}" in commands[0][0]
        assert symmetric_memory in commands[1][0]
        assert commands[1][0][commands[1][0].index("-n") + 1] == "1"


def test_compile_warmup_selects_explicit_markers():
    enabled = SimpleNamespace(kwargs={})
    excluded = object()

    def item(markers):
        return SimpleNamespace(get_closest_marker=lambda name: markers.get(name))

    items = [item({}), item({"enable_warmup": enabled}), item({"enable_warmup": enabled, "disable_warmup": excluded})]
    deselected = []
    config = SimpleNamespace(getoption=lambda _: True,
                             hook=SimpleNamespace(pytest_deselected=lambda items: deselected.extend(items)))

    selected = items[1]
    pytest_collection_modifyitems(config, items)

    assert items == [selected]
    assert len(deselected) == 2


def test_compile_warmup_filters_gpu_capabilities(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 0))
    marker = SimpleNamespace(kwargs={"min_capability": 9})
    items = [SimpleNamespace(get_closest_marker=lambda name: marker if name == "enable_warmup" else None)]
    config = SimpleNamespace(getoption=lambda _: True, hook=SimpleNamespace(pytest_deselected=lambda items: None))

    pytest_collection_modifyitems(config, items)

    assert items == []


def test_compile_warmup_attributes_combined_session_phases(monkeypatch):
    monkeypatch.setenv("TRITON_CI_CACHE_PHASE", "warmup-default")
    config = SimpleNamespace(
        rootpath=Path("/checkout"), getoption=lambda _: [
            "python/tutorials/06-fused-attention.py=warmup-attention",
            "python/examples/gluon=warmup-gluon-examples",
        ])

    attention = SimpleNamespace(config=config, path=Path("/checkout/python/tutorials/06-fused-attention.py"))
    example = SimpleNamespace(config=config, path=Path("/checkout/python/examples/gluon/01-attention-forward.py"))
    unit = SimpleNamespace(config=config, path=Path("/checkout/python/test/unit/language/test_core.py"))

    assert _cache_phase_for_item(attention) == "warmup-attention"
    assert _cache_phase_for_item(example) == "warmup-gluon-examples"
    assert _cache_phase_for_item(unit) == "warmup-default"


def test_compile_warmup_does_not_record_skipped_cases(tmp_path, monkeypatch):
    monkeypatch.setenv("TRITON_CI_COMPILE_TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("TRITON_CI_CACHE_PHASE", "warmup-unit")
    recorded = []
    dispatcher = SimpleNamespace(current_test=None, current_phase="warmup-unit",
                                 record_test=lambda test, phase: recorded.append((test, phase)))
    config = SimpleNamespace(rootpath=tmp_path, _triton_warmup_dispatcher=dispatcher, getoption=lambda name: []
                             if name == "--warmup-phase" else True)
    item = SimpleNamespace(config=config, path=tmp_path / "test.py", nodeid="test.py::unsupported")

    hook = pytest_runtest_call(item)
    next(hook)
    with pytest.raises(pytest.skip.Exception):
        hook.throw(pytest.skip.Exception("unsupported fake operation"))

    assert recorded == []
    assert dispatcher.current_test is None


def test_compile_warmup_serializes_local_jit_function():

    @triton.jit
    def kernel(output, value: tl.constexpr):
        tl.store(output, value)

    restored = cloudpickle.loads(_jit_dumps(kernel))

    assert restored.src == kernel.src
    assert restored.__name__ == kernel.__name__


def test_compile_warmup_serializes_patched_source_globals():

    @triton.jit
    def helper(value):
        return value + 1

    @triton.jit
    def kernel(output, value: tl.constexpr):
        tl.store(output, value)

    name = "_compile_warmup_serialization_helper"
    kernel.fn.__globals__[name] = helper
    kernel._unsafe_update_src(kernel.src.replace("tl.store(output, value)", f"tl.store(output, {name}(value))"))
    try:
        assert cloudpickle.loads(_jit_dumps(kernel)).cache_key == kernel.cache_key
    finally:
        del kernel.fn.__globals__[name]


def test_compile_warmup_serializes_wrapped_module_jit_function(monkeypatch):
    function = _compile_warmup_importable_kernel
    monkeypatch.setattr(sys.modules[__name__], function.__name__, SimpleNamespace(fn=function))

    assert _jit_callable_import_path(function) is None
    assert cloudpickle.loads(_jit_dumps(function)).cache_key == function.cache_key


def test_compile_warmup_serializes_generated_module_jit_function(monkeypatch):
    function = _compile_warmup_importable_kernel
    monkeypatch.setattr(sys.modules[__name__], "__file__", "/missing/generated_test_driver.py")

    assert _jit_callable_import_path(function) is None
    assert cloudpickle.loads(_jit_dumps(function)).cache_key == function.cache_key


def _trace(directory, phase, test, digest, hit):
    source = SimpleNamespace(name="kernel", fn=SimpleNamespace(_fn_name="package.kernel"), hash=lambda: "source")
    CompilationTrace(str(directory), phase, test)(src=source, metadata={"hash": digest}, metadata_group={},
                                                  times=SimpleNamespace(total=125_000), cache_hit=hit)


def test_compilation_trace_requires_warmed_cache_hits(tmp_path):
    _trace(tmp_path, "warmup-unit", "test.py::test", "warmed", False)
    _trace(tmp_path, "unit", "test.py::test", "warmed", True)
    _trace(tmp_path, "unit", "test.py::test", "different", False)

    report = summarize_compile_trace(str(tmp_path))

    assert report["phases"]["unit"]["warmed_test_hits"] == 1
    assert report["phases"]["unit"]["warmed_test_misses"] == 1
    with pytest.raises(SystemExit, match="not complete cache hits"):
        _require_complete_warmup(report)


def test_compilation_trace_rejects_unused_warmed_specializations(tmp_path):
    _trace(tmp_path, "warmup-unit", "test.py::test", "used", False)
    _trace(tmp_path, "warmup-unit", "test.py::test", "unused", False)
    _trace(tmp_path, "unit", "test.py::test", "used", True)

    report = summarize_compile_trace(str(tmp_path))

    assert report["unused_warmup_hashes"] == 1
    with pytest.raises(SystemExit, match="unused specializations"):
        _require_complete_warmup(report)


def test_compilation_trace_ignores_identical_hashes_in_isolated_caches(tmp_path):
    _trace(tmp_path, "warmup-unit", "test.py::warmed", "shared-hash", False)
    _trace(tmp_path, "unit", "test.py::warmed", "shared-hash", True)
    with triton.knobs.cache.scope():
        triton.knobs.cache.dir = str(tmp_path / "isolated")
        _trace(tmp_path, "unit", "test.py::isolated", "shared-hash", False)

    report = summarize_compile_trace(str(tmp_path))

    assert report["phases"]["unit"]["warmed_misses"] == 0
    assert report["phases"]["unit"]["warmed_test_misses"] == 0
    assert report["unused_warmup_hashes"] == 0
    _require_complete_warmup(report)


def test_compilation_trace_accepts_global_cross_suite_deduplication(tmp_path):
    _trace(tmp_path, "warmup-unit", "unit.py::test", "shared", False)
    attempted = tmp_path / "warmup-gluon-1.tests"
    attempted.write_text('{"phase": "warmup-gluon", "test": "gluon.py::test"}\n')
    _trace(tmp_path, "unit", "unit.py::test", "shared", True)
    _trace(tmp_path, "gluon", "gluon.py::test", "shared", True)

    report = summarize_compile_trace(str(tmp_path))

    assert report["phases"]["warmup-unit"]["warmed_misses"] == 0
    assert report["phases"]["gluon"]["warmed_test_hits"] == 1
    assert report["unused_warmup_hashes"] == 0
    _require_complete_warmup(report)


def test_compilation_trace_rejects_missing_marked_execution(tmp_path):
    _trace(tmp_path, "warmup-unit", "test.py::present", "used", False)
    _trace(tmp_path, "warmup-unit", "test.py::missing", "missing", False)
    _trace(tmp_path, "unit", "test.py::present", "used", True)

    with pytest.raises(SystemExit, match="marked warmup tests were not executed"):
        _require_complete_warmup(summarize_compile_trace(str(tmp_path)))


def test_compilation_trace_rejects_entire_missing_runtime_phase(tmp_path):
    _trace(tmp_path, "warmup-unit", "unit.py::test", "shared", False)
    _trace(tmp_path, "warmup-gluon", "gluon.py::test", "shared", False)
    _trace(tmp_path, "unit", "unit.py::test", "shared", True)

    report = summarize_compile_trace(str(tmp_path))

    assert report["unused_warmup_hashes"] == 0
    assert report["phases"]["gluon"]["missing_warmed_tests"] == ["gluon.py::test"]
    with pytest.raises(SystemExit, match="marked warmup tests were not executed"):
        _require_complete_warmup(report)


def test_compilation_trace_grades_attempted_warmup_tests(tmp_path):
    attempted = tmp_path / "warmup-unit-1.tests"
    attempted.write_text('{"phase": "warmup-unit", "test": "test_core.py::test_missing"}\n')
    _trace(tmp_path, "unit", "test_core.py::test_missing", "cold", False)

    report = summarize_compile_trace(str(tmp_path), "unit")

    assert report["warmup_tests"] == 1
    assert report["phases"]["unit"]["warmed_test_misses"] == 1


def test_compilation_trace_reports_multiple_phases(tmp_path, monkeypatch, capsys):
    for phase in ("unit", "attention"):
        _trace(tmp_path, phase, f"{phase}::test", phase, True)
    monkeypatch.setattr(sys, "argv",
                        ["warmup", "report", "--directory",
                         str(tmp_path), "--phase", "unit", "--phase", "attention"])

    _main()

    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 2
    assert '"phases": {"unit":' in lines[0]
    assert '"phases": {"attention":' in lines[1]


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

    interface = DeviceInterface()
    driver = SimpleNamespace(get_active_torch_device=lambda: "cuda:0", get_device_interface=lambda: interface)
    monkeypatch.setattr(torch, "zeros", zeros)

    scratch = GPUDriver.allocate_default_profile_scratch(driver, 16, 8, 0)

    assert interface.default_stream_arg == "cuda:0"
    assert interface.stream_args == [interface]
    assert zeros_calls == [(16, torch.int8, "cuda:0")]
    assert scratch.recorded_streams == []


def test_kernel_in_thread(device):
    buf = torch.zeros((38016 * 1024, ), dtype=torch.float32, device=device)

    @triton.jit
    def _kernel(P, BLOCK: tl.constexpr):
        pid = tl.program_id(0).to(tl.int64)
        offset = pid * BLOCK + tl.arange(0, BLOCK)
        tl.store(P + offset, tl.load(P + offset))

    def call_triton():
        grid = lambda meta: (triton.cdiv(buf.numel(), meta["BLOCK"]), )
        _kernel[grid](buf, BLOCK=1024)
        getattr(torch, device).synchronize()

    call_triton()
    with ThreadPoolExecutor(1) as pool:
        pool.submit(call_triton).result()


def test_expand_signature_with_aggregate_tensordesc():
    signature = ("i32", ("tensordesc<fp16[16,32]>", "i64"), "tensordesc_im2col<fp32[1,16],input_rank=4,'layout'>")
    expanded = expand_signature(signature, [], "nvTmaDesc")

    assert expanded[0] == "i32"
    assert expanded[1] == ("*fp16", *["i64"] * 4, *["i1"] * 2, *["i32"] * 2, *["i64"] * 3)
    assert expanded[2:] == ["*fp32", *["i64"] * 8, *["i1"] * 2, *["i32"] * 4, *["i64"] * 4]

    expanded = expand_signature(signature, [{}, {}], "nvTmaDesc")
    assert expanded[0] == "i32"
    assert expanded[1] == ("nvTmaDesc", *["i32"] * 2, *["i64"] * 3)
    assert expanded[2:] == ["nvTmaDesc", *["i32"] * 4, *["i64"] * 4]


def test_wrap_tensordesc_handles_aggregate_arguments():
    signature = {0: ("tensordesc<fp16[16,16]>", "i32"), 1: "i64", 2: "tensordesc<fp16[16,16]>"}
    outer_meta = {"tag": "outer"}
    calls = []

    def launcher(*args):
        calls.append(args)
        return "ok"

    def descriptor(arg, meta, base_args):
        return [("desc", arg, meta, base_args[0]), ("shape", arg)]

    wrapped = wrap_handle_tensordesc_impl(launcher, signature, [None, outer_meta], descriptor)
    assert wrapped("meta0", "meta1", (("A", 7), 9, "B")) == "ok"
    assert calls == [("meta0", "meta1", [(("desc", "A", None, "meta0"), ("shape", "A"), 7), 9,
                                         ("desc", "B", outer_meta, "meta0"), ("shape", "B")])]


def test_wrap_tensordesc_is_noop_without_tensordesc():

    def launcher(*args):
        return args

    assert wrap_handle_tensordesc_impl(launcher, {0: "i32", 1: ("i64", "constexpr")}, None, lambda *_: []) is launcher
