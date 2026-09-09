import json
import os
import warnings
import weakref
from contextlib import contextmanager

import pytest
import torch
from torch.overrides import TorchFunctionMode

import triton
from triton._compile_trace import CompilationTrace


class _FakeCudaTensorMode(TorchFunctionMode):
    """Preserve CUDA allocation alignment and view offsets in fake pointers."""

    _STORAGE_ALIGNMENT = 256
    _STORAGE_STRIDE = 1 << 20

    def __init__(self):
        from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

        self.fake_mode = FakeTensorMode(allow_fallback_kernels=False, allow_non_fake_inputs=True)
        self.fake_tensor_type = FakeTensor
        self.storage_pointers = weakref.WeakKeyDictionary()
        self.next_storage_pointer = self._STORAGE_STRIDE + self._STORAGE_ALIGNMENT

    def __enter__(self):
        self.fake_mode.__enter__()
        try:
            return super().__enter__()
        except BaseException as error:
            self.fake_mode.__exit__(type(error), error, error.__traceback__)
            raise

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            self.fake_mode.__exit__(exc_type, exc_value, traceback)

    def __torch_function__(self, func, types, args=(), kwargs=None):
        name = getattr(func, "__name__", "")
        if name == "data_ptr":
            tensor = args[0]
            storage = tensor.untyped_storage()
            pointer = self.storage_pointers.get(storage)
            if pointer is None:
                pointer = self.next_storage_pointer
                self.next_storage_pointer += self._STORAGE_STRIDE
                self.storage_pointers[storage] = pointer
            return pointer + tensor.storage_offset() * tensor.element_size()
        is_fake = any(isinstance(argument, self.fake_tensor_type) for argument in args)
        if name in ("allclose", "equal") and is_fake:
            return True
        if name == "__bool__" and is_fake and args[0].dtype == torch.bool:
            return getattr(args[0], "_triton_warmup_truth", True)
        if name in ("all", "any") and is_fake:
            result = func(*args, **(kwargs or {}))
            if isinstance(result, torch.Tensor) and result.numel() == 1:
                result._triton_warmup_truth = name == "all"
            return result
        return func(*args, **(kwargs or {}))


@contextmanager
def compile_warmup_only(dispatcher=None):
    """Capture launch specializations without GPU allocations or kernel execution."""
    from torch._subclasses.fake_tensor import FakeTensor
    from triton._compile_warmup_state import _COMPILE_WARMUP_ACTIVE

    def fake_assert_close(*args, **kwargs):
        if not any(isinstance(value, FakeTensor) for value in args):
            previous_assert_close(*args, **kwargs)

    def dispatch(kernel, grid, *args, **kwargs):
        if dispatcher is None:
            return kernel.warmup(*args, grid=grid, **kwargs)
        return dispatcher.dispatch(*args, kernel=kernel, grid=grid, test=dispatcher.current_test, **kwargs)

    with triton.knobs.runtime.scope(), warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Accessing the data pointer of FakeTensor.*")
        with _FakeCudaTensorMode():
            previous_getitem = triton.KernelInterface.__getitem__
            previous_assert_close = torch.testing.assert_close
            triton.KernelInterface.__getitem__ = lambda kernel, grid: lambda *args, **kwargs: dispatch(
                kernel, grid, *args, **kwargs)
            torch.testing.assert_close = fake_assert_close
            active_token = _COMPILE_WARMUP_ACTIVE.set(True)
            try:
                yield
            finally:
                _COMPILE_WARMUP_ACTIVE.reset(active_token)
                triton.KernelInterface.__getitem__ = previous_getitem
                torch.testing.assert_close = previous_assert_close


@contextmanager
def process_pool_compile_warmup(*, workers, directory, phase):
    from triton._compile_warmup_pool import ProcessPoolWarmupDispatcher

    dispatcher = ProcessPoolWarmupDispatcher(max_workers=workers, trace_directory=directory, phase=phase)
    dispatcher.current_test = None
    try:
        with compile_warmup_only(dispatcher):
            yield dispatcher
    finally:
        dispatcher.finish()


def pytest_addoption(parser):
    parser.addoption("--warmup-only", action="store_true", help="compile explicitly enabled tests without execution")
    parser.addoption("--warmup-workers", type=int, default=1, help="maximum shared compiler processes")
    parser.addoption("--warmup-phase", action="append", default=[], metavar="PATH=PHASE",
                     help="associate a test path with its runtime suite")


@pytest.hookimpl(optionalhook=True)
def pytest_xdist_setupnodes(config, specs):
    requested = os.environ.get("TRITON_TEST_NUM_GPUS")
    if not requested:
        return

    visibility_variable = "HIP_VISIBLE_DEVICES" if torch.version.hip else "CUDA_VISIBLE_DEVICES"
    visible = os.environ.get("TRITON_TEST_VISIBLE_GPUS", os.environ.get(visibility_variable))
    if visible:
        devices = [device.strip() for device in visible.split(",") if device.strip()]
    else:
        devices = [str(index) for index in range(int(requested))]

    for index, spec in enumerate(specs):
        spec.env[visibility_variable] = devices[index % int(requested)]


def _cache_phase_for_item(item):
    phase = os.environ.get("TRITON_CI_CACHE_PHASE", "unclassified")
    root = os.path.abspath(str(item.config.rootpath))
    path = os.path.relpath(os.path.abspath(str(item.path)), root)
    for rule in item.config.getoption("--warmup-phase"):
        prefix, separator, configured_phase = rule.rpartition("=")
        if not separator or not prefix or not configured_phase:
            raise pytest.UsageError(f"invalid --warmup-phase {rule!r}; expected PATH=PHASE")
        prefix = os.path.normpath(prefix)
        if path == prefix or path.startswith(prefix + os.sep):
            phase = configured_phase
    return phase


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--warmup-only"):
        return

    selected = []
    deselected = []
    target = None
    capability = None
    for item in items:
        marker = item.get_closest_marker("enable_warmup")
        excluded = item.get_closest_marker("disable_warmup")
        if marker is None or excluded is not None:
            deselected.append(item)
            continue
        minimum = marker.kwargs.get("min_capability")
        if minimum is not None:
            if target is None:
                target = triton.runtime.driver.active.get_current_target()
            if target.backend == "cuda":
                if capability is None:
                    capability = torch.cuda.get_device_capability()[0]
                if capability < minimum:
                    deselected.append(item)
                    continue
        selected.append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
    selected.sort(key=lambda item: item.get_closest_marker("enable_warmup").kwargs.get("priority", 0), reverse=True)
    capture_workers = int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", "1"))
    items[:] = [item for worker in range(capture_workers) for item in selected[worker::capture_workers]]


@pytest.fixture(scope="session", autouse=True)
def compile_warmup(request):
    if not request.config.getoption("--warmup-only"):
        yield
        return

    directory = os.environ.get("TRITON_CI_COMPILE_TRACE_DIR")
    phase = os.environ.get("TRITON_CI_CACHE_PHASE", "warmup-unit")
    workers = request.config.getoption("--warmup-workers")
    with process_pool_compile_warmup(workers=workers, directory=directory, phase=phase) as dispatcher:
        request.config._triton_warmup_dispatcher = dispatcher
        try:
            yield
        finally:
            del request.config._triton_warmup_dispatcher


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    directory = os.environ.get("TRITON_CI_COMPILE_TRACE_DIR")
    phase = _cache_phase_for_item(item)
    previous_listener = triton.knobs.compilation.listener
    if directory:
        triton.knobs.compilation.listener = CompilationTrace(directory, phase, item.nodeid)
        if (not item.config.getoption("--warmup-only") and item.get_closest_marker("enable_warmup") is not None
                and item.get_closest_marker("disable_warmup") is None):
            path = os.path.join(directory, f"{phase}-{os.getpid()}.tests")
            with open(path, "a", encoding="utf-8") as output:
                output.write(json.dumps({"phase": phase, "test": item.nodeid}, sort_keys=True) + "\n")

    dispatcher = getattr(item.config, "_triton_warmup_dispatcher", None)
    if dispatcher is not None:
        previous_test = dispatcher.current_test
        previous_phase = dispatcher.current_phase
        dispatcher.current_test = item.nodeid
        dispatcher.current_phase = phase

    try:
        result = yield
        if dispatcher is not None:
            dispatcher.record_test(item.nodeid, phase)
        return result
    finally:
        if dispatcher is not None:
            dispatcher.current_test = previous_test
            dispatcher.current_phase = previous_phase
        triton.knobs.compilation.listener = previous_listener
