import argparse
import json
import os
import warnings
import weakref
from contextlib import contextmanager

import pytest
import torch
from torch.overrides import TorchFunctionMode

import triton


class _SyntheticDataPtr(int):
    pass


class _FakeCudaTensorMode(TorchFunctionMode):
    """Preserve CUDA allocation alignment and view offsets in fake pointers."""

    _STORAGE_ALIGNMENT = 256
    _STORAGE_STRIDE = 1 << 20

    def __init__(self):
        from torch._subclasses.fake_tensor import FakeTensorMode

        self.fake_mode = FakeTensorMode(allow_fallback_kernels=False, allow_non_fake_inputs=True)
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
        if getattr(func, "__name__", "") == "data_ptr":
            tensor = args[0]
            storage = tensor.untyped_storage()
            pointer = self.storage_pointers.get(storage)
            if pointer is None:
                pointer = self.next_storage_pointer
                self.next_storage_pointer += self._STORAGE_STRIDE
                self.storage_pointers[storage] = pointer
            return _SyntheticDataPtr(pointer + tensor.storage_offset() * tensor.element_size())
        return func(*args, **(kwargs or {}))


@contextmanager
def compile_warmup_only(dispatcher=None):
    """Capture launch specializations without GPU allocations or kernel execution."""

    def dispatch(kernel, grid, *args, **kwargs):
        if dispatcher is None:
            return kernel.warmup(*args, grid=grid, **kwargs)
        return dispatcher.dispatch(*args, kernel=kernel, grid=grid, test=dispatcher.current_test, **kwargs)

    with triton.knobs.runtime.scope():
        triton.knobs.runtime.compile_warmup = True
        triton.knobs.runtime.launch_dispatcher = dispatch
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Accessing the data pointer of FakeTensor.*")
            with _FakeCudaTensorMode():
                yield


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


class CompilationTrace:

    def __init__(self, directory, phase, test=None):
        self.directory = directory
        self.phase = phase
        self.test = test
        os.makedirs(directory, exist_ok=True)
        self.path = os.path.join(directory, f"{phase}-{os.getpid()}.jsonl")

    def __call__(self, *, src, metadata, metadata_group, times, cache_hit):
        function = getattr(src, "fn", None)
        kernel = getattr(function, "_fn_name", src.name)
        record = {
            "phase": self.phase,
            "test": self.test,
            "kernel": kernel,
            "hash": metadata["hash"],
            "source_hash": src.hash(),
            "cache_hit": cache_hit,
            "duration_us": times.total,
            "worker": os.environ.get("PYTEST_XDIST_WORKER", "main"),
            "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "compiler_worker": os.environ.get("TRITON_WARMUP_COMPILER_WORKER"),
            "cache_dir": triton.knobs.cache.dir,
        }
        with open(self.path, "a", encoding="utf-8") as output:
            output.write(json.dumps(record, sort_keys=True) + "\n")


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

    visible = os.environ.get("TRITON_TEST_VISIBLE_GPUS", os.environ.get("CUDA_VISIBLE_DEVICES"))
    if visible:
        devices = [device.strip() for device in visible.split(",") if device.strip()]
    else:
        devices = [str(index) for index in range(int(requested))]

    for index, spec in enumerate(specs):
        spec.env["CUDA_VISIBLE_DEVICES"] = devices[index % int(requested)]


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
    capability = None
    for item in items:
        marker = item.get_closest_marker("enable_warmup")
        excluded = item.get_closest_marker("disable_warmup")
        if marker is None or excluded is not None:
            deselected.append(item)
            continue
        minimum = marker.kwargs.get("min_capability")
        if minimum is not None:
            if capability is None:
                capability = torch.cuda.get_device_capability()[0]
            if capability < minimum:
                deselected.append(item)
                continue
        selected.append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
    items[:] = selected


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


def summarize_compile_trace(directory, phase=None):
    records = []
    attempted_tests = {}
    if os.path.isdir(directory):
        for name in sorted(os.listdir(directory)):
            path = os.path.join(directory, name)
            if name.endswith(".jsonl"):
                with open(path, encoding="utf-8") as source:
                    records.extend(json.loads(line) for line in source if line.strip())
            elif name.endswith(".tests"):
                with open(path, encoding="utf-8") as source:
                    for line in source:
                        if line.strip():
                            attempted = json.loads(line)
                            attempted_tests.setdefault(attempted["phase"], set()).add(attempted["test"])

    warmup_records = [record for record in records if record["phase"].startswith("warmup-")]
    all_warmed_hashes = {record["hash"] for record in warmup_records}
    all_warmed_entries = {(record["cache_dir"], record["hash"]) for record in warmup_records}
    all_warmed_tests = {record["test"] for record in warmup_records if record.get("test") is not None}
    for name, tests in attempted_tests.items():
        if name.startswith("warmup-"):
            all_warmed_tests.update(tests)

    runtime_records = [record for record in records if not record["phase"].startswith("warmup-")]
    globally_used_hashes = {
        record["hash"]
        for record in runtime_records
        if record["cache_hit"] and (record["cache_dir"], record["hash"]) in all_warmed_entries
    }
    summaries = {}
    phase_names = {record["phase"] for record in records} | set(attempted_tests)
    phase_names.update(name.removeprefix("warmup-") for name in tuple(phase_names) if name.startswith("warmup-"))
    for name in sorted(phase_names):
        if phase is not None and name != phase:
            continue
        events = [record for record in records if record["phase"] == name]
        misses = [record for record in events if not record["cache_hit"]]
        is_warmup_phase = name.startswith("warmup-")
        matching = [] if is_warmup_phase else [
            record for record in warmup_records if record["phase"] == f"warmup-{name}"
        ]
        warmed_hashes = {record["hash"] for record in matching}
        warmed_tests = {record["test"] for record in matching if record.get("test") is not None}
        if not is_warmup_phase:
            warmed_tests.update(attempted_tests.get(f"warmup-{name}", set()))
        warmed_test_events = [] if is_warmup_phase else [
            record for record in events if record.get("test") in warmed_tests
        ]
        warmed_test_hits = [
            record for record in warmed_test_events
            if record["cache_hit"] and (record["cache_dir"], record["hash"]) in all_warmed_entries
        ]
        incomplete_tests = {
            record["test"]
            for record in warmed_test_events
            if not record["cache_hit"] or (record["cache_dir"], record["hash"]) not in all_warmed_entries
        }
        runtime_tests = {record.get("test") for record in events} | attempted_tests.get(name, set())
        missing_tests = warmed_tests - runtime_tests if not is_warmup_phase else set()
        runtime_warmed_entries = set() if is_warmup_phase else all_warmed_entries
        warmed_events = [record for record in events if (record["cache_dir"], record["hash"]) in runtime_warmed_entries]
        warmed_hits = sum(record["cache_hit"] for record in warmed_events)
        summaries[name] = {
            "events": len(events),
            "disk_hits": sum(record["cache_hit"] for record in events),
            "disk_misses": len(misses),
            "warmed_hits": warmed_hits,
            "warmed_misses": len(warmed_events) - warmed_hits,
            "warmed_test_events": len(warmed_test_events),
            "warmed_test_hits": len(warmed_test_hits),
            "warmed_test_misses": len(warmed_test_events) - len(warmed_test_hits),
            "incomplete_warmed_test_count": len(incomplete_tests),
            "incomplete_warmed_tests": sorted(incomplete_tests)[:20],
            "missing_warmed_test_count": len(missing_tests),
            "missing_warmed_tests": sorted(missing_tests)[:20],
            "unused_warmup_hashes": len(warmed_hashes - globally_used_hashes),
            "compile_seconds": round(sum(record["duration_us"] for record in misses) / 1_000_000, 3),
        }

    return {
        "phases": summaries,
        "warmup_hashes": len(all_warmed_hashes),
        "warmup_tests": len(all_warmed_tests),
        "unused_warmup_hashes": len(all_warmed_hashes - globally_used_hashes),
    }


def _require_complete_warmup(report):
    runtime = {name: summary for name, summary in report["phases"].items() if not name.startswith("warmup-")}
    if not runtime or not any(summary["warmed_test_events"] for summary in runtime.values()):
        raise SystemExit("warmup did not produce any compiler events in warmed runtime tests")

    incomplete = {
        name: summary["incomplete_warmed_tests"]
        for name, summary in runtime.items()
        if summary["warmed_test_misses"]
    }
    if incomplete:
        raise SystemExit(f"warmed runtime tests were not complete cache hits: {json.dumps(incomplete, sort_keys=True)}")

    missing = {
        name: summary["missing_warmed_tests"]
        for name, summary in runtime.items()
        if summary["missing_warmed_test_count"]
    }
    if missing:
        raise SystemExit(f"marked warmup tests were not executed: {json.dumps(missing, sort_keys=True)}")

    if report.get("unused_warmup_hashes", 0):
        raise SystemExit(f"warmup produced {report['unused_warmup_hashes']} unused specializations")


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["report"])
    parser.add_argument("--phase", action="append")
    parser.add_argument("--require-warmed-hits", action="store_true")
    parser.add_argument("--require-complete-warmup", action="store_true")
    parser.add_argument("--directory", default=os.environ.get("TRITON_CI_COMPILE_TRACE_DIR"))
    args = parser.parse_args()
    if args.directory is None:
        parser.error("set TRITON_CI_COMPILE_TRACE_DIR or pass --directory")

    complete = summarize_compile_trace(args.directory)
    reports = [{**complete, "phases": {name: complete["phases"][name]} if name in complete["phases"] else {}}
               for name in args.phase] if args.phase else [complete]
    for report in reports:
        print(f"TRITON_CI_COMPILE_TRACE {json.dumps(report, sort_keys=True)}")
        if args.require_warmed_hits and not any(summary["warmed_hits"] for summary in report["phases"].values()):
            raise SystemExit("warmup did not produce any runtime disk-cache hits")
        if args.require_complete_warmup:
            _require_complete_warmup(report)


if __name__ == "__main__":
    _main()
