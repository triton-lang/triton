import argparse
import hashlib
import json
import os
import tempfile
import warnings
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
    _STORAGE_STRIDE = 1 << 40

    def __init__(self):
        from torch._subclasses.fake_tensor import FakeTensorMode

        self.fake_mode = FakeTensorMode(allow_fallback_kernels=False, allow_non_fake_inputs=True)
        self.storage_pointers = {}

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
            storage_id = id(storage)
            entry = self.storage_pointers.get(storage_id)
            if entry is None:
                pointer = (len(self.storage_pointers) + 1) * self._STORAGE_STRIDE + self._STORAGE_ALIGNMENT
                self.storage_pointers[storage_id] = (storage, pointer)
            else:
                _, pointer = entry
            return _SyntheticDataPtr(pointer + tensor.storage_offset() * tensor.element_size())
        return func(*args, **(kwargs or {}))


@contextmanager
def _coordinate_compiles():
    """Ensure xdist workers compile each exact specialization only once."""
    import fcntl
    from triton.runtime.jit import JITFunction

    trace_dir = os.environ.get("TRITON_CI_COMPILE_TRACE_DIR")
    run_uid = os.environ.get("PYTEST_XDIST_TESTRUNUID", "local")
    lock_dir = os.path.join(trace_dir or tempfile.gettempdir(), "triton-warmup-locks", run_uid)
    os.makedirs(lock_dir, mode=0o700, exist_ok=True)
    previous_do_compile = JITFunction._do_compile

    def do_compile(kernel, key, signature, device, constexprs, options, attrs, warmup):
        if not warmup:
            return previous_do_compile(kernel, key, signature, device, constexprs, options, attrs, warmup)
        digest = hashlib.sha256(f"{kernel.cache_key}\0{key}".encode()).hexdigest()
        with open(os.path.join(lock_dir, f"{digest}.lock"), "a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                return previous_do_compile(kernel, key, signature, device, constexprs, options, attrs, warmup)
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    JITFunction._do_compile = do_compile
    try:
        yield
    finally:
        JITFunction._do_compile = previous_do_compile


@contextmanager
def compile_warmup_only():
    """Compile intercepted Triton launches without allocating or running on the GPU."""
    previous_getitem = triton.KernelInterface.__getitem__
    previous_assert_close = torch.testing.assert_close

    def getitem(kernel, grid):

        def warmup(*args, **kwargs):
            return kernel.warmup(*args, grid=grid, **kwargs)

        return warmup

    triton.KernelInterface.__getitem__ = getitem
    torch.testing.assert_close = lambda *args, **kwargs: None
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Accessing the data pointer of FakeTensor.*",
            )
            with _FakeCudaTensorMode(), _coordinate_compiles():
                yield
    finally:
        torch.testing.assert_close = previous_assert_close
        triton.KernelInterface.__getitem__ = previous_getitem


class CompilationTrace:

    def __init__(self, directory, phase):
        self.directory = directory
        self.phase = phase
        os.makedirs(directory, exist_ok=True)
        self.path = os.path.join(directory, f"{phase}-{os.getpid()}.jsonl")

    def __call__(self, *, src, metadata, metadata_group, times, cache_hit):
        function = getattr(src, "fn", None)
        kernel = getattr(function, "_fn_name", src.name)
        record = {
            "phase": self.phase,
            "kernel": kernel,
            "hash": metadata["hash"],
            "source_hash": src.hash(),
            "cache_hit": cache_hit,
            "duration_us": times.total,
            "worker": os.environ.get("PYTEST_XDIST_WORKER", "main"),
            "cache_dir": triton.knobs.cache.dir,
        }
        with open(self.path, "a", encoding="utf-8") as output:
            output.write(json.dumps(record, sort_keys=True) + "\n")


class _WarmupComplete(Exception):
    """Stop a warmup test once its production kernels have been compiled."""


def _finish_before_reference(*args, **kwargs):
    raise _WarmupComplete()


@contextmanager
def _warmup_test_case(item):
    module_path = str(getattr(item.module, "__file__", ""))
    if not module_path.endswith("/triton_kernels/tests/test_matmul.py") or item.originalname != "test_op":
        yield
        return

    parameters = getattr(getattr(item, "callspec", None), "params", {})
    if parameters.get("inner_expt_opt") is not None:
        pytest.skip("warmup cannot infer data-dependent ragged padding")

    import triton_kernels.testing as testing

    previous_alloc_rand = testing.alloc_rand
    previous_make_slice_sizes = testing.make_slice_sizes
    previous_matmul_torch = item.module.matmul_torch

    def alloc_rand(shape, device, dtype, requires_grad=False):
        return torch.empty(shape, device=device, dtype=dtype, requires_grad=requires_grad)

    def make_slice_sizes(n_slices, total_size, device="cuda"):
        return torch.empty((max(n_slices, 0), ), dtype=torch.int32, device=device)

    testing.alloc_rand = alloc_rand
    testing.make_slice_sizes = make_slice_sizes
    item.module.matmul_torch = _finish_before_reference
    try:
        yield
    finally:
        item.module.matmul_torch = previous_matmul_torch
        testing.make_slice_sizes = previous_make_slice_sizes
        testing.alloc_rand = previous_alloc_rand


def pytest_addoption(parser):
    parser.addoption("--warmup-only", action="store_true", help="compile Triton launches without executing GPU kernels")


@pytest.fixture(scope="session", autouse=True)
def compile_warmup(request):
    if not request.config.getoption("--warmup-only"):
        yield
        return
    with compile_warmup_only():
        yield


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    directory = os.environ.get("TRITON_CI_COMPILE_TRACE_DIR")
    phase = os.environ.get("TRITON_CI_CACHE_PHASE", "unclassified")
    previous_listener = triton.knobs.compilation.listener
    if directory:
        triton.knobs.compilation.listener = CompilationTrace(directory, phase)
    try:
        if item.config.getoption("--warmup-only"):
            with _warmup_test_case(item):
                try:
                    return (yield)
                except _WarmupComplete:
                    return
        return (yield)
    finally:
        triton.knobs.compilation.listener = previous_listener


def summarize_compile_trace(directory, phase=None):
    records = []
    if os.path.isdir(directory):
        for name in sorted(os.listdir(directory)):
            if not name.endswith(".jsonl"):
                continue
            with open(os.path.join(directory, name), encoding="utf-8") as source:
                records.extend(json.loads(line) for line in source if line.strip())

    warmed_hashes = {record["hash"] for record in records if record["phase"].startswith("warmup-")}
    phases = sorted({record["phase"] for record in records})
    summaries = {}
    for name in phases:
        if phase is not None and name != phase:
            continue
        events = [record for record in records if record["phase"] == name]
        misses = [record for record in events if not record["cache_hit"]]
        summaries[name] = {
            "events": len(events),
            "disk_hits": sum(record["cache_hit"] for record in events),
            "disk_misses": len(misses),
            "warmed_hits": sum(record["cache_hit"] and record["hash"] in warmed_hashes for record in events),
            "warmed_misses": sum(not record["cache_hit"] and record["hash"] in warmed_hashes for record in events),
            "compile_seconds": round(sum(record["duration_us"] for record in misses) / 1_000_000, 3),
        }
    return {"phases": summaries, "warmup_hashes": len(warmed_hashes)}


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["report"])
    parser.add_argument("--phase")
    parser.add_argument("--require-warmed-hits", action="store_true")
    parser.add_argument("--directory", default=os.environ.get("TRITON_CI_COMPILE_TRACE_DIR"))
    args = parser.parse_args()
    if args.directory is None:
        parser.error("set TRITON_CI_COMPILE_TRACE_DIR or pass --directory")
    report = summarize_compile_trace(args.directory, args.phase)
    print(f"TRITON_CI_COMPILE_TRACE {json.dumps(report, sort_keys=True)}")
    if args.require_warmed_hits and not any(summary["warmed_hits"] for summary in report["phases"].values()):
        raise SystemExit("warmup did not produce any runtime disk-cache hits")


if __name__ == "__main__":
    _main()
