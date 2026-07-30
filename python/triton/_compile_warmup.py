import argparse
import hashlib
import json
import os
import tempfile
import warnings
from contextlib import contextmanager
from dataclasses import replace

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
            "cache_dir": triton.knobs.cache.dir,
        }
        with open(self.path, "a", encoding="utf-8") as output:
            output.write(json.dumps(record, sort_keys=True) + "\n")


@contextmanager
def _warmup_test_case(item):
    module_path = str(getattr(item.module, "__file__", ""))
    if module_path.endswith("/python/test/unit/language/test_core.py") and item.originalname in {
            "test_bin_op",
            "test_bitwise_op",
            "test_cast",
            "test_compare_op",
            "test_floordiv",
            "test_math_divide_op",
            "test_shift_op",
    }:
        previous_to_numpy = item.module.to_numpy
        previous_assertions = {
            name: getattr(item.module.np.testing, name)
            for name in ("assert_", "assert_allclose", "assert_array_equal", "assert_equal")
        }
        item.module.to_numpy = lambda tensor: item.module.np.empty(tuple(tensor.shape))
        for name in previous_assertions:
            setattr(item.module.np.testing, name, lambda *args, **kwargs: None)
        try:
            yield
        finally:
            item.module.to_numpy = previous_to_numpy
            for name, assertion in previous_assertions.items():
                setattr(item.module.np.testing, name, assertion)
        return

    if not module_path.endswith("/triton_kernels/tests/test_matmul.py") or item.originalname != "test_op":
        yield
        return

    parameters = getattr(getattr(item, "callspec", None), "params", {})
    if parameters["inner_expt_opt"] is not None and 0 in (
            parameters["m"],
            parameters["n"],
            parameters["k"],
    ):
        pytest.skip("zero-sized inner-expert kernels specialize differently with FakeTensor")

    import triton_kernels.testing as testing
    import triton_kernels.tensor_details.layout_details.blackwell_scale as blackwell_scale

    previous_alloc_rand = testing.alloc_rand
    previous_assert_close = testing.assert_close
    previous_make_slice_sizes = testing.make_slice_sizes
    previous_pad_ragged_tensor = testing.pad_ragged_tensor
    previous_module_assert_close = item.module.assert_close
    previous_matmul_torch = item.module.matmul_torch
    previous_is_fake = blackwell_scale.is_fake
    slice_sizes = {}

    def alloc_rand(shape, device, dtype, requires_grad=False):
        return torch.empty(shape, device=device, dtype=dtype, requires_grad=requires_grad)

    def make_slice_sizes(n_slices, total_size, device="cuda"):
        from torch._subclasses.fake_tensor import unset_fake_temporarily

        with unset_fake_temporarily():
            values = testing._make_slice_sizes_cpu(n_slices, total_size).tolist()
        result = torch.empty((max(n_slices, 0), ), dtype=torch.int32, device=device)
        slice_sizes[id(result)] = values
        return result

    def pad_ragged_tensor(tensor, metadata, hbm_swizzling, transpose):
        multiple = 128 if hbm_swizzling else 64
        dimension = 1 if transpose else 0
        shape = list(tensor.shape)
        values = slice_sizes[id(metadata.slice_sizes)]
        shape[dimension] = sum(triton.cdiv(value, multiple) * multiple for value in values)
        padded = torch.empty(shape, dtype=tensor.dtype, device=tensor.device)
        metadata = replace(
            metadata,
            slice_offs=metadata.block_offs(multiple) * multiple,
            slice_sizes_divisibility=multiple,
        )
        return padded, metadata

    def matmul_torch(*args, **kwargs):
        try:
            # Compile reference-path device helpers until FakeTensor reaches a
            # data-dependent operation. The production matmul has already been
            # warmed, so the shape-only result below can finish the test path.
            return previous_matmul_torch(*args, **kwargs)
        except Exception:
            output_dtype = item.module.DType(parameters["output_dtype_str"] or parameters["act_dtype_str"])
            act_dtype = item.module.DType(parameters["act_dtype_str"])
            reference_dtype = torch.float32 if output_dtype.is_nvfp4 or parameters["inner_expt_opt"] is not None else (
                torch.bfloat16 if act_dtype.has_mx_scale else act_dtype.torch_dtype)
            shape = ((parameters["n_slices"], )
                     if parameters["mode"] == "batched" or parameters["inner_expt_opt"] is not None else tuple())
            shape += (parameters["m"], parameters["n"])
            return torch.empty(shape, dtype=reference_dtype, device="cuda")

    testing.alloc_rand = alloc_rand
    testing.assert_close = lambda *args, **kwargs: None
    testing.make_slice_sizes = make_slice_sizes
    testing.pad_ragged_tensor = pad_ragged_tensor
    blackwell_scale.is_fake = lambda tensor: False
    item.module.assert_close = lambda *args, **kwargs: None
    item.module.matmul_torch = matmul_torch
    try:
        yield
    finally:
        blackwell_scale.is_fake = previous_is_fake
        item.module.matmul_torch = previous_matmul_torch
        item.module.assert_close = previous_module_assert_close
        testing.pad_ragged_tensor = previous_pad_ragged_tensor
        testing.make_slice_sizes = previous_make_slice_sizes
        testing.assert_close = previous_assert_close
        testing.alloc_rand = previous_alloc_rand


def pytest_addoption(parser):
    parser.addoption("--warmup-only", action="store_true", help="compile Triton launches without executing GPU kernels")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--warmup-only"):
        return
    unsupported_specializations = {
        "/python/test/unit/language/test_core.py": {
            "test_argmax_argmin_tie_break_fast_with_nan",
            "test_argmax_argmin_with_nan",
            "test_atomic_cas",
            "test_constexpr_if_return",
            "test_full",
            "test_globaltimer",
            "test_if_return",
            "test_max_min_with_nan",
            "test_num_threads",
            "test_short_circuiting",
            "test_smid",
            "test_sum_dtype",
            "test_unsplat",
            "test_where",
            "test_where_broadcast",
        },
        "/python/test/unit/language/test_matmul.py": {
            "test_block_scale_fp4",
            "test_mxfp8_mxfp4_matmul",
        },
        "/python/test/unit/language/test_standard.py": {
            "test_maximum_minium",
        },
    }
    for item in items:
        module_path = str(getattr(item.module, "__file__", ""))
        for suffix, originalnames in unsupported_specializations.items():
            if module_path.endswith(suffix) and item.originalname in originalnames:
                item.add_marker(pytest.mark.skip(reason="FakeTensor does not reproduce this kernel specialization"))
                break


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
        triton.knobs.compilation.listener = CompilationTrace(directory, phase, item.nodeid)
    try:
        if item.config.getoption("--warmup-only"):
            with _warmup_test_case(item):
                try:
                    return (yield)
                # Warmup is best-effort; the runtime test and exact per-test
                # cache audit remain authoritative. FakeTensor can fail on
                # data-dependent checks after compiling useful device kernels.
                except (Exception, pytest.fail.Exception):
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

    warmup_records = [record for record in records if record["phase"].startswith("warmup-")]
    all_warmed_hashes = {record["hash"] for record in warmup_records}
    all_warmed_tests = {record.get("test") for record in warmup_records if record.get("test") is not None}
    phases = sorted({record["phase"] for record in records})
    summaries = {}
    for name in phases:
        if phase is not None and name != phase:
            continue
        events = [record for record in records if record["phase"] == name]
        misses = [record for record in events if not record["cache_hit"]]
        is_warmup_phase = name.startswith("warmup-")
        matching_warmup_records = [record for record in warmup_records
                                   if record["phase"] == f"warmup-{name}"] if not is_warmup_phase else []
        warmed_hashes = {record["hash"] for record in matching_warmup_records}
        warmed_tests = {record.get("test") for record in matching_warmup_records if record.get("test") is not None}
        warmed_test_events = [] if is_warmup_phase else [
            record for record in events if record.get("test") in warmed_tests
        ]
        warmed_test_hits = [
            record for record in warmed_test_events if record["cache_hit"] and record["hash"] in warmed_hashes
        ]
        incomplete_tests = {
            record["test"]
            for record in warmed_test_events
            if not record["cache_hit"] or record["hash"] not in warmed_hashes
        }
        used_warmup_hashes = {record["hash"] for record in events if record["hash"] in warmed_hashes}
        summaries[name] = {
            "events": len(events),
            "disk_hits": sum(record["cache_hit"] for record in events),
            "disk_misses": len(misses),
            "warmed_hits": sum(record["cache_hit"] and record["hash"] in warmed_hashes for record in events),
            "warmed_misses": sum(not record["cache_hit"] and record["hash"] in warmed_hashes for record in events),
            "warmed_test_events": len(warmed_test_events),
            "warmed_test_hits": len(warmed_test_hits),
            "warmed_test_misses": len(warmed_test_events) - len(warmed_test_hits),
            "incomplete_warmed_test_count": len(incomplete_tests),
            "incomplete_warmed_tests": sorted(incomplete_tests)[:20],
            "unused_warmup_hashes": len(warmed_hashes - used_warmup_hashes),
            "compile_seconds": round(sum(record["duration_us"] for record in misses) / 1_000_000, 3),
        }
    return {
        "phases": summaries,
        "warmup_hashes": len(all_warmed_hashes),
        "warmup_tests": len(all_warmed_tests),
    }


def _require_complete_warmup(report):
    if not report["phases"] or not any(summary["warmed_test_events"] for summary in report["phases"].values()):
        raise SystemExit("warmup did not produce any compiler events in warmed runtime tests")
    incomplete = {
        phase: summary["incomplete_warmed_tests"]
        for phase, summary in report["phases"].items()
        if summary["warmed_test_misses"]
    }
    if incomplete:
        raise SystemExit(f"warmed runtime tests were not complete cache hits: {json.dumps(incomplete, sort_keys=True)}")


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["report"])
    parser.add_argument("--phase")
    parser.add_argument("--require-warmed-hits", action="store_true")
    parser.add_argument("--require-complete-warmup", action="store_true")
    parser.add_argument("--directory", default=os.environ.get("TRITON_CI_COMPILE_TRACE_DIR"))
    args = parser.parse_args()
    if args.directory is None:
        parser.error("set TRITON_CI_COMPILE_TRACE_DIR or pass --directory")
    report = summarize_compile_trace(args.directory, args.phase)
    print(f"TRITON_CI_COMPILE_TRACE {json.dumps(report, sort_keys=True)}")
    if args.require_warmed_hits and not any(summary["warmed_hits"] for summary in report["phases"].values()):
        raise SystemExit("warmup did not produce any runtime disk-cache hits")
    if args.require_complete_warmup:
        _require_complete_warmup(report)


if __name__ == "__main__":
    _main()
