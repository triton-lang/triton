import argparse
import hashlib
import json
import math
import os
import tempfile
import warnings
import weakref
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

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
def _coordinate_compiles():
    """Ensure concurrent warmup workers compile each exact specialization only once."""
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


@contextmanager
def process_pool_compile_warmup(*, workers, directory, phase):
    """Capture tests once and compile their exact launches in spawned workers."""
    from functools import partial

    from triton._compile_warmup_pool import ProcessPoolWarmupDispatcher

    previous_getitem = triton.KernelInterface.__getitem__
    previous_assert_close = torch.testing.assert_close
    dispatcher = ProcessPoolWarmupDispatcher(max_workers=workers, trace_directory=directory, phase=phase)
    dispatcher.current_test = None

    def getitem(kernel, grid):
        return partial(dispatcher.dispatch, kernel=kernel, grid=grid, test=dispatcher.current_test)

    triton.KernelInterface.__getitem__ = getitem
    torch.testing.assert_close = lambda *args, **kwargs: None
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Accessing the data pointer of FakeTensor.*",
            )
            with _FakeCudaTensorMode():
                try:
                    yield dispatcher
                finally:
                    dispatcher.finish()
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
            "compiler_worker": os.environ.get("TRITON_WARMUP_COMPILER_WORKER"),
            "cache_dir": triton.knobs.cache.dir,
        }
        with open(self.path, "a", encoding="utf-8") as output:
            output.write(json.dumps(record, sort_keys=True) + "\n")


@contextmanager
def _warmup_test_case(item):
    module_path = str(getattr(item.module, "__file__", ""))
    if module_path.endswith("/python/test/unit/language/test_warp_specialization.py"):
        previous_cublas = item.module.cublas
        item.module.cublas = SimpleNamespace(matmul=lambda *args, **kwargs: None)
        try:
            yield
        finally:
            item.module.cublas = previous_cublas
        return

    if module_path.endswith("/python/test/unit/language/test_core.py") and item.originalname == "test_scaled_dot":
        previous_isfinite = torch.Tensor.isfinite

        def always_finite(tensor):
            return SimpleNamespace(all=lambda: True)

        torch.Tensor.isfinite = always_finite
        try:
            yield
        finally:
            torch.Tensor.isfinite = previous_isfinite
        return

    if module_path.endswith("/python/test/unit/language/test_core.py") and item.originalname == "test_scan2d":
        if item.callspec.params["op"] != "cummax":
            yield
            return
        previous_cummax = torch.cummax

        def shape_only_cummax(tensor, *args, **kwargs):
            indices = item.module.np.zeros(tuple(tensor.shape), dtype=item.module.np.int64)
            return SimpleNamespace(indices=SimpleNamespace(numpy=lambda: indices))

        torch.cummax = shape_only_cummax
        try:
            yield
        finally:
            torch.cummax = previous_cummax
        return

    if (module_path.endswith("/python/test/unit/language/test_tensor_descriptor.py")
            and item.originalname == "test_tensor_descriptor_reduce"):
        kind = item.callspec.params["kind"]
        previous_reduce = item.module.REDUCE_OP[kind]
        item.module.REDUCE_OP[kind] = lambda input_tensor, output_tensor: output_tensor
        try:
            yield
        finally:
            item.module.REDUCE_OP[kind] = previous_reduce
        return

    if module_path.endswith("/python/test/gluon/test_lowerings.py") and item.originalname == "test_reduce_layouts":
        parameters = item.callspec.params
        previous_prod = torch.prod

        def concrete_warp_count(value, *args, **kwargs):
            if isinstance(value, torch.Tensor) and value.ndim == 1:
                shape = (parameters["M"], parameters["N"])
                return math.prod(item.module.ttgl._layouts.warps_per_cta(parameters["src_layout"], shape))
            return previous_prod(value, *args, **kwargs)

        torch.prod = concrete_warp_count
        try:
            yield
        finally:
            torch.prod = previous_prod
        return

    if module_path.endswith("/triton_kernels/tests/test_reduce.py") and item.originalname == "test_op":
        previous_allclose = torch.allclose
        torch.allclose = lambda *args, **kwargs: True
        try:
            yield
        finally:
            torch.allclose = previous_allclose
        return

    if (module_path.endswith("/python/examples/gluon/04-2cta-block-scale-matmul.py")
            and item.originalname == "test_mma_scaled_warp_specialized"):
        previous_random_quantized_tensor = item.module.random_quantized_tensor

        def random_quantized_tensor(rows, cols, format):
            vector_size = 16 if format == "nvfp4" else 32
            value_dtype = torch.float8_e4m3fn if format == "mxfp8" else torch.uint8
            scale_dtype = torch.float8_e4m3fn if format == "nvfp4" else torch.uint8
            values = torch.empty((rows, cols if format == "mxfp8" else cols // 2), dtype=value_dtype, device="cuda")
            scales = torch.empty((rows, cols // vector_size), dtype=scale_dtype, device="cuda")
            reference = torch.empty((rows, cols), dtype=torch.float32, device="cuda")
            return values, scales, reference

        item.module.random_quantized_tensor = random_quantized_tensor
        try:
            yield
        finally:
            item.module.random_quantized_tensor = previous_random_quantized_tensor
        return

    if module_path.endswith("/python/examples/gluon/05-moe-bmm1-fused-gather.py") and item.originalname in {
            "test_op",
            "test_op_consan",
            "test_op_fpsan",
    }:
        import triton_kernels.tensor_details.layout_details.blackwell_scale as blackwell_scale

        previous_assert_close = item.module.assert_close
        previous_is_fake = blackwell_scale.is_fake
        item.module.assert_close = lambda *args, **kwargs: None
        blackwell_scale.is_fake = lambda tensor: False
        try:
            yield
        finally:
            blackwell_scale.is_fake = previous_is_fake
            item.module.assert_close = previous_assert_close
        return

    if module_path.endswith("/python/test/unit/language/test_matmul.py") and item.originalname in {
            "test_block_scale_fp4",
            "test_mxfp8_mxfp4_matmul",
            "test_preshuffle_scale_mxfp_cdna4",
    }:
        fp4_type = item.module.MXFP4Tensor
        scale_type = item.module.MXScaleTensor
        previous_fp4_to = fp4_type.to
        previous_scale_from_float = scale_type._from_float
        previous_scale_random = scale_type.random
        previous_scale_to = scale_type.to

        def fp4_to(instance, dtype):
            return torch.empty(instance.data.shape, dtype=dtype, device=instance.data.device)

        def scale_from_float(instance, values):
            return torch.empty_like(values, dtype=torch.uint8)

        def scale_random(instance, low=None, high=None):
            instance.data = torch.empty(instance.size, dtype=torch.uint8, device=instance.device)
            return instance

        def scale_to(instance, dtype):
            return torch.empty(instance.data.shape, dtype=dtype, device=instance.data.device)

        fp4_type.to = fp4_to
        scale_type._from_float = scale_from_float
        scale_type.random = scale_random
        scale_type.to = scale_to
        try:
            yield
        finally:
            scale_type.to = previous_scale_to
            scale_type.random = previous_scale_random
            scale_type._from_float = previous_scale_from_float
            fp4_type.to = previous_fp4_to
        return

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
    parser.addoption(
        "--warmup-workers",
        type=int,
        default=1,
        help="number of spawned compiler processes used by --warmup-only",
    )
    parser.addoption(
        "--warmup-phase",
        action="append",
        default=[],
        metavar="PATH=PHASE",
        help="attribute tests under PATH to PHASE instead of TRITON_CI_CACHE_PHASE",
    )


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
        "/python/test/unit/language/test_standard.py": {
            "test_maximum_minium",
        },
    }
    for item in items:
        module_path = str(getattr(item.module, "__file__", ""))
        if (module_path.endswith("/python/test/gluon/test_consan.py")
                and item.originalname == "test_consan_uses_profile_scratch"):
            item.add_marker(pytest.mark.skip(reason="test intentionally compiles in an isolated temporary cache"))
            continue
        if module_path.endswith("/triton_kernels/tests/test_matmul.py") and item.originalname == "test_op":
            parameters = item.callspec.params
            if parameters["inner_expt_opt"] is not None and 0 in (
                    parameters["m"],
                    parameters["n"],
                    parameters["k"],
            ):
                item.add_marker(
                    pytest.mark.skip(reason="zero-sized inner-expert kernels specialize differently with FakeTensor"))
                continue
        if (module_path.endswith("/python/test/unit/language/test_tensor_descriptor.py")
                and item.originalname == "test_tensor_descriptor_reduce"):
            parameters = item.callspec.params
            dtype = getattr(item.module.tl, parameters["dtype_str"])
            native = item.module.is_cuda() and torch.cuda.get_device_capability()[0] >= 9
            supported_dtypes = (item.module.NATIVE_SUPPORTED_REDUCE_DTYPES[parameters["kind"]]
                                if native else item.module.FALLBACK_SUPPORTED_REDUCE_DTYPES[parameters["kind"]])
            if dtype not in supported_dtypes:
                item.add_marker(pytest.mark.skip(reason="tensor-descriptor atomic reduction cannot compile"))
                continue
        for suffix, originalnames in unsupported_specializations.items():
            if module_path.endswith(suffix) and item.originalname in originalnames:
                item.add_marker(pytest.mark.skip(reason="FakeTensor does not reproduce this kernel specialization"))
                break


@pytest.fixture(scope="session", autouse=True)
def compile_warmup(request):
    if not request.config.getoption("--warmup-only"):
        yield
        return
    directory = os.environ.get("TRITON_CI_COMPILE_TRACE_DIR")
    phase = os.environ.get("TRITON_CI_CACHE_PHASE", "unclassified")
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
    dispatcher = getattr(item.config, "_triton_warmup_dispatcher", None)
    previous_test = None
    previous_phase = None
    if dispatcher is not None:
        previous_test = dispatcher.current_test
        previous_phase = dispatcher.current_phase
        dispatcher.current_test = item.nodeid
        dispatcher.current_phase = phase
        dispatcher.record_test(item.nodeid, phase)
    try:
        if item.config.getoption("--warmup-only"):
            with _warmup_test_case(item):
                try:
                    return (yield)
                # Warmup is best-effort; the runtime test and exact per-test
                # cache audit remain authoritative. FakeTensor can fail on
                # data-dependent checks after compiling useful device kernels.
                except (Exception, pytest.fail.Exception) as error:
                    if os.environ.get("TRITON_WARMUP_DEBUG"):
                        import traceback

                        warnings.warn(
                            f"Warmup stopped early for {item.nodeid}: {type(error).__name__}: {error}\n"
                            f"{traceback.format_exc()}",
                            stacklevel=2,
                        )
                    return
        return (yield)
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
                        if not line.strip():
                            continue
                        attempted = json.loads(line)
                        attempted_tests.setdefault(attempted["phase"], set()).add(attempted["test"])

    warmup_records = [record for record in records if record["phase"].startswith("warmup-")]
    all_warmed_hashes = {record["hash"] for record in warmup_records}
    all_warmed_tests = {record.get("test")
                        for record in warmup_records
                        if record.get("test") is not None} | set().union(
                            *(tests for name, tests in attempted_tests.items() if name.startswith("warmup-")))
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
        if not is_warmup_phase:
            warmed_tests.update(attempted_tests.get(f"warmup-{name}", set()))
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
    parser.add_argument("--phase", action="append")
    parser.add_argument("--require-warmed-hits", action="store_true")
    parser.add_argument("--require-complete-warmup", action="store_true")
    parser.add_argument("--directory", default=os.environ.get("TRITON_CI_COMPILE_TRACE_DIR"))
    args = parser.parse_args()
    if args.directory is None:
        parser.error("set TRITON_CI_COMPILE_TRACE_DIR or pass --directory")
    complete_report = summarize_compile_trace(args.directory)
    if args.phase:
        reports = [{
            **complete_report,
            "phases": {phase: complete_report["phases"][phase]} if phase in complete_report["phases"] else {},
        } for phase in args.phase]
    else:
        reports = [complete_report]
    for report in reports:
        print(f"TRITON_CI_COMPILE_TRACE {json.dumps(report, sort_keys=True)}")
        if args.require_warmed_hits and not any(summary["warmed_hits"] for summary in report["phases"].values()):
            raise SystemExit("warmup did not produce any runtime disk-cache hits")
        if args.require_complete_warmup:
            _require_complete_warmup(report)


if __name__ == "__main__":
    _main()
