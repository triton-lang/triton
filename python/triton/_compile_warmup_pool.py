import ast
import atexit
import hashlib
import importlib
import io
import json
import linecache
import multiprocessing as mp
from multiprocessing.connection import Client, Listener
import os
import statistics
import sys
import tempfile
import threading
import time
import types
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass

import cloudpickle
from cloudpickle.cloudpickle import CloudPickler, _extract_code_globals

import triton


class _UnsupportedPreloadError(Exception):
    pass


class _CompilationFailedError(_UnsupportedPreloadError):
    pass


@contextmanager
def _cache_invalidating_environment(environment):
    from triton._C.libtriton import get_cache_invalidating_env_vars

    previous = get_cache_invalidating_env_vars()
    changed_names = previous.keys() | environment.keys()
    previous_raw = {name: os.environ.get(name) for name in changed_names}
    try:
        for name in changed_names:
            if name in environment:
                os.environ[name] = environment[name]
            else:
                os.environ.pop(name, None)
        yield
    finally:
        for name, value in previous_raw.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _load_jit_callable(module_name, name):
    from triton.runtime.jit import JITCallable

    ref = importlib.import_module(module_name)
    for part in name.split("."):
        ref = getattr(ref, part)
    assert isinstance(ref, JITCallable)
    return ref


def _jit_callable_import_path(callable_):
    function = getattr(callable_, "fn", None)
    if not isinstance(function, types.FunctionType):
        return None
    module_name = function.__module__
    name = function.__qualname__
    module = sys.modules.get(module_name)
    module_file = getattr(module, "__file__", None)
    if module_file is not None and not os.path.isfile(module_file):
        return None
    try:
        imported = _load_jit_callable(module_name, name)
    except (ImportError, AttributeError, AssertionError):
        return None
    if imported is callable_:
        return module_name, name
    return None


@dataclass(frozen=True)
class _CodeGenFunction:
    jit_function: object


def _iter_argument_expressions(function):
    target = next((node for node in ast.parse(function.src).body
                   if isinstance(node, ast.FunctionDef) and node.name == function.__name__), None)
    if target is None:
        raise ValueError(f"Cannot find function {function.__name__} in source code of {function}")
    args = target.args
    for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg):
        if arg is not None and arg.annotation is not None:
            yield arg.annotation
    for default in args.defaults + args.kw_defaults:
        if default is not None:
            yield default


def _restore_dynamic_source(python_function, src):
    if isinstance(python_function, types.FunctionType):
        filename = python_function.__code__.co_filename
        if not os.path.isfile(filename):
            source = "\n" * (python_function.__code__.co_firstlineno - 1) + src
            if not source.endswith("\n"):
                source += "\n"
            linecache.cache[filename] = (
                len(source),
                None,
                source.splitlines(keepends=True),
                filename,
            )


def _make_dynamic_jit_callable(callable_type, args, src, starting_line_number):
    python_function = args[0]
    if isinstance(python_function, _CodeGenFunction):
        python_function = python_function.jit_function.fn
    _restore_dynamic_source(python_function, src)
    function = callable_type(*args)
    function._unsafe_update_src(src)
    function.starting_line_number = starting_line_number
    return function


class _JITFunctionPickler(CloudPickler):

    def _reduce_jit_codegen_fn(self, codegen_function):
        jit_function = codegen_function.jit_function
        function = jit_function.fn
        result = self._function_reduce(function)
        globals_ = result[2][1]["__globals__"]
        capture_scope = jit_function.get_capture_scope()
        for node in ast.walk(ast.parse(jit_function.src)):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in capture_scope:
                globals_[node.id] = capture_scope[node.id]
        for expression in _iter_argument_expressions(jit_function):
            if isinstance(expression, ast.Constant) and isinstance(expression.value, str):
                expression = ast.parse(expression.value, mode="eval").body
            for node in ast.walk(expression):
                if (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in function.__globals__):
                    globals_[node.id] = function.__globals__[node.id]
        if function.__code__.co_name != function.__name__:
            code = compile(jit_function.src, function.__name__, "exec")
            namespace = {**globals_, **function.__globals__}
            exec(code, namespace)
            global_names = _extract_code_globals(namespace[function.__name__].__code__)
            for global_name in global_names:
                if global_name in function.__globals__:
                    globals_[global_name] = function.__globals__[global_name]
        return result

    def reducer_override(self, obj):
        from triton.runtime.jit import BoundConstexprFunction, ConstexprFunction, JITCallable, JITFunction

        if isinstance(obj, _CodeGenFunction):
            return self._reduce_jit_codegen_fn(obj)
        if not isinstance(obj, JITCallable):
            return super().reducer_override(obj)
        if isinstance(obj, BoundConstexprFunction):
            return BoundConstexprFunction, (obj.__self__, obj.__func__)
        if not isinstance(obj, JITFunction | ConstexprFunction):
            raise ValueError(f"Don't know how to pickle JITCallable subclass: {obj}")
        if (import_path := _jit_callable_import_path(obj)) is not None:
            return _load_jit_callable, import_path
        args = (obj.fn, ) if isinstance(obj, ConstexprFunction) else (
            _CodeGenFunction(obj),
            obj.version,
            obj.do_not_specialize,
            obj.do_not_specialize_on_alignment,
            obj.debug,
            obj.noinline,
            obj._repr,
            obj.launch_metadata,
        )
        return _make_dynamic_jit_callable, (type(obj), args, obj.src, obj.starting_line_number)


def _jit_dumps(obj):
    function = getattr(obj, "fn", None)
    module = sys.modules.get(function.__module__) if isinstance(function, types.FunctionType) else None
    module_file = getattr(module, "__file__", None)
    register_by_value = (module is not None and module_file is not None and not os.path.isfile(module_file)
                         and module.__name__ not in cloudpickle.list_registry_pickle_by_value())
    if register_by_value:
        cloudpickle.register_pickle_by_value(module)
    buffer = io.BytesIO()
    try:
        _JITFunctionPickler(buffer).dump(obj)
        return buffer.getvalue()
    finally:
        if register_by_value:
            cloudpickle.unregister_pickle_by_value(module)


def _preload_with_compile_context(function, name, key, serialized_target, serialized_options, signature, constants,
                                  attrs):
    from triton.runtime.driver import driver

    if name != function._fn_name:
        raise RuntimeError(f"Specialization data is for {name} but trying to preload for {function._fn_name}")
    active_driver = driver.active
    device = active_driver.get_current_device()
    _, _, target, backend, _ = function.device_caches[device]
    if target.__dict__ != serialized_target:
        raise RuntimeError(f"Specialization data is for {serialized_target} but trying to preload for {target}")
    options = backend.parse_options(serialized_options)
    return function._do_compile(
        key,
        signature,
        device,
        constants,
        options,
        attrs,
        warmup=True,
    )


def _warmup_kernel(kernel, args, grid, kwargs):
    from triton.runtime.autotuner import Autotuner

    if not isinstance(kernel, Autotuner):
        return kernel.warmup(*args, grid=grid, **kwargs)

    kernel.nargs = dict(zip(kernel.arg_names, args))
    try:
        results = []
        for config in kernel.prune_configs(kwargs):
            config_kwargs = config.all_kwargs()
            if config.pre_hook is not None:
                config.pre_hook({**kernel.nargs, **kwargs, **config_kwargs})
            results.append(kernel.fn.warmup(*args, grid=grid, **kwargs, **config_kwargs))
        return results
    finally:
        kernel.nargs = None


_SHUTDOWN_STDERR_SILENCED = False


def _silence_worker_shutdown_stderr():
    descriptor = os.open(os.devnull, os.O_WRONLY)
    os.dup2(descriptor, 2)
    os.close(descriptor)


def _silence_worker_shutdown_diagnostics():
    global _SHUTDOWN_STDERR_SILENCED
    if _SHUTDOWN_STDERR_SILENCED:
        return
    # Compilation failures are returned through the Future. Suppress only
    # orderly interpreter-shutdown diagnostics: libtriton's nanobind teardown
    # otherwise emits the same known leak report from every ephemeral worker.
    atexit.register(_silence_worker_shutdown_stderr)
    _SHUTDOWN_STDERR_SILENCED = True


def _preload_worker(module_name, qualified_name, fn_bytes, compile_context_bytes, kernel_repr, instrumentation_mode,
                    environment, cache_dir, trace_directory, phase, test, capture_worker):
    """Load one JIT function and populate the shared disk cache."""
    previous_capture_worker = os.environ.get("PYTEST_XDIST_WORKER")
    os.environ["PYTEST_XDIST_WORKER"] = capture_worker
    _silence_worker_shutdown_diagnostics()
    worker = os.getpid()
    os.environ["TRITON_WARMUP_COMPILER_WORKER"] = str(worker)
    try:
        with (
                triton.knobs.compilation.scope(),
                _cache_invalidating_environment(environment),
                triton.knobs.cache.scope(),
        ):
            triton.knobs.compilation.instrumentation_mode = instrumentation_mode
            triton.knobs.cache.dir = cache_dir
            if module_name is not None and qualified_name is not None:
                function = _load_jit_callable(module_name, qualified_name)
            elif fn_bytes is not None:
                function = cloudpickle.loads(fn_bytes)
            else:
                raise AssertionError("missing JIT function import and serialized payload")
            compile_context = cloudpickle.loads(compile_context_bytes)
            if trace_directory is not None:
                from triton._compile_warmup import CompilationTrace

                triton.knobs.compilation.listener = CompilationTrace(trace_directory, phase, test)
            start = time.monotonic()
            _preload_with_compile_context(function, *compile_context)
            return worker, time.monotonic() - start
    except triton.compiler.errors.CompilationError as error:
        details = f"{type(error).__name__}: {error}"
        cause = error.__cause__
        while cause is not None:
            details += f"\nCaused by {type(cause).__name__}: {cause}"
            cause = cause.__cause__
        if os.environ.get("TRITON_WARMUP_DEBUG"):
            details += f"\nCompile context: {compile_context!r}"
        raise _CompilationFailedError(f"Compilation failed for {kernel_repr}: {details}") from error
    except Exception as error:
        raise _UnsupportedPreloadError(
            f"Failed to load or preload {kernel_repr}: {type(error).__name__}: {error}") from error
    finally:
        if previous_capture_worker is None:
            os.environ.pop("PYTEST_XDIST_WORKER", None)
        else:
            os.environ["PYTEST_XDIST_WORKER"] = previous_capture_worker


@dataclass(frozen=True)
class _CapturedPreload:
    module_name: str | None
    qualified_name: str | None
    fn_bytes: bytes | None
    compile_context_bytes: bytes
    environment: dict
    test: str | None

    def digest_payload(self):
        if self.module_name is not None and self.qualified_name is not None:
            function_payload = f"import\0{self.module_name}\0{self.qualified_name}".encode()
        else:
            assert self.fn_bytes is not None
            function_payload = b"cloudpickle\0" + self.fn_bytes
        environment_payload = json.dumps(self.environment, sort_keys=True).encode()
        return (function_payload + b"\0compile-context\0" + self.compile_context_bytes + b"\0environment\0" +
                environment_payload)


class ProcessPoolWarmupDispatcher:
    """Capture fake launches and compile them in a persistent process pool."""

    def __init__(self, *, max_workers, trace_directory, phase):
        if max_workers < 1:
            raise ValueError(f"max_workers must be >= 1, got {max_workers}")
        coordinator = os.environ.get("TRITON_WARMUP_COORDINATOR")
        self._connection = Client(coordinator, family="AF_UNIX") if coordinator else None
        self._executor = None
        if self._connection is None:
            context = mp.get_context("spawn")
            self._executor = ProcessPoolExecutor(max_workers=max_workers, mp_context=context)
        self._max_workers = max_workers
        self._cache_dir = triton.knobs.cache.dir
        self._trace_directory = trace_directory
        self._phase = phase
        self.current_phase = phase
        self._capture_lock = threading.Lock()
        self._pending = []
        self._first_submit = None
        self._attempted_tests = set()
        self._serialized_keys = set()
        self._serialized_functions = {}
        self._submitted_keys = set()

    def record_test(self, test, phase=None):
        phase = phase or self.current_phase
        attempted_test = (phase, test)
        if self._trace_directory is None or test is None or attempted_test in self._attempted_tests:
            return
        self._attempted_tests.add(attempted_test)
        os.makedirs(self._trace_directory, exist_ok=True)
        path = os.path.join(self._trace_directory, f"{phase}-{os.getpid()}.tests")
        with open(path, "a", encoding="utf-8") as output:
            output.write(json.dumps({"phase": phase, "test": test}, sort_keys=True) + "\n")

    def dispatch(self, *args, kernel, grid, test, **kwargs):
        from triton._C.libtriton import get_cache_invalidating_env_vars

        with self._capture_lock:
            instrumentation_mode = triton.knobs.compilation.instrumentation_mode
            environment = get_cache_invalidating_env_vars()
            captures = self._capture_preloads(
                instrumentation_mode,
                environment,
                kernel,
                args,
                grid,
                kwargs,
                test,
            )
            phase = self.current_phase
            self.record_test(test, phase)
            for capture in captures:
                digest = hashlib.sha256(instrumentation_mode.encode() + b"\0" + capture.digest_payload()).hexdigest()
                if digest in self._submitted_keys:
                    continue
                task = (
                    capture.module_name,
                    capture.qualified_name,
                    capture.fn_bytes,
                    capture.compile_context_bytes,
                    repr(kernel),
                    instrumentation_mode,
                    capture.environment,
                    self._cache_dir,
                    self._trace_directory,
                    phase,
                    capture.test,
                    os.environ.get("PYTEST_XDIST_WORKER", "main"),
                )
                if self._connection is None:
                    self._submit(digest, task)
                else:
                    self._submitted_keys.add(digest)
                    self._connection.send((digest, task))

    def _submit(self, digest, task):
        if digest in self._submitted_keys:
            return
        self._submitted_keys.add(digest)
        if self._first_submit is None:
            self._first_submit = time.monotonic()
        self._pending.append(self._executor.submit(_preload_worker, *task))

    def _capture_preloads(self, instrumentation_mode, environment, kernel, args, grid, kwargs, test):
        from triton.runtime import jit

        previous_hook = triton.knobs.runtime.jit_cache_hook
        previous_serializer = jit.serialize_specialization_data
        captures = []

        def serialize_specialization(name, signature, constants, attrs, options, key, target):
            try:
                return previous_serializer(name, signature, constants, attrs, options, key, target)
            except TypeError as error:
                if "not JSON serializable" not in str(error):
                    raise
                return json.dumps({"name": name, "options": options.__dict__, "target": target.__dict__})

        def capture_specialization(*, key, repr, compile, fn, is_manual_warmup, already_compiled=False):
            if previous_hook is not None:
                previous_hook(key=key, repr=repr, compile=compile, fn=fn, is_manual_warmup=is_manual_warmup,
                              already_compiled=already_compiled)
            specialization = json.loads(compile["specialization_data"])
            options = {
                name: tuple(value) if isinstance(value, list) else value
                for name, value in specialization["options"].items()
            }
            captures.append((
                fn.jit_function,
                specialization["name"],
                key,
                specialization["target"],
                options,
                compile["signature"],
                compile["constants"] or {},
                compile.get("configs", [{}])[0],
            ))
            return True

        triton.knobs.runtime.jit_cache_hook = capture_specialization
        jit.serialize_specialization_data = serialize_specialization
        try:
            try:
                _warmup_kernel(kernel, args, grid, kwargs)
            except OverflowError as error:
                arg_details = []
                for arg in args:
                    if hasattr(arg, "data_ptr"):
                        arg_details.append(f"{type(arg).__name__}(shape={tuple(arg.shape)}, dtype={arg.dtype}, "
                                           f"data_ptr={arg.data_ptr()})")
                    else:
                        arg_details.append(repr(arg))
                raise OverflowError(
                    f"{error}; kernel={kernel}; args=[{', '.join(arg_details)}]; kwargs={kwargs}") from error
        finally:
            jit.serialize_specialization_data = previous_serializer
            triton.knobs.runtime.jit_cache_hook = previous_hook
        result = []
        for function, name, key, target, options, signature, constants, attrs in captures:
            # Triton's in-memory cache uses this exact key, which already
            # includes the specialization and compile options. Deduplicate
            # before cloudpickling the function and compile context.
            function_identity = id(function)
            self._serialized_functions[function_identity] = function
            serialized_key = (instrumentation_mode, tuple(sorted(environment.items())), function_identity, key)
            if serialized_key in self._serialized_keys:
                continue
            self._serialized_keys.add(serialized_key)
            import_path = _jit_callable_import_path(function)
            if import_path is not None:
                module_name, qualified_name = import_path
                fn_bytes = None
            else:
                module_name = None
                qualified_name = None
                try:
                    fn_bytes = _jit_dumps(function)
                except Exception as error:
                    self._serialized_keys.remove(serialized_key)
                    raise _UnsupportedPreloadError(
                        f"Unable to serialize JITFunction {function}: {type(error).__name__}: {error}") from error
            try:
                compile_context_bytes = _jit_dumps((name, key, target, options, signature, constants, attrs))
            except Exception as error:
                self._serialized_keys.remove(serialized_key)
                raise _UnsupportedPreloadError(
                    f"Unable to serialize compile context for {function}: {type(error).__name__}: {error}") from error
            result.append(
                _CapturedPreload(
                    module_name,
                    qualified_name,
                    fn_bytes,
                    compile_context_bytes,
                    environment,
                    test,
                ))
        return result

    def finish(self):
        if self._connection is not None:
            self._connection.close()
            self._connection = None
            return
        worker_results = []
        submitted = len(self._pending)
        try:
            for future in self._pending:
                worker_results.append(future.result())
        finally:
            self._executor.shutdown(wait=True)
            end = time.monotonic()
            elapsed = end - self._first_submit if self._first_submit is not None else 0.0
            worker_seconds = {}
            for worker, duration in worker_results:
                worker_seconds[worker] = worker_seconds.get(worker, 0.0) + duration
            task_seconds = sum(worker_seconds.values())
            workers_used = len(worker_seconds)
            maximum = max(worker_seconds.values(), default=0.0)
            metrics = {
                "phase": self._phase,
                "capture_worker": os.environ.get("PYTEST_XDIST_WORKER", "main"),
                "workers": self._max_workers,
                "workers_used": workers_used,
                "submitted": submitted,
                "active_seconds": round(elapsed, 3),
                "task_seconds": round(task_seconds, 3),
                "effective_parallelism": round(task_seconds / elapsed, 3) if elapsed else 0.0,
                "pool_utilization": round(task_seconds / (self._max_workers * elapsed), 4) if elapsed else 0.0,
                "load_balance": round(task_seconds / (workers_used * maximum), 4) if workers_used and maximum else 0.0,
                "worker_seconds_min": round(min(worker_seconds.values()), 3) if worker_seconds else 0.0,
                "worker_seconds_median":
                round(statistics.median(worker_seconds.values()), 3) if worker_seconds else 0.0,
                "worker_seconds_max": round(maximum, 3),
            }
            print(f"TRITON_CI_WARMUP_POOL {json.dumps(metrics, sort_keys=True)}", flush=True)
            self._pending.clear()
            self._attempted_tests.clear()
            self._serialized_keys.clear()
            self._serialized_functions.clear()
            self._submitted_keys.clear()


class SharedWarmupCoordinator:
    """Share one bounded compiler pool across every pytest capture worker."""

    def __init__(self, *, max_workers, trace_directory):
        self._directory = tempfile.TemporaryDirectory(prefix="triton-warmup-")
        self.address = os.path.join(self._directory.name, "coordinator.sock")
        self._listener = Listener(self.address, family="AF_UNIX")
        self._dispatcher = ProcessPoolWarmupDispatcher(max_workers=max_workers, trace_directory=trace_directory,
                                                       phase="warmup")
        self._connections = []
        self._accept_thread = threading.Thread(target=self._accept, daemon=True)
        self._accept_thread.start()

    def _accept(self):
        while True:
            connection = self._listener.accept()
            try:
                message = connection.recv()
            except EOFError:
                connection.close()
                continue
            if message is None:
                connection.close()
                return
            thread = threading.Thread(target=self._consume, args=(connection, message), daemon=True)
            self._connections.append(thread)
            thread.start()

    def _consume(self, connection, message):
        try:
            while True:
                digest, task = message
                with self._dispatcher._capture_lock:
                    self._dispatcher._submit(digest, task)
                message = connection.recv()
        except EOFError:
            connection.close()

    def close(self):
        stopper = Client(self.address, family="AF_UNIX")
        stopper.send(None)
        stopper.close()
        self._accept_thread.join()
        for thread in self._connections:
            thread.join()
        self._listener.close()
        try:
            self._dispatcher.finish()
        finally:
            self._directory.cleanup()
