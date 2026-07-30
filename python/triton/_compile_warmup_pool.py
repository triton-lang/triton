import ast
import hashlib
import importlib
import io
import json
import linecache
import multiprocessing as mp
import os
import sys
import threading
import types
import warnings
from collections import deque
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
def _instrumentation_mode(instrumentation_mode):
    with triton.knobs.compilation.scope():
        triton.knobs.compilation.instrumentation_mode = instrumentation_mode
        yield


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
    module = ast.parse(function.src)
    target = None
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == function.__name__:
            target = node
            break
    if target is None:
        raise ValueError(f"Cannot find function {function.__name__} in source code of {function}")
    args = target.args
    all_args = list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs)
    if args.vararg is not None:
        all_args.append(args.vararg)
    if args.kwarg is not None:
        all_args.append(args.kwarg)
    for arg in all_args:
        if arg.annotation is not None:
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


def _make_dynamic_jit_fn(jit_fn_type, args, src, starting_line_number):
    python_function = args[0]
    if isinstance(python_function, _CodeGenFunction):
        python_function = python_function.jit_function.fn
    _restore_dynamic_source(python_function, src)
    function = jit_fn_type(*args)
    function._unsafe_update_src(src)
    function.starting_line_number = starting_line_number
    return function


def _make_dynamic_constexpr_fn(constexpr_fn_type, python_function, src, starting_line_number):
    _restore_dynamic_source(python_function, src)
    function = constexpr_fn_type(python_function)
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
        if isinstance(obj, ConstexprFunction):
            return _make_dynamic_constexpr_fn, (
                type(obj),
                obj.fn,
                obj.src,
                obj.starting_line_number,
            )
        return _make_dynamic_jit_fn, (
            type(obj),
            (
                _CodeGenFunction(obj),
                obj.version,
                obj.do_not_specialize,
                obj.do_not_specialize_on_alignment,
                obj.debug,
                obj.noinline,
                obj._repr,
                obj.launch_metadata,
            ),
            obj.src,
            obj.starting_line_number,
        )


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


@contextmanager
def _compile_trace(directory, phase, test):
    if directory is None:
        yield
        return
    from triton._compile_warmup import CompilationTrace

    previous_listener = triton.knobs.compilation.listener
    triton.knobs.compilation.listener = CompilationTrace(directory, phase, test)
    try:
        yield
    finally:
        triton.knobs.compilation.listener = previous_listener


def _preload_worker(module_name, qualified_name, fn_bytes, compile_context_bytes, kernel_repr, instrumentation_mode,
                    environment, trace_directory, phase, test):
    """Load one JIT function and populate the shared disk cache."""
    try:
        with (
                _instrumentation_mode(instrumentation_mode),
                _cache_invalidating_environment(environment),
                _compile_trace(trace_directory, phase, test),
        ):
            if module_name is not None and qualified_name is not None:
                function = _load_jit_callable(module_name, qualified_name)
            elif fn_bytes is not None:
                function = cloudpickle.loads(fn_bytes)
            else:
                raise AssertionError("missing JIT function import and serialized payload")
            compile_context = cloudpickle.loads(compile_context_bytes)
            _preload_with_compile_context(function, *compile_context)
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


@dataclass(frozen=True)
class _CapturedPreload:
    function: object
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
        context = mp.get_context("spawn")
        self._executor = ProcessPoolExecutor(max_workers=max_workers, mp_context=context)
        self._trace_directory = trace_directory
        self._phase = phase
        self._capture_lock = threading.Lock()
        self._pending = []
        self._attempted_tests = set()
        self._serialized_keys = set()
        self._submitted_keys = set()

    def record_test(self, test):
        if self._trace_directory is None or test is None or test in self._attempted_tests:
            return
        self._attempted_tests.add(test)
        os.makedirs(self._trace_directory, exist_ok=True)
        path = os.path.join(self._trace_directory, f"{self._phase}-{os.getpid()}.tests")
        with open(path, "a", encoding="utf-8") as output:
            output.write(json.dumps({"phase": self._phase, "test": test}, sort_keys=True) + "\n")

    def dispatch(self, *args, kernel, grid, test, **kwargs):
        from triton._C.libtriton import get_cache_invalidating_env_vars

        with self._capture_lock:
            instrumentation_mode = triton.knobs.compilation.instrumentation_mode
            environment = get_cache_invalidating_env_vars()
            try:
                captures = self._capture_preloads(
                    instrumentation_mode,
                    environment,
                    kernel,
                    args,
                    grid,
                    kwargs,
                    test,
                )
            except _UnsupportedPreloadError as error:
                warnings.warn(
                    f"Falling back to in-process warmup for {kernel}: {error!r}",
                    stacklevel=2,
                )
                with _compile_trace(self._trace_directory, self._phase, test):
                    return _warmup_kernel(kernel, args, grid, kwargs)
            self.record_test(test)
            for capture in captures:
                digest = hashlib.sha256(instrumentation_mode.encode() + b"\0" + capture.digest_payload()).hexdigest()
                if digest in self._submitted_keys:
                    continue
                self._submitted_keys.add(digest)
                future = self._executor.submit(
                    _preload_worker,
                    capture.module_name,
                    capture.qualified_name,
                    capture.fn_bytes,
                    capture.compile_context_bytes,
                    repr(kernel),
                    instrumentation_mode,
                    capture.environment,
                    self._trace_directory,
                    self._phase,
                    capture.test,
                )
                self._pending.append((future, capture, instrumentation_mode))

    def _capture_preloads(self, instrumentation_mode, environment, kernel, args, grid, kwargs, test):
        from triton.runtime import jit as triton_jit

        previous_hook = triton.knobs.runtime.jit_cache_hook
        previous_serializer = triton_jit.serialize_specialization_data
        captures = []
        specializations = deque()

        def serialize_specialization_data(name, signature, constants, attrs, options, key, target):
            # Triton's JSON payload cannot represent some constexprs, including
            # Gluon layouts. Keep the exact Python objects for the spawned
            # compiler and return a harmless payload to let _call_hook proceed.
            specializations.append((name, key, dict(target.__dict__), dict(options.__dict__)))
            try:
                return previous_serializer(name, signature, constants, attrs, options, key, target)
            except TypeError as error:
                if "not JSON serializable" not in str(error):
                    raise
                return json.dumps({"name": name, "process_pool_warmup": True})

        def cache_hook(key, repr, compile, fn, is_manual_warmup, already_compiled=False):
            if previous_hook is not None:
                previous_hook(
                    key=key,
                    repr=repr,
                    compile=compile,
                    fn=fn,
                    is_manual_warmup=is_manual_warmup,
                    already_compiled=already_compiled,
                )
            if not specializations:
                raise _UnsupportedPreloadError(f"Triton did not provide specialization metadata for {fn.jit_function}")
            name, compile_key, target, options = specializations.popleft()
            configs = compile.get("configs", [{}])
            captures.append((
                fn.jit_function,
                name,
                compile_key,
                target,
                options,
                compile.get("signature", {}),
                compile.get("constants") or {},
                configs[0],
            ))
            return True

        triton.knobs.runtime.jit_cache_hook = cache_hook
        triton_jit.serialize_specialization_data = serialize_specialization_data
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
            triton_jit.serialize_specialization_data = previous_serializer
            triton.knobs.runtime.jit_cache_hook = previous_hook

        if specializations:
            raise _UnsupportedPreloadError(
                f"Triton produced {len(specializations)} specialization payloads without invoking the cache hook")
        result = []
        for function, name, key, target, options, signature, constants, attrs in captures:
            # Triton's in-memory cache uses this exact key, which already
            # includes the specialization and compile options. Deduplicate
            # before cloudpickling the function and compile context.
            serialized_key = (instrumentation_mode, tuple(sorted(environment.items())), id(function), key)
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
                    function,
                    module_name,
                    qualified_name,
                    fn_bytes,
                    compile_context_bytes,
                    environment,
                    test,
                ))
        return result

    def finish(self):
        compilation_failures = []
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Accessing the data pointer of FakeTensor.*")
                for future, capture, instrumentation_mode in self._pending:
                    try:
                        future.result()
                    except _CompilationFailedError as error:
                        compilation_failures.append(error)
                    except _UnsupportedPreloadError as error:
                        warnings.warn(
                            "Falling back to in-process preload after child failure: "
                            f"{error!r}; cause={error.__cause__!r}",
                            stacklevel=2,
                        )
                        try:
                            with (
                                    _instrumentation_mode(instrumentation_mode),
                                    _cache_invalidating_environment(capture.environment),
                                    _compile_trace(self._trace_directory, self._phase, capture.test),
                            ):
                                compile_context = cloudpickle.loads(capture.compile_context_bytes)
                                _preload_with_compile_context(capture.function, *compile_context)
                        except Exception as fallback_error:
                            warnings.warn(
                                f"Skipping failed in-process warmup: {type(fallback_error).__name__}: "
                                f"{fallback_error}",
                                stacklevel=2,
                            )
                if compilation_failures:
                    if os.environ.get("TRITON_WARMUP_DEBUG"):
                        preview = "\n\n".join(str(error) for error in compilation_failures[:3])
                    else:
                        preview = "; ".join(str(error).splitlines()[0] for error in compilation_failures[:3])
                    remaining = len(compilation_failures) - 3
                    if remaining > 0:
                        preview += f"; and {remaining} more"
                    warnings.warn(
                        f"{len(compilation_failures)} captured launches failed warmup compilation: {preview}",
                        stacklevel=2,
                    )
        finally:
            self._pending.clear()
            self._attempted_tests.clear()
            self._serialized_keys.clear()
            self._submitted_keys.clear()
            self._executor.shutdown(wait=True)
