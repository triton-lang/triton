from __future__ import annotations
from typing import Callable, Optional
from concurrent.futures import Executor, as_completed, Future
from contextvars import ContextVar

active_mode: ContextVar[Optional[AsyncCompileMode]] = ContextVar("async_compile_active_mode", default=None)


class FutureKernel:

    def __init__(self, future: Future):
        # Several warmups can share one pending compile by cache key. Keep
        # per-waiter callbacks so every JIT cache entry is finalized or cleaned.
        self.finalize_compile = []
        self.cleanup_compile = []
        self.kernel = None
        self.future = future

    def add_callbacks(self, finalize_compile: Callable, cleanup_compile: Callable):
        self.finalize_compile.append(finalize_compile)
        self.cleanup_compile.append(cleanup_compile)

    def result(self, ignore_errors: bool = False):
        if self.kernel is not None:
            return self.kernel

        try:
            kernel = self.future.result()
        except Exception:
            for cleanup_compile in self.cleanup_compile:
                cleanup_compile(self)
            self.future = None
            self.finalize_compile = []
            self.cleanup_compile = []
            if ignore_errors:
                return
            else:
                raise
        for finalize_compile in self.finalize_compile:
            finalize_compile(kernel)
        self.future = None
        self.finalize_compile = []
        self.cleanup_compile = []
        self.kernel = kernel
        return kernel

    def __getattr__(self, name):
        # Defer to the compiled kernel so users can interact with this object
        # like a normal CompiledKernel without needing to call result() first.
        return getattr(self.result(), name)


class AsyncCompileMode:

    def __init__(self, executor: Executor, *, ignore_errors=False):
        self.executor = executor
        self.ignore_errors = ignore_errors
        self.raw_futures = []
        self.future_kernels = {}

    def submit(self, key, compile_fn, finalize_fn, cleanup_fn):
        future = self.future_kernels.get(key)
        if future is not None:
            future.add_callbacks(finalize_fn, cleanup_fn)
            return future

        future = self.executor.submit(compile_fn)
        future._key = key
        self.raw_futures.append(future)
        future_kernel = FutureKernel(future)
        future_kernel.add_callbacks(finalize_fn, cleanup_fn)
        self.future_kernels[key] = future_kernel
        return future_kernel

    def __enter__(self):
        if active_mode.get() is not None:
            raise RuntimeError("Another AsyncCompileMode is already active")
        active_mode.set(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        active_mode.set(None)
        # Finalize any outstanding compiles
        try:
            for future in as_completed(self.raw_futures):
                self.future_kernels[future._key].result(self.ignore_errors)
        finally:
            # Completed futures can still point back to compile frames so need
            # to drop them to avoid resource leakage.
            self.raw_futures = []
            self.future_kernels = {}
