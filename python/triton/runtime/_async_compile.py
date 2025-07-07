from __future__ import annotations
from typing import Callable, Optional
from concurrent.futures import Executor, as_completed, Future

active_mode: Optional[AsyncCompileMode] = None


class FutureKernel:

    def __init__(self, finalize_compile: Callable, future: Future):
        self.finalize_compile = finalize_compile
        self.kernel = None
        self.future = future

    def result(self):
        if self.kernel is not None:
            return self.kernel

        kernel = self.future.result()
        self.finalize_compile(kernel)
        self.kernel = kernel
        return kernel


class AsyncCompileMode:

    def __init__(self, executor: Executor):
        self.executor = executor
        self.raw_futures = []
        self.future_kernels = {}

    def submit(self, key, compile_fn, finalize_fn):
        future = self.future_kernels.get(key)
        if future is not None:
            return future

        future = self.executor.submit(compile_fn)
        future._key = key
        self.raw_futures.append(future)
        future_kernel = FutureKernel(finalize_fn, future)
        self.future_kernels[key] = future_kernel
        return future_kernel

    def __enter__(self):
        global active_mode
        if active_mode is not None:
            raise RuntimeError("Another AsyncCompileMode is already active")
        active_mode = self

    def __exit__(self, exc_type, exc_value, traceback):
        global active_mode
        # Finalize any outstanding compiles
        for future in as_completed(self.raw_futures):
            self.future_kernels[future._key].result()
        active_mode = None
