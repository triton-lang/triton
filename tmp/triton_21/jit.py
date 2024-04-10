import os
from typing import Callable, Iterable, Optional, Union

from triton.runtime.jit import JITFunction, T


def jit(
    fn: Optional[T] = None,
    *,
    version=None,
    do_not_specialize: Optional[Iterable[int]] = None,
    debug: Optional[bool] = None,
    noinline: Optional[bool] = None,
) -> Union[JITFunction[T], Callable[[T], JITFunction[T]]]:
    def decorator(fn: T) -> JITFunction[T]:
        assert callable(fn)
        if os.getenv("TRITON_INTERPRET", "0") == "1":
            from triton.runtime.interpreter import InterpretedFunction

            return InterpretedFunction(fn)
        else:
            jit_cls = JITFunction
            if os.getenv("TRITON_EXPERIMENTAL_JIT_FUNCTION_CPP", "0") == "1":
                from .low_latency_jit_cpp import LowLatencyJITFunctionCPP as jit_cls
            elif os.getenv("TRITON_EXPERIMENTAL_JIT_FUNCTION_PYTHON", "0") == "1":
                from .low_latency_jit_python import (
                    LowLatencyJITFunctionPython as jit_cls,
                )
            return jit_cls(
                fn,
                version=version,
                do_not_specialize=do_not_specialize,
                debug=debug,
                noinline=noinline,
            )

    if fn is not None:
        return decorator(fn)

    else:
        return decorator
