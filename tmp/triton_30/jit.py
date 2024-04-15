import os
from typing import Callable, Iterable, Optional, Union

from triton.runtime.jit import JITFunction, T


def _getitem(self, grid):
    return lambda *args, **kwargs: self.run(*args, grid=grid, **kwargs)


JITFunction.__getitem__ = _getitem


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
            if os.getenv("TRITON_OLD_JIT_FUNCTION", "0") == "1":
                from .jit_old import JITFunction as jit_cls
            else:
                jit_cls = JITFunction
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
