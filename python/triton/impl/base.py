from __future__ import annotations

from functools import wraps
from typing import TypeVar

T = TypeVar("T")

TRITON_BUILTIN = "__triton_builtin__"


def builtin(fn: T) -> T:
    """Mark a function as a builtin."""
    assert callable(fn)

    @wraps(fn)
    def wrapper(*args, **kwargs):
        if "_builder" not in kwargs or kwargs["_builder"] is None:
            raise ValueError(
                "Did you forget to add @triton.jit ? "
                "(`_builder` argument must be provided outside of JIT functions.)"
            )
        return fn(*args, **kwargs)

    setattr(wrapper, TRITON_BUILTIN, True)

    return wrapper


def is_builtin(fn) -> bool:
    """Is this a registered triton builtin function?"""
    return getattr(fn, TRITON_BUILTIN, False)


def extern(fn: T) -> T:
    """A decorator for external functions."""
    return builtin(fn)
