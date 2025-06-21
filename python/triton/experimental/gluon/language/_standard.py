# flake8: noqa
import triton
import triton.language.standard as tl_standard
from .._runtime import jit
from triton import knobs
from . import _core as ttgl

_IMPORT_FROM_TRITON = [
    "sum",
    "max",
    "min",
    "reduce_or",
    "xor_sum",
]

__all__ = [
    "full_like",
    "zeros",
    "zeros_like",
    *_IMPORT_FROM_TRITON,
]

for name in _IMPORT_FROM_TRITON:
    # Convert JITFunction -> GluonJitFunction
    fn = getattr(tl_standard, name)
    assert knobs.runtime.interpret or isinstance(fn, triton.runtime.JITFunction)
    globals()[name] = jit(fn.fn)


@jit
def zeros(shape, dtype, layout):
    return ttgl.full(shape, 0, dtype, layout)


@jit
def full_like(input, value, shape=None, dtype=None, layout=None):
    return ttgl.full(
        input.shape if shape is None else shape,
        value,
        input.dtype if dtype is None else dtype,
        input.type.layout if layout is None else layout,
    )


@jit
def zeros_like(input, shape=None, dtype=None, layout=None):
    return full_like(input, 0, shape=shape, dtype=dtype, layout=layout)
