# flake8: noqa
import triton
import triton.language.standard as tl_standard
from .._runtime import jit
from triton import knobs

_IMPORT_FROM_TRITON = [
    "sum",
    "max",
    "min",
    "reduce_or",
    "xor_sum",
]

__all__ = _IMPORT_FROM_TRITON

for name in _IMPORT_FROM_TRITON:
    # Convert JITFunction -> GluonJitFunction
    fn = getattr(tl_standard, name)
    assert knobs.runtime.interpret or isinstance(fn, triton.runtime.JITFunction)
    globals()[name] = jit(fn.fn)
