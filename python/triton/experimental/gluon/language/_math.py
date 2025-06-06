# flake8: noqa
import triton.language.math as tl_math
from ._core import builtin

__all__ = [
    "umulhi", "exp", "exp2", "fma", "log", "log2", "cos", "rsqrt", "sin", "sqrt", "sqrt_rn", "abs", "fdiv", "div_rn",
    "erf", "floor", "ceil"
]

for name in __all__:
    fn = getattr(tl_math, name)
    globals()[name] = builtin(fn)
