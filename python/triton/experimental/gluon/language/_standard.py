from triton.experimental import gluon
import triton.experimental.gluon.language as ttgl

__all__ = [
    "zeros",
]


@gluon.jit
def zeros(shape, dtype, layout):
    return ttgl.full(shape, 0, dtype, layout)
