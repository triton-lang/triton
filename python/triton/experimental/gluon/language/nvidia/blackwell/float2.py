import triton.experimental.gluon as gluon
from triton.language.core import _aggregate as aggregate
from triton.experimental.gluon.language import _core as ttgl


@gluon.jit
def _add_f32x2(a, b):
    return ttgl.inline_asm_elementwise(
        """
        add.f32x2 $0, $1, $2;
        """,
        "=l,l,l",
        [a, b],
        dtype=ttgl.int64,
        is_pure=True,
        pack=1,
    )


@gluon.jit
def _sub_f32x2(a, b):
    return ttgl.inline_asm_elementwise(
        """
        sub.f32x2 $0, $1, $2;
        """,
        "=l,l,l",
        [a, b],
        dtype=ttgl.int64,
        is_pure=True,
        pack=1,
    )


@gluon.jit
def _mul_f32x2(a, b):
    return ttgl.inline_asm_elementwise(
        """
        mul.f32x2 $0, $1, $2;
        """,
        "=l,l,l",
        [a, b],
        dtype=ttgl.int64,
        is_pure=True,
        pack=1,
    )


@gluon.jit
def _fma_f32x2(a, b, c):
    return ttgl.inline_asm_elementwise(
        """
        fma.rn.f32x2 $0, $1, $2, $3;
        """,
        "=l,l,l,l",
        [a, b, c],
        dtype=ttgl.int64,
        is_pure=True,
        pack=1,
    )


@aggregate
class Float2Tensor:
    _value: ttgl.tensor

    def __init__(self, value: ttgl.tensor):
        self._value = value

    @gluon.jit
    def __add__(self, rhs):
        ttgl.static_assert(isinstance(rhs, Float2Tensor), "rhs must be a Float2Tensor")
        return Float2Tensor(_add_f32x2(self._value, rhs._value))

    @gluon.jit
    def __sub__(self, rhs):
        ttgl.static_assert(isinstance(rhs, Float2Tensor), "rhs must be a Float2Tensor")
        return Float2Tensor(_sub_f32x2(self._value, rhs._value))

    @gluon.jit
    def __mul__(self, rhs):
        ttgl.static_assert(isinstance(rhs, Float2Tensor), "rhs must be a Float2Tensor")
        return Float2Tensor(_mul_f32x2(self._value, rhs._value))


@gluon.jit
def pack(x0, x1):
    value = ttgl.inline_asm_elementwise(
        """
        mov.b64 $0, { $1, $2 };
        """,
        "=l,r,r",
        [x0, x1],
        dtype=ttgl.int64,
        is_pure=True,
        pack=1,
    )
    return Float2Tensor(value)


@gluon.jit
def unpack(x):
    return ttgl.inline_asm_elementwise(
        """
        mov.b64 { $0, $1 }, $2;
        """,
        "=r,=r,l",
        [x._value],
        dtype=[ttgl.float32, ttgl.float32],
        is_pure=True,
        pack=1,
    )


@gluon.jit
def fma(a, b, c):
    return Float2Tensor(_fma_f32x2(a._value, b._value, c._value))
