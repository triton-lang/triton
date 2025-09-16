from triton.language.core import _aggregate as aggregate
from triton.experimental.gluon.language import _core as ttgl, _standard as stdlib
from triton.experimental.gluon._runtime import constexpr_function, jit

__all__ = [
    "pack2",
    "unpack2",
    "pack",
    "unpack",
    "fma",
    "Float2Tensor",
]


@jit
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


@jit
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


@jit
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


@jit
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

    @jit
    def __add__(self, rhs):
        ttgl.static_assert(isinstance(rhs, Float2Tensor), "rhs must be a Float2Tensor")
        return Float2Tensor(_add_f32x2(self._value, rhs._value))

    @jit
    def __sub__(self, rhs):
        ttgl.static_assert(isinstance(rhs, Float2Tensor), "rhs must be a Float2Tensor")
        return Float2Tensor(_sub_f32x2(self._value, rhs._value))

    @jit
    def __mul__(self, rhs):
        ttgl.static_assert(isinstance(rhs, Float2Tensor), "rhs must be a Float2Tensor")
        return Float2Tensor(_mul_f32x2(self._value, rhs._value))


@jit
def pack2(x0, x1):
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


@jit
def unpack2(x):
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


@constexpr_function
def _get_split_shape(shape, axis):
    shape = [d for d in shape]
    assert shape[axis] >= 2, f"not enough elements to pack along axis {axis}"
    shape[axis] //= 2
    shape.insert(axis + 1, 2)
    permute = list(range(len(shape)))
    permute[axis + 1], permute[len(permute) - 1] = permute[len(permute) - 1], permute[axis + 1]
    return ttgl.tuple(shape), ttgl.tuple(permute)


@constexpr_function
def _get_join_shape(shape, axis):
    shape = [d for d in shape]
    shape[axis] *= 2
    permute = list(range(len(shape)))
    permute.insert(axis + 1, len(permute))
    return ttgl.tuple(shape), ttgl.tuple(permute)


@jit
def pack(x, axis):
    sp: ttgl.constexpr = _get_split_shape(x.shape, axis)
    x0, x1 = x.reshape(*sp[0]).permute(*sp[1]).split()
    return pack2(x0, x1)


@jit
def unpack(x, axis):
    shape: ttgl.constexpr = x._value.shape
    sp: ttgl.constexpr = _get_join_shape(shape, axis)
    x0, x1 = unpack2(x)
    return ttgl.join(x0, x1).permute(*sp[1]).reshape(*sp[0])


@jit
def full_like(x, fill_value):
    ttgl.static_assert(fill_value.dtype == ttgl.float32, "fill_value must be a float32")
    fill = stdlib.full_like(x._value, fill_value, dtype=ttgl.float32)
    return pack2(fill, fill)


@jit
def fma(a, b, c):
    return Float2Tensor(_fma_f32x2(a._value, b._value, c._value))
