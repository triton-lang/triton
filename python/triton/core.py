from typing import Callable, TypeVar

from .base import *
from . import base

C = TypeVar("C", bound=Callable)


@base.builtin
def atomic_cas(pointer, cmp, val, _builder: base.ir.builder = None):
    """
    Performs an atomic compare-and-swap at the memory location specified by :code:`pointer`.

    Return the data stored at :code:`pointer` before the atomic operation.

    :param pointer: The memory locations to compare-and-swap.
    :type pointer: Block of dtype=triton.PointerDType
    :param cmp: The values expected to be found in the atomic object
    :type cmp: Block of dtype=`pointer.dtype.element_ty`
    :param val: The values to copy in case the expected value matches the contained value.
    :type val: Block of dtype=`pointer.dtype.element_ty`
    """
    cmp = base._to_tensor(cmp, _builder)
    val = base._to_tensor(val, _builder)
    return base._atomic_cas(pointer, cmp, val, _builder)


def _add_atomic_docstr(name: str) -> Callable[[C], C]:
    def _decorator(func: C) -> C:
        docstr = """
    Performs an atomic {name} at the memory location specified by :code:`pointer`.

    Return the data stored at :code:`pointer` before the atomic operation.

    :param pointer: The memory locations to apply {name}.
    :type pointer: Block of dtype=triton.PointerDType
    :param val: The values to {name} in the atomic object.
    :type val: Block of dtype=`pointer.dtype.element_ty`
    :param mask: If mask[idx] is false, do not apply {name}.
    :type mask: Block of triton.int1, optional
    """
        func.__doc__ = docstr.format(name=name)
        return func

    return _decorator


@base.builtin
@_add_atomic_docstr("exchange")
def atomic_xchg(pointer, val, mask=None, _builder: base.ir.builder = None):
    val = base._to_tensor(val, _builder)
    return base._atomic_xchg(pointer, val, mask, _builder)


@base.builtin
@_add_atomic_docstr("add")
def atomic_add(pointer, val, mask=None, _builder: base.ir.builder = None):
    val = base._to_tensor(val, _builder)
    return base._atomic_add(pointer, val, mask, _builder)


@base.builtin
@_add_atomic_docstr("max")
def atomic_max(pointer, val, mask=None, _builder: base.ir.builder = None):
    val = base._to_tensor(val, _builder)
    return base._atomic_max(pointer, val, mask, _builder)


@base.builtin
@_add_atomic_docstr("min")
def atomic_min(pointer, val, mask=None, _builder: base.ir.builder = None):
    val = base._to_tensor(val, _builder)
    return base._atomic_min(pointer, val, mask, _builder)


@base.builtin
@_add_atomic_docstr("logical and")
def atomic_and(pointer, val, mask=None, _builder: base.ir.builder = None):
    val = base._to_tensor(val, _builder)
    return base._atomic_and(pointer, val, mask, _builder)


@base.builtin
@_add_atomic_docstr("logical or")
def atomic_or(pointer, val, mask=None, _builder: base.ir.builder = None):
    val = base._to_tensor(val, _builder)
    return base._atomic_or(pointer, val, mask, _builder)


@base.builtin
@_add_atomic_docstr("logical xor")
def atomic_xor(pointer, val, mask=None, _builder: base.ir.builder = None):
    val = base._to_tensor(val, _builder)
    return base._atomic_xor(pointer, val, mask, _builder)


# -----------------------
# Conditioning
# -----------------------


@base.builtin
def where(condition, x, y, _builder: base.ir.builder = None):
    """
    Returns a tensor of elements from either :code:`x` or :code:`y`, depending on :code:`condition`.

    Note that :code:`x` and :code:`y` are always evaluated regardless of the value of :code:`condition`.

    If you want to avoid unintended memory operations, use the :code:`mask` arguments in `triton.load` and `triton.store` instead.

    The shape of :code:`x` and :code:`y` are both broadcast to the shape of :code:`condition`.
    :code:`x` and :code:`y` must have the data type.

    :param condition: When True (nonzero), yield x, otherwise yield y.
    :type condition: Block of triton.bool
    :param x: values selected at indices where condition is True.
    :param y: values selected at indices where condition is False.
    """
    condition = base._to_tensor(condition, _builder)
    x = base._to_tensor(x, _builder)
    y = base._to_tensor(y, _builder)
    return base._where(condition, x, y, _builder)


# -----------------------
# Math
# -----------------------


@base.builtin
def umulhi(x, y, _builder: base.ir.builder = None):
    x = base._to_tensor(x, _builder)
    y = base._to_tensor(y, _builder)
    return base._umulhi(x, y, _builder)


@base.builtin
def fdiv(x, y, ieee_rounding=False, _builder: base.ir.builder = None):
    ieee_rounding = base._constexpr_to_value(ieee_rounding)
    return base._fdiv(x, y, ieee_rounding, _builder)


def _add_math_1arg_docstr(name: str) -> Callable[[C], C]:
    def _decorator(func: C) -> C:
        docstr = """
    Computes the element-wise {name} of :code:`x`

    :param x: the input values
    :type x: Block
    """
        func.__doc__ = docstr.format(name=name)
        return func

    return _decorator


@base.builtin
@_add_math_1arg_docstr("exponential")
def exp(x, _builder: base.ir.builder = None):
    return base._exp(x, _builder)


@base.builtin
@_add_math_1arg_docstr("natural logarithm")
def log(x, _builder: base.ir.builder = None):
    return base._log(x, _builder)


@base.builtin
@_add_math_1arg_docstr("cosine")
def cos(x, _builder: base.ir.builder = None):
    return base._cos(x, _builder)


@base.builtin
@_add_math_1arg_docstr("sine")
def sin(x, _builder: base.ir.builder = None):
    return base._sin(x, _builder)


@base.builtin
@_add_math_1arg_docstr("square root")
def sqrt(x, _builder: base.ir.builder = None):
    return base._sqrt(x, _builder)


# -----------------------
# Reductions
# -----------------------


def _add_reduction_docstr(name: str) -> Callable[[C], C]:
    def _decorator(func: C) -> C:
        docstr = """
    Returns the {name} of all elements in the :code:`input` tensor along the provided :code:`axis`

    :param input: the input values
    :param axis: the dimension along which the reduction should be done
    """
        func.__doc__ = docstr.format(name=name)
        return func

    return _decorator


@base.builtin
@_add_reduction_docstr("maximum")
def max(input, axis, _builder: base.ir.builder = None):
    axis = base._constexpr_to_value(axis)
    return base._max(input, axis, _builder)


@base.builtin
@_add_reduction_docstr("maximum index")
def argmax(input, axis, _builder: base.ir.builder = None):
    axis = base._constexpr_to_value(axis)
    return base._argmax(input, axis, _builder)


@base.builtin
@_add_reduction_docstr("minimum")
def min(input, axis, _builder: base.ir.builder = None):
    axis = base._constexpr_to_value(axis)
    return base._min(input, axis, _builder)


@base.builtin
@_add_reduction_docstr("minimum index")
def argmin(input, axis, _builder: base.ir.builder = None):
    axis = base._constexpr_to_value(axis)
    return base._argmin(input, axis, _builder)


@base.builtin
@_add_reduction_docstr("sum")
def sum(input, axis, _builder: base.ir.builder = None):
    axis = base._constexpr_to_value(axis)
    return base._sum(input, axis, _builder)


@base.builtin
@_add_reduction_docstr("xor sum")
def xor_sum(input, axis, _builder: base.ir.builder = None):
    axis = base._constexpr_to_value(axis)
    return base._xor_sum(input, axis, _builder)


# -----------------------
# Utilities
# -----------------------


@base.builtin
def globaltimer(_builder: base.ir.builder = None):
    return base._globaltimer(_builder)


@base.builtin
def clock(_builder: base.ir.builder = None):
    return base._clock(_builder)


# -----------------------
# Internal for debugging
# -----------------------


@base.builtin
def debug_barrier(_builder: base.ir.builder = None):
    return base._debug_barrier(_builder)


@base.builtin
def multiple_of(input, values, _builder: base.ir.builder = None):
    """
    Let the compiler knows that the values in :code:`input` are all multiples of :code:`value`.
    """
    if isinstance(values, constexpr):
        values = [values]
    for i, d in enumerate(values):
        if not isinstance(d, constexpr):
            raise TypeError(f"values element {i} must have type `constexpr`")
        if not isinstance(d.value, int):
            raise TypeError(
                f"values element {i} must have type `constexpr[int]`, got `constexpr[{type(d.value)}]"
            )
    values = [x.value for x in values]
    return base._multiple_of(input, values)


@base.builtin
def max_contiguous(input, values, _builder: base.ir.builder = None):
    """
    Let the compiler knows that the `value` first values in :code:`input` are contiguous.
    """
    if isinstance(values, constexpr):
        values = [values]
    for i, d in enumerate(values):
        if not isinstance(d, constexpr):
            raise TypeError(f"values element {i} must have type `constexpr`")
        if not isinstance(d.value, int):
            raise TypeError(
                f"values element {i} must have type `constexpr[int]`, got `constexpr[{type(d.value)}]"
            )
    values = [x.value for x in values]
    return base._max_contiguous(input, values)


# -----------------------
# Standard library
# -----------------------


@jit
def abs(x):
    return where(x >= 0, x, -x)


@jit
def cdiv(x, div):
    """
    Computes the ceiling division of :code:`x` by :code:`div`

    :param x: the input number
    :type input: Block
    :param div: the divisor
    :param div: Block
    """
    return (x + div - 1) // div


@jit
def minimum(x, y):
    """
    Computes the element-wise minimum of :code:`x` and :code:`y`.

    :param input: the first input tensor
    :type input: Block
    :param other: the second input tensor
    :type other: Block
    """
    return where(x < y, x, y)


@jit
def maximum(x, y):
    """
    Computes the element-wise maximum of :code:`x` and :code:`y`.

    :param input: the first input tensor
    :type input: Block
    :param other: the second input tensor
    :type other: Block
    """
    return where(x > y, x, y)


@jit
@_add_math_1arg_docstr("sigmoid")
def sigmoid(x):
    return 1 / (1 + exp(-x))


@jit
@_add_math_1arg_docstr("softmax")
def softmax(x, ieee_rounding: constexpr = constexpr(False)):
    z = x - max(x, 0)
    num = exp(z)
    den = sum(num, 0)
    return fdiv(num, den, ieee_rounding)


@jit
def ravel(x):
    """
    Returns a contiguous flattened view of :code:`x`

    :param x: the input tensor
    :type x: Block
    """
    return reshape(x, [x.numel])


@jit
def swizzle2d(i, j, size_i, size_j, size_g):
    """
    transformes indices of a row-major size_i*size_j matrix into those
    of one where indices are row major for each group of size_j rows.
    For example, for size_i = size_j = 4 and size_g = 2, it will transform
    [[0 , 1 , 2 , 3 ],
     [4 , 5 , 6 , 7 ],
     [8 , 9 , 10, 11],
     [12, 13, 14, 15]]
    into
    [[0, 2,  4 , 6 ],
     [1, 3,  5 , 7 ],
     [8, 10, 12, 14],
     [9, 11, 13, 15]]
    """
    # "unrolled index in array"
    ij = i * size_j + j
    # number of elements in `size_g` groups
    # of `size_j` columns
    size_gj = size_g * size_j
    # index of the group in which (i,j) is
    group_id = ij // size_gj
    # row-index of the first element of this group
    off_i = group_id * size_g
    # last group may have fewer rows
    size_g = minimum(size_i - off_i, size_g)
    # new row and column indices
    new_i = off_i + (ij % size_g)
    new_j = (ij % size_gj) // size_g
    return new_i, new_j


@jit
def zeros_like(input):
    return zeros(input.shape, input.dtype)
