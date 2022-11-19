from __future__ import division, annotations

from typing import TypeVar, Callable

from triton import ir
from . import base
from . import jitlib

from .base import constexpr

CallableT = TypeVar("CallableT", bound=Callable)


@base.builtin
def program_id(axis, _builder: ir.builder = None):
    """
    Returns the id of the current program instance along the given :code:`axis`.

    :param axis: The axis of the 3D launch grid. Has to be either 0, 1 or 2.
    :type axis: int
    """
    # if axis == -1:
    #     pid0 = program_id(0, _builder)
    #     pid1 = program_id(1, _builder)
    #     pid2 = program_id(2, _builder)
    #     npg0 = num_programs(0, _builder)
    #     npg1 = num_programs(0, _builder)
    #     return pid0 + pid1*npg0 + pid2*npg0*npg1
    axis = base._constexpr_to_value(axis)
    return base._program_id(axis, _builder)


@base.builtin
def num_programs(axis, _builder: ir.builder = None):
    """
    Returns the number of program instances launched along the given :code:`axis`.

    :param axis: The axis of the 3D launch grid. Has to be either 0, 1 or 2.
    :type axis: int
    """
    axis = base._constexpr_to_value(axis)
    return base._num_programs(axis, _builder)


@base.builtin
def arange(start, end, _builder: ir.builder = None):
    """
    Returns contiguous values within the open interval [:code:`start`, :code:`end`).

    :param start: Start of the interval. Must be a power of two.
    :type start: int
    :param stop: End of the interval. Must be a power of two >= start.
    :type stop: int
    """
    start = base._constexpr_to_value(start)
    end = base._constexpr_to_value(end)
    return base._arange(start, end, _builder)


@base.builtin
def zeros(shape, dtype, _builder: ir.builder = None):
    """
    Returns a tensor filled with the scalar value 0 for the given :code:`shape` and :code:`dtype`.

    :param shape: Shape of the new array, e.g., (8, 16) or (8, )
    :type shape: tuple of ints
    :param dtype: Data-type of the new array, e.g., :code:`float16`
    :type dtype: DType
    """
    for i, d in enumerate(shape):
        if not isinstance(d, constexpr):
            raise TypeError(f"Shape element {i} must have type `constexpr`")
        if not isinstance(d.value, int):
            raise TypeError(
                f"Shape element {i} must have type `constexpr[int]`, got `constexpr[{type(d.value)}]"
            )
    shape = [x.value for x in shape]
    dtype = base._constexpr_to_value(dtype)
    return base._zeros(shape, dtype, _builder)


@base.builtin
def dequantize(
    input,
    scale,
    shift,
    nbit,
    dst_ty: base.dtype = base.float16,
    _builder: ir.builder = None,
):
    """
    Tries to dequantize the input to given dtype
    """
    nbit = base._constexpr_to_value(nbit)
    return base._dequantize(input, scale, shift, nbit, dst_ty, _builder)


@base.builtin
def broadcast(input, other, _builder: ir.builder = None):
    """
    Tries to broadcast the two given blocks to a common compatible shape.

    :param input: The first input tensor.
    :type input: Block
    :param other: The second input tensor.
    :type other: Block
    """
    return base._broadcast_impl_value(input, other, _builder)


@base.builtin
def broadcast_to(input, shape, _builder: ir.builder = None):
    """
    Tries to broadcast the given tensor to a new :code:`shape`.

    :param input: The input tensor.
    :type input: Block
    :param shape: The desired shape.
    :type shape: Tuple[int]
    """
    return base._broadcast_impl_shape(input, shape, _builder)


@base.builtin
def cat(input, other, _builder: ir.builder = None):
    """
    Concatenate the given blocks

    :param input: The first input tensor.
    :type input:
    :param other: The second input tensor.
    :type other:
    """
    return base._cat(input, other, _builder)


@base.builtin
def reshape(input, shape, _builder: ir.builder = None):
    """
    Tries to reshape the given tensor to a new shape.

    :param input: The input tensor.
    :type input:
    :param shape: The desired shape.
    :type shape: Tuple[int]

    """
    shape = [x.value for x in shape]
    return base._reshape(input, shape, _builder)


@base.builtin
def dot(
    input,
    other,
    trans_a=False,
    trans_b=False,
    allow_tf32=True,
    _builder: ir.builder = None,
):
    """
    Returns the matrix product of two blocks.

    The two blocks must be two dimensionals and have compatible inner dimensions.

    :param input: The first tensor to be multiplied.
    :type input: 2D tensor of scalar-type in {:code:`float16`, :code:`bfloat16`, :code:`float32`}
    :param other: The second tensor to be multiplied.
    :type other: 2D tensor of scalar-type in {:code:`float16`, :code:`bfloat16`, :code:`float32`}
    """
    allow_tf32 = base._constexpr_to_value(allow_tf32)
    return base._dot(input, other, trans_a, trans_b, allow_tf32, _builder)


@base.builtin
def load(
    pointer,
    mask=None,
    other=None,
    cache_modifier="",
    eviction_policy="",
    volatile=False,
    _builder: ir.builder = None,
) -> base.tensor:
    """
    Return a tensor of data whose values are, elementwise, loaded from memory at location defined by :code:`pointer`.

    :param *:
    :code:`mask` and :code:`other` are implicitly broadcast to :code:`pointer.shape`.

    :code:`other` is implicitly typecast to :code:`pointer.dtype.element_ty`.

    :param pointer: Pointers to the data to be loaded.
    :type pointer: Block of dtype=tr.PointerDType
    :param mask: if mask[idx] is false, do not load the data at address :code:`pointer[idx]`.
    :type mask: Block of tr.int1, optional
    :param other: if mask[idx] is false, return other[idx]
    :type other: Block, optional
    :param cache_modifier: changes cache option in nvidia ptx
    'type cache_modifier: str, optional
    """
    # mask, other can be constexpr
    if mask is not None:
        mask = base._to_tensor(mask, _builder)
    if other is not None:
        other = base._to_tensor(other, _builder)
    cache_modifier = base._constexpr_to_value(cache_modifier)
    eviction_policy = base._constexpr_to_value(eviction_policy)
    volatile = base._constexpr_to_value(volatile)
    return base._load(
        pointer,
        mask,
        other,
        cache_modifier,
        eviction_policy,
        volatile,
        _builder,
    )


@base.builtin
def store(
    pointer,
    value,
    mask=None,
    eviction_policy="",
    _builder: ir.builder = None,
) -> base.tensor:
    """
    Stores :code:`value` tensor of elements in memory, element-wise, at the memory locations specified by :code:`pointer`.

    :param *:
    :code:`value` is implicitly broadcast to :code:`pointer.shape` and typecast to :code:`pointer.dtype.element_ty`.

    :param pointer: The memory locations where the elements of :code:`value` are stored.
    :type pointer: Block of dtype=tr.PointerDType
    :param value: The tensor of elements to be stored.
    :type value: Block
    :param mask: If mask[idx] is false, do not store :code:`value[idx]` at :code:`pointer[idx]`.
    :type mask: Block of tr.int1, optional
    """
    # value can be constexpr
    value = base._to_tensor(value, _builder)
    if mask is not None:
        mask = base._to_tensor(mask, _builder)
    return base._store(pointer, value, mask, eviction_policy, _builder)


@base.builtin
def atomic_cas(pointer, cmp, val, _builder: ir.builder = None):
    """
    Performs an atomic compare-and-swap at the memory location specified by :code:`pointer`.

    Return the data stored at :code:`pointer` before the atomic operation.

    :param pointer: The memory locations to compare-and-swap.
    :type pointer: Block of dtype=tr.PointerDType
    :param cmp: The values expected to be found in the atomic object
    :type cmp: Block of dtype=`pointer.dtype.element_ty`
    :param val: The values to copy in case the expected value matches the contained value.
    :type val: Block of dtype=`pointer.dtype.element_ty`
    """
    cmp = base._to_tensor(cmp, _builder)
    val = base._to_tensor(val, _builder)
    return base._atomic_cas(pointer, cmp, val, _builder)


def _add_atomic_docstr(name: str) -> Callable[[CallableT], CallableT]:
    def _decorator(func: CallableT) -> CallableT:
        docstr = """
    Performs an atomic {name} at the memory location specified by :code:`pointer`.

    Return the data stored at :code:`pointer` before the atomic operation.

    :param pointer: The memory locations to apply {name}.
    :type pointer: Block of dtype=tr.PointerDType
    :param val: The values to {name} in the atomic object.
    :type val: Block of dtype=`pointer.dtype.element_ty`
    :param mask: If mask[idx] is false, do not apply {name}.
    :type mask: Block of tr.int1, optional
    """
        func.__doc__ = docstr.format(name=name)
        return func

    return _decorator


@base.builtin
@_add_atomic_docstr("exchange")
def atomic_xchg(pointer, val, mask=None, _builder: ir.builder = None):
    val = base._to_tensor(val, _builder)
    return base._atomic_xchg(pointer, val, mask, _builder)


@base.builtin
@_add_atomic_docstr("add")
def atomic_add(pointer, val, mask=None, _builder: ir.builder = None):
    val = base._to_tensor(val, _builder)
    return base._atomic_add(pointer, val, mask, _builder)


@base.builtin
@_add_atomic_docstr("max")
def atomic_max(pointer, val, mask=None, _builder: ir.builder = None):
    val = base._to_tensor(val, _builder)
    return base._atomic_max(pointer, val, mask, _builder)


@base.builtin
@_add_atomic_docstr("min")
def atomic_min(pointer, val, mask=None, _builder: ir.builder = None):
    val = base._to_tensor(val, _builder)
    return base._atomic_min(pointer, val, mask, _builder)


@base.builtin
@_add_atomic_docstr("logical and")
def atomic_and(pointer, val, mask=None, _builder: ir.builder = None):
    val = base._to_tensor(val, _builder)
    return base._atomic_and(pointer, val, mask, _builder)


@base.builtin
@_add_atomic_docstr("logical or")
def atomic_or(pointer, val, mask=None, _builder: ir.builder = None):
    val = base._to_tensor(val, _builder)
    return base._atomic_or(pointer, val, mask, _builder)


@base.builtin
@_add_atomic_docstr("logical xor")
def atomic_xor(pointer, val, mask=None, _builder: ir.builder = None):
    val = base._to_tensor(val, _builder)
    return base._atomic_xor(pointer, val, mask, _builder)


# -----------------------
# Conditioning
# -----------------------


@base.builtin
def where(condition, x, y, _builder: ir.builder = None) -> base.tensor:
    """
    Returns a tensor of elements from either :code:`x` or :code:`y`, depending on :code:`condition`.

    Note that :code:`x` and :code:`y` are always evaluated regardless of the value of :code:`condition`.

    If you want to avoid unintended memory operations, use the :code:`mask` arguments in `tr.load` and `tr.store` instead.

    The shape of :code:`x` and :code:`y` are both broadcast to the shape of :code:`condition`.
    :code:`x` and :code:`y` must have the data type.

    :param condition: When True (nonzero), yield x, otherwise yield y.
    :type condition: Block of tr.bool
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
def umulhi(x, y, _builder: ir.builder = None):
    x = base._to_tensor(x, _builder)
    y = base._to_tensor(y, _builder)
    return base._umulhi(x, y, _builder)


@base.builtin
def fdiv(x, y, ieee_rounding=False, _builder: ir.builder = None):
    ieee_rounding = base._constexpr_to_value(ieee_rounding)
    return base._fdiv(x, y, ieee_rounding, _builder)


def _add_math_1arg_docstr(name: str) -> Callable[[CallableT], CallableT]:
    def _decorator(func: CallableT) -> CallableT:
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
def exp(x, _builder: ir.builder = None):
    return base._exp(x, _builder)


@base.builtin
@_add_math_1arg_docstr("natural logarithm")
def log(x, _builder: ir.builder = None):
    return base._log(x, _builder)


@base.builtin
@_add_math_1arg_docstr("cosine")
def cos(x, _builder: ir.builder = None):
    return base._cos(x, _builder)


@base.builtin
@_add_math_1arg_docstr("sine")
def sin(x, _builder: ir.builder = None):
    return base._sin(x, _builder)


@base.builtin
@_add_math_1arg_docstr("square root")
def sqrt(x, _builder: ir.builder = None):
    return base._sqrt(x, _builder)


# -----------------------
# Reductions
# -----------------------


def _add_reduction_docstr(name: str) -> Callable[[CallableT], CallableT]:
    def _decorator(func: CallableT) -> CallableT:
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
def max(input, axis, _builder: ir.builder = None):
    axis = base._constexpr_to_value(axis)
    return base._max(input, axis, _builder)


@base.builtin
@_add_reduction_docstr("maximum index")
def argmax(input, axis, _builder: ir.builder = None):
    axis = base._constexpr_to_value(axis)
    return base._argmax(input, axis, _builder)


@base.builtin
@_add_reduction_docstr("minimum")
def min(input, axis, _builder: ir.builder = None):
    axis = base._constexpr_to_value(axis)
    return base._min(input, axis, _builder)


@base.builtin
@_add_reduction_docstr("minimum index")
def argmin(input, axis, _builder: ir.builder = None):
    axis = base._constexpr_to_value(axis)
    return base._argmin(input, axis, _builder)


@base.builtin
@_add_reduction_docstr("sum")
def sum(input, axis, _builder: ir.builder = None):
    axis = base._constexpr_to_value(axis)
    return base._sum(input, axis, _builder)


@base.builtin
@_add_reduction_docstr("xor sum")
def xor_sum(input, axis, _builder: ir.builder = None):
    axis = base._constexpr_to_value(axis)
    return base._xor_sum(input, axis, _builder)


# -----------------------
# Utilities
# -----------------------


@base.builtin
def globaltimer(_builder: ir.builder = None):
    return base._globaltimer(_builder)


@base.builtin
def clock(_builder: ir.builder = None):
    return base._clock(_builder)


# -----------------------
# Internal for debugging
# -----------------------


@base.builtin
def debug_barrier(_builder: ir.builder = None):
    return base._debug_barrier(_builder)


@base.builtin
def multiple_of(input, values, _builder: ir.builder = None):
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
def max_contiguous(input, values, _builder: ir.builder = None):
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


@jitlib.jit
def abs(x):
    return where(x >= 0, x, -x)


@jitlib.jit
def cdiv(x, div):
    """
    Computes the ceiling division of :code:`x` by :code:`div`

    :param x: the input number
    :type input: Block
    :param div: the divisor
    :param div: Block
    """
    return (x + div - 1) // div


@jitlib.jit
def minimum(x, y):
    """
    Computes the element-wise minimum of :code:`x` and :code:`y`.

    :param input: the first input tensor
    :type input: Block
    :param other: the second input tensor
    :type other: Block
    """
    return where(x < y, x, y)


@jitlib.jit
def maximum(x, y):
    """
    Computes the element-wise maximum of :code:`x` and :code:`y`.

    :param input: the first input tensor
    :type input: Block
    :param other: the second input tensor
    :type other: Block
    """
    return where(x > y, x, y)


@jitlib.jit
@_add_math_1arg_docstr("sigmoid")
def sigmoid(x):
    return 1 / (1 + exp(-x))


@jitlib.jit
@_add_math_1arg_docstr("softmax")
def softmax(x, ieee_rounding: constexpr = False):
    z = x - max(x, 0)
    num = exp(z)
    den = sum(num, 0)
    return fdiv(num, den, ieee_rounding)


@jitlib.jit
def ravel(x):
    """
    Returns a contiguous flattened view of :code:`x`

    :param x: the input tensor
    :type x: Block
    """
    return reshape(x, [x.numel])


@jitlib.jit
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


@jitlib.jit
def zeros_like(input):
    return zeros(input.shape, input.dtype)
