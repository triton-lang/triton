from __future__ import annotations

from ..runtime.jit import jit
from . import core, math

# -----------------------
# Standard library
# -----------------------


@jit
def cdiv(x, div):
    """
    Computes the ceiling division of :code:`x` by :code:`div`

    :param x: the input number
    :type x: Block
    :param div: the divisor
    :param div: Block
    """
    return (x + div - 1) // div


@jit
@core._add_math_1arg_docstr("sigmoid")
def sigmoid(x):
    return 1 / (1 + core.exp(-x))


@jit
@core._add_math_1arg_docstr("softmax")
def softmax(x, ieee_rounding=False):
    z = x - max(x, 0)
    num = core.exp(z)
    den = sum(num, 0)
    return core.fdiv(num, den, ieee_rounding)


@jit
def ravel(x):
    """
    Returns a contiguous flattened view of :code:`x`.

    :param x: the input tensor
    :type x: Block
    """
    return core.view(x, [x.numel])


@jit
def swizzle2d(i, j, size_i, size_j, size_g):
    """
    Transforms indices of a row-major size_i*size_j matrix into those
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
def zeros(shape, dtype):
    """
    Returns a tensor filled with the scalar value 0 for the given :code:`shape` and :code:`dtype`.

    :param shape: Shape of the new array, e.g., (8, 16) or (8, )
    :type shape: tuple of ints
    :param dtype: Data-type of the new array, e.g., :code:`tl.float16`
    :type dtype: DType
    """
    return core.full(shape, 0, dtype)


@jit
def zeros_like(input):
    return zeros(input.shape, input.dtype)


@jit
def minimum(x, y):
    """
    Computes the element-wise minimum of :code:`x` and :code:`y`.

    :param input: the first input tensor
    :type input: Block
    :param other: the second input tensor
    :type other: Block
    """
    return math.min(x, y)


@jit
def maximum(x, y):
    """
    Computes the element-wise maximum of :code:`x` and :code:`y`.

    :param input: the first input tensor
    :type input: Block
    :param other: the second input tensor
    :type other: Block
    """
    return math.max(x, y)


# max and argmax


@jit
def _argmax_combine(value1, index1, value2, index2, tie_break_left):
    if tie_break_left:
        tie = value1 == value2 and index1 < index2
    else:
        tie = False
    gt = value1 > value2 or tie
    v_ret = core.where(gt, value1, value2)
    i_ret = core.where(gt, index1, index2)
    return v_ret, i_ret


@jit
def _argmax_combine_tie_break_left(value1, index1, value2, index2):
    return _argmax_combine(value1, index1, value2, index2, True)


@jit
def _argmax_combine_tie_break_fast(value1, index1, value2, index2):
    return _argmax_combine(value1, index1, value2, index2, False)


@jit
@core._add_reduction_docstr("maximum", return_indices_arg="return_indices",
                            tie_break_arg="return_indices_tie_break_left")
def max(input, axis=None, return_indices=False, return_indices_tie_break_left=True):
    input = core._promote_reduction_input(input)
    if return_indices:
        if return_indices_tie_break_left:
            return core._reduce_with_indices(input, axis, _argmax_combine_tie_break_left)
        else:
            return core._reduce_with_indices(input, axis, _argmax_combine_tie_break_fast)
    else:
        if core.constexpr(input.dtype.primitive_bitwidth) < core.constexpr(32):
            if core.constexpr(input.dtype.is_floating()):
                input = input.to(core.float32)
            else:
                assert input.dtype.is_integer_type()
                input = input.to(core.int32)
        return core.reduce(input, axis, maximum)


@jit
@core._add_reduction_docstr("maximum index", tie_break_arg="tie_break_left")
def argmax(input, axis, tie_break_left=True):
    (_, ret) = max(input, axis, return_indices=True, return_indices_tie_break_left=tie_break_left)
    return ret


# min and argmin


@jit
def _argmin_combine(value1, index1, value2, index2, tie_break_left):
    if tie_break_left:
        tie = value1 == value2 and index1 < index2
    else:
        tie = False
    lt = value1 < value2 or tie
    value_ret = core.where(lt, value1, value2)
    index_ret = core.where(lt, index1, index2)
    return value_ret, index_ret


@jit
def _argmin_combine_tie_break_left(value1, index1, value2, index2):
    return _argmin_combine(value1, index1, value2, index2, True)


@jit
def _argmin_combine_tie_break_fast(value1, index1, value2, index2):
    return _argmin_combine(value1, index1, value2, index2, False)


@jit
@core._add_reduction_docstr("minimum", return_indices_arg="return_indices",
                            tie_break_arg="return_indices_tie_break_left")
def min(input, axis=None, return_indices=False, return_indices_tie_break_left=True):
    input = core._promote_reduction_input(input)
    if return_indices:
        if return_indices_tie_break_left:
            return core._reduce_with_indices(input, axis, _argmin_combine_tie_break_left)
        else:
            return core._reduce_with_indices(input, axis, _argmin_combine_tie_break_fast)
    else:
        if core.constexpr(input.dtype.primitive_bitwidth) < 32:
            if core.constexpr(input.dtype.is_floating()):
                input = input.to(core.float32)
            else:
                assert input.dtype.is_integer_type()
                input = input.to(core.int32)
        return core.reduce(input, axis, minimum)


@jit
@core._add_reduction_docstr("minimum index", tie_break_arg="tie_break_left")
def argmin(input, axis, tie_break_left=True):
    _, ret = min(input, axis, return_indices=True, return_indices_tie_break_left=tie_break_left)
    return ret


@jit
def _sum_combine(a, b):
    return a + b


# sum


@jit
@core._add_reduction_docstr("sum")
def sum(input, axis=None):
    input = core._promote_reduction_input(input)
    return core.reduce(input, axis, _sum_combine)


@jit
def _xor_combine(a, b):
    return a ^ b


# xor sum


@core.builtin
@core._add_reduction_docstr("xor sum")
def xor_sum(input, axis=None, _builder=None, _generator=None):
    scalar_ty = input.type.scalar
    if not scalar_ty.is_int():
        raise ValueError("xor_sum only supported for integers")

    input = core._promote_reduction_input(input, _builder=_builder)
    return core.reduce(input, axis, _xor_combine, _builder=_builder, _generator=_generator)


# cumsum


@jit
@core._add_scan_docstr("cumsum")
def cumsum(input, axis=0):
    # todo rename this to a generic function name
    input = core._promote_reduction_input(input)
    return core.associative_scan(input, axis, _sum_combine)


# cumprod


@jit
def _prod_combine(a, b):
    return a * b


@jit
@core._add_scan_docstr("cumprod")
def cumprod(input, axis=0):
    # todo rename this to a generic function name
    input = core._promote_reduction_input(input)
    return core.associative_scan(input, axis, _prod_combine)


# sort


@jit
def _indicator(n_dims: core.constexpr, idx: core.constexpr, pos: core.constexpr):
    core.static_assert(idx < n_dims)
    core.static_assert((pos == 0) or (pos == 1))
    y = core.arange(0, 2)
    if pos == 0:
        y = 1 - y

    for n in core.static_range(0, n_dims):
        if n != n_dims - 1 - idx:
            y = core.expand_dims(y, n)
    return y


@jit
def _take_slice(x, n_dims: core.constexpr, idx: core.constexpr, pos: core.constexpr, keep_dim: core.constexpr = True):
    y = sum(x * _indicator(n_dims, idx, pos), n_dims - 1 - idx)
    if keep_dim:
        y = core.expand_dims(y, n_dims - 1 - idx)

    return y


@jit
def _compare_and_swap(x, desc_mask, n_dims: core.constexpr, idx: core.constexpr):
    l = _take_slice(x, n_dims, idx, 0)
    r = _take_slice(x, n_dims, idx, 1)

    x_int = x
    l_int = l
    r_int = r
    if x.dtype.is_floating():
        if core.constexpr(x.dtype.primitive_bitwidth) == 16:
            dtype_int = core.int16
        elif core.constexpr(x.dtype.primitive_bitwidth) == 32:
            dtype_int = core.int32
        elif core.constexpr(x.dtype.primitive_bitwidth) == 64:
            dtype_int = core.int64
        else:
            raise ValueError("Unsupported dtype")
        x_int = x.to(dtype_int, bitcast=True)
        l_int = l.to(dtype_int, bitcast=True)
        r_int = r.to(dtype_int, bitcast=True)
    desc_mask = desc_mask.to(x_int.dtype)
    zero = zeros_like(x_int)
    y = x_int ^ core.where((l > r) ^ desc_mask, l_int ^ r_int, zero)
    y = y.to(x.dtype, bitcast=True)
    return y


@jit
def _bitonic_merge(x, n_dims: core.constexpr, active_dims: core.constexpr, order_type: core.constexpr):
    '''
    order_type 0 == ascending
    order_type 1 == descending
    order_type 2 == alternating
    '''
    core.static_assert(active_dims <= n_dims)

    if order_type == 2:
        desc_mask = _indicator(n_dims, active_dims, 1)
    else:
        desc_mask = order_type

    for i in core.static_range(active_dims):
        x = _compare_and_swap(x, desc_mask, n_dims, active_dims - 1 - i)

    return x


def _log2(i: core.constexpr):
    log2 = 0
    n = i.value
    while n > 1:
        n >>= 1
        log2 += 1
    return core.constexpr(log2)


def _is_power_of_two(i: core.constexpr):
    n = i.value
    return core.constexpr((n & (n - 1)) == 0 and n != 0)


def _unwrap_if_constexpr(o):
    return o.value if isinstance(o, core.constexpr) else o


def _get_sort_dim(dim, shape):
    dim = _unwrap_if_constexpr(dim)
    shape = _unwrap_if_constexpr(shape)
    if dim is None:
        dim = len(shape) - 1
    assert dim == len(shape) - 1, "Currently only support sorting on the last dimension"
    return core.constexpr(dim)


@jit
def sort(x, dim=None, descending: core.constexpr = 0):
    core.static_assert(_is_power_of_two(x.shape[_get_sort_dim(dim, x.shape)]))
    core.static_assert(_is_power_of_two(x.numel))
    # reshape the tensor to have all dimensions be 2.
    # TODO: We shouldn't have to change the dimensions not sorted.
    y = core.reshape(x, [2] * _log2(x.numel))
    for i in core.static_range(1, _log2(x.shape[_get_sort_dim(dim, x.shape)]) + 1):
        y = _bitonic_merge(y, _log2(x.numel), i, (descending if
                                                  (i == _log2(x.shape[_get_sort_dim(dim, x.shape)])) else 2))

    x = core.reshape(y, x.shape)
    return x
