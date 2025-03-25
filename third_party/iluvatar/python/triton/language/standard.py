from __future__ import annotations

from ..runtime.jit import jit
from . import core
from . import math

# constexpr utilities (triton metaprogramming sucks)


def _unwrap_if_constexpr(o):
    return o.value if isinstance(o, core.constexpr) else o


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


# -----------------------
# Standard library
# -----------------------


@core._tensor_member_fn
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


@core._tensor_member_fn
@jit
@math._add_math_1arg_docstr("sigmoid")
def sigmoid(x):
    return 1 / (1 + math.exp(-x))


@core._tensor_member_fn
@jit
@math._add_math_1arg_docstr("softmax")
def softmax(x, ieee_rounding=False):
    z = x - max(x, 0)
    num = math.exp(z)
    den = sum(num, 0)
    return math.fdiv(num, den, ieee_rounding)


@core._tensor_member_fn
@jit
def ravel(x):
    """
    Returns a contiguous flattened view of :code:`x`.

    :param x: the input tensor
    :type x: Block
    """
    return core.reshape(x, [x.numel], can_reorder=True)


@jit
def swizzle2d(i, j, size_i, size_j, size_g):
    """
    Transforms indices of a row-major :code:`size_i * size_j` matrix into those
    of one where the indices are col-major for each group of :code:`size_g`
    rows.

    For example, for :code:`size_i = size_j = 4` and :code:`size_g = 2`, it will
    transform ::

        [[0 , 1 , 2 , 3 ],
         [4 , 5 , 6 , 7 ],
         [8 , 9 , 10, 11],
         [12, 13, 14, 15]]

    into ::

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
    size_g = core.minimum(size_i - off_i, size_g)
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
    """
    Creates a tensor of zeros with the same shape and type as a given tensor.
    """
    return zeros(input.shape, input.dtype)


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
def _elementwise_max(a, b):
    return core.maximum(a, b)


@core._tensor_member_fn
@jit
@core._add_reduction_docstr("maximum", return_indices_arg="return_indices",
                            tie_break_arg="return_indices_tie_break_left")
def max(input, axis=None, return_indices=False, return_indices_tie_break_left=True, keep_dims=False):
    input = core._promote_bfloat16_to_float32(input)
    if return_indices:
        if return_indices_tie_break_left:
            return core._reduce_with_indices(input, axis, _argmax_combine_tie_break_left, keep_dims=keep_dims)
        else:
            return core._reduce_with_indices(input, axis, _argmax_combine_tie_break_fast, keep_dims=keep_dims)
    else:
        if core.constexpr(input.dtype.primitive_bitwidth) < core.constexpr(32):
            if core.constexpr(input.dtype.is_floating()):
                input = input.to(core.float32)
            else:
                assert input.dtype.is_int(), "Expecting input to be integer type"
                input = input.to(core.int32)
        return core.reduce(input, axis, _elementwise_max, keep_dims=keep_dims)


@core._tensor_member_fn
@jit
@core._add_reduction_docstr("maximum index", tie_break_arg="tie_break_left")
def argmax(input, axis, tie_break_left=True, keep_dims=False):
    (_, ret) = max(input, axis, return_indices=True, return_indices_tie_break_left=tie_break_left, keep_dims=keep_dims)
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
def _elementwise_min(a, b):
    return core.minimum(a, b)


@core._tensor_member_fn
@jit
@core._add_reduction_docstr("minimum", return_indices_arg="return_indices",
                            tie_break_arg="return_indices_tie_break_left")
def min(input, axis=None, return_indices=False, return_indices_tie_break_left=True, keep_dims=False):
    input = core._promote_bfloat16_to_float32(input)
    if return_indices:
        if return_indices_tie_break_left:
            return core._reduce_with_indices(input, axis, _argmin_combine_tie_break_left, keep_dims=keep_dims)
        else:
            return core._reduce_with_indices(input, axis, _argmin_combine_tie_break_fast, keep_dims=keep_dims)
    else:
        if core.constexpr(input.dtype.primitive_bitwidth) < 32:
            if core.constexpr(input.dtype.is_floating()):
                input = input.to(core.float32)
            else:
                assert input.dtype.is_int(), "Expecting input to be integer type"
                input = input.to(core.int32)
        return core.reduce(input, axis, _elementwise_min, keep_dims=keep_dims)


@core._tensor_member_fn
@jit
@core._add_reduction_docstr("minimum index", tie_break_arg="tie_break_left")
def argmin(input, axis, tie_break_left=True, keep_dims=False):
    _, ret = min(input, axis, return_indices=True, return_indices_tie_break_left=tie_break_left, keep_dims=keep_dims)
    return ret


@jit
def _sum_combine(a, b):
    return a + b


# sum


@core._tensor_member_fn
@jit
@core._add_reduction_docstr("sum")
def sum(input, axis=None, keep_dims=False):
    input = core._promote_bfloat16_to_float32(input)
    return core.reduce(input, axis, _sum_combine, keep_dims=keep_dims)


@jit
def _xor_combine(a, b):
    return a ^ b


# xor sum


@core._tensor_member_fn
@core.builtin
@core._add_reduction_docstr("xor sum")
def xor_sum(input, axis=None, keep_dims=False, _builder=None, _generator=None):
    scalar_ty = input.type.scalar
    if not scalar_ty.is_int():
        raise ValueError("xor_sum only supported for integers")

    input = core._promote_bfloat16_to_float32(input, _builder=_builder)
    return core.reduce(input, axis, _xor_combine, keep_dims=keep_dims, _builder=_builder, _generator=_generator)


# cumsum


@core._tensor_member_fn
@jit
@core._add_scan_docstr("cumsum")
def cumsum(input, axis=0, reverse=False):
    # todo rename this to a generic function name
    input = core._promote_bfloat16_to_float32(input)
    return core.associative_scan(input, axis, _sum_combine, reverse)


# cumprod


@jit
def _prod_combine(a, b):
    return a * b


@core._tensor_member_fn
@jit
@core._add_scan_docstr("cumprod")
def cumprod(input, axis=0, reverse=False):
    # todo rename this to a generic function name
    input = core._promote_bfloat16_to_float32(input)
    return core.associative_scan(input, axis, _prod_combine, reverse)


# sort


@jit
def _compare_and_swap(x, flip, i: core.constexpr, n_dims: core.constexpr):
    n_outer: core.constexpr = x.numel >> n_dims
    shape: core.constexpr = [n_outer * 2**i, 2, 2**(n_dims - i - 1)]
    y = core.reshape(x, shape)
    # slice left/right with 'stride' 2**(n_dims - i - 1)
    mask = core.arange(0, 2)[None, :, None]
    left = core.broadcast_to(sum(y * (1 - mask), 1)[:, None, :], shape)
    right = core.broadcast_to(sum(y * mask, 1)[:, None, :], shape)
    left = core.reshape(left, x.shape)
    right = core.reshape(right, x.shape)
    # actual compare-and-swap
    idtype = core.get_int_dtype(bitwidth=x.dtype.primitive_bitwidth, signed=True)
    ileft = left.to(idtype, bitcast=True)
    iright = right.to(idtype, bitcast=True)
    ix = x.to(idtype, bitcast=True)
    ret = ix ^ core.where((left > right) ^ flip, ileft ^ iright, zeros_like(ix))
    return ret.to(x.dtype, bitcast=True)


@jit
def _bitonic_merge(x, stage: core.constexpr, order: core.constexpr, n_dims: core.constexpr):
    '''
    order_type 0 == ascending
    order_type 1 == descending
    order_type 2 == alternating
    '''
    n_outer: core.constexpr = x.numel >> n_dims
    core.static_assert(stage <= n_dims)
    # flip denotes whether to re-arrange sub-sequences of elements in ascending or
    # descending order.
    # if flip = 00000000... then all elements will be re-arranged ascendingly at this stage
    # if flip = 00110011... then all the elements will be re-arranged alternatingly (with
    # a stride of 2) at this stage
    if order == 2:
        shape: core.constexpr = [n_outer * 2**(n_dims - 1 - stage), 2, 2**stage]
        flip = core.reshape(core.broadcast_to(core.arange(0, 2)[None, :, None], shape), x.shape)
    else:
        flip = order
    # perform `stage` rounds of `compare-and-swap`
    for i in core.static_range(stage):
        x = _compare_and_swap(x, flip, i + (n_dims - stage), n_dims)
    return x


@core._tensor_member_fn
@jit
def sort(x, dim: core.constexpr = None, descending: core.constexpr = core.CONSTEXPR_0):
    # handle default dimension or check that it is the most minor dim
    _dim: core.constexpr = len(x.shape) - 1 if dim is None else dim
    core.static_assert(_dim == len(x.shape) - 1, "only minor dimension is currently supported")
    # iteratively run bitonic merge-sort steps
    n_dims: core.constexpr = _log2(x.shape[_dim])
    for i in core.static_range(1, n_dims + 1):
        x = _bitonic_merge(x, i, 2 if i < n_dims else descending, n_dims)
    return x


# flip


def _get_flip_dim(dim, shape):
    dim = _unwrap_if_constexpr(dim)
    shape = _unwrap_if_constexpr(shape)
    if dim is None:
        dim = len(shape) - 1
    assert dim == len(shape) - 1, "Currently only support flipping the last dimension"
    return core.constexpr(dim)


@core._tensor_member_fn
@jit
def flip(x, dim=None):
    """
    Flips a tensor `x` along the dimension `dim`.

    :param x: the first input tensor
    :type x: Block
    :param dim: the dimension to flip along (currently only final dimension supported)
    :type dim: int
    """
    core.static_assert(_is_power_of_two(x.shape[_get_flip_dim(dim, x.shape)]))
    core.static_assert(_is_power_of_two(x.numel))
    # # reshape the tensor to have all dimensions be 2.
    # # TODO: We shouldn't have to change the dimensions not sorted.
    steps: core.constexpr = _log2(x.numel)
    start: core.constexpr = _log2(x.numel) - _log2(x.shape[_get_flip_dim(dim, x.shape)])
    y = core.reshape(x, [2] * steps)
    y = core.expand_dims(y, start)
    flip = (core.arange(0, 2)[:, None] == 1 - core.arange(0, 2))
    for i in core.static_range(start, steps):
        flip2 = flip
        for j in core.static_range(0, steps + 1):
            if j != i and j != i + 1:
                flip2 = core.expand_dims(flip2, j)
        y = sum(y * flip2, i + 1, keep_dims=True)
    x = core.reshape(y, x.shape)
    return x


@jit
def interleave(a, b):
    """
    Interleaves the values of two tensors along their last dimension.

    The two tensors must have the same shape.

    Equivalent to `tl.join(a, b).reshape(a.shape[-1:] + [2 * a.shape[-1]])`
    """
    c = core.join(a, b)

    assert isinstance(c.shape, list)
    if len(c.shape) == 1:
        # We must have interleaved two scalars.
        return c
    else:
        # This `else` is necessary because Triton's AST parser doesn't
        # understand that if we take the `if` above we definitely don't run this
        # `else`.
        return core.reshape(c, c.shape[:-2] + [2 * c.shape[-2]])
