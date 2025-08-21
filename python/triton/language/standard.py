from __future__ import annotations

from ..runtime.jit import jit, constexpr_function
from . import core
from . import math

# constexpr utilities


@constexpr_function
def _log2(i):
    log2 = 0
    n = i
    while n > 1:
        n >>= 1
        log2 += 1
    return log2


@constexpr_function
def _is_power_of_two(i):
    return (i & (i - 1)) == 0 and i != 0


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
    :type div: Block
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
def softmax(x, dim=None, keep_dims=False, ieee_rounding=False):
    if dim is None:
        _dim: core.constexpr = 0
    else:
        _dim: core.constexpr = dim
    z = x - max(x, _dim, keep_dims=keep_dims)
    num = math.exp(z)
    den = sum(num, _dim, keep_dims=keep_dims)
    return math.fdiv(num, den, ieee_rounding)


@core._tensor_member_fn
@jit
def ravel(x, can_reorder=False):
    """
    Returns a contiguous flattened view of :code:`x`.

    :param x: the input tensor
    :type x: Block
    """
    return core.reshape(x, [x.numel], can_reorder=can_reorder)


@jit
def swizzle2d(i, j, size_i, size_j, size_g):
    """
    Transforms the indices of a row-major `size_i * size_j` matrix into
    the indices of a column-major matrix for each group of `size_g` rows.

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
    # linear index with respect to the first element in this group
    ij = ij % size_gj
    # new row and column indices
    new_i = off_i + ij % size_g
    new_j = ij // size_g
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
    Returns a tensor of zeros with the same shape and type as a given tensor.

    :param input: input tensor
    :type input: Tensor
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


@constexpr_function
def _pick_sum_dtype(in_dtype, dtype):
    if dtype is not None:
        return dtype

    # For integer bitwidths less than 32, pick int32 with the same sign to
    # avoid overflow.
    out_dtype = None
    if in_dtype.is_int_signed():
        out_dtype = core.int32 if in_dtype.int_bitwidth < 32 else None
    elif in_dtype.is_int_unsigned():
        out_dtype = core.uint32 if in_dtype.int_bitwidth < 32 else None
    return out_dtype


@core._tensor_member_fn
@jit
@core._add_reduction_docstr("sum", dtype_arg="dtype")
def sum(input, axis=None, keep_dims=False, dtype: core.constexpr = None):
    # Pick a default dtype for the reduction if one was not specified.
    out_dtype: core.constexpr = _pick_sum_dtype(input.dtype, dtype)

    if out_dtype is not None:
        input = input.to(out_dtype)
    return core.reduce(input, axis, _sum_combine, keep_dims=keep_dims)


@jit
def _xor_combine(a, b):
    return a ^ b


# xor sum


@core._tensor_member_fn
@jit
@core._add_reduction_docstr("xor sum")
def xor_sum(input, axis=None, keep_dims=False):
    core.static_assert(input.type.scalar.is_int(), "xor_sum only supported for integers")
    return core.reduce(input, axis, _xor_combine, keep_dims=keep_dims)


# or reduction


@jit
def _or_combine(x, y):
    return x | y


@core._tensor_member_fn
@jit
@core._add_reduction_docstr("reduce_or")
def reduce_or(input, axis, keep_dims=False):
    core.static_assert(input.type.scalar.is_int(), "reduce_or only supported for integers")
    return core.reduce(input, axis, _or_combine, keep_dims=keep_dims)


# cumsum


@core._tensor_member_fn
@jit
@core._add_scan_docstr("cumsum", dtype_arg="dtype")
def cumsum(input, axis=0, reverse=False, dtype: core.constexpr = None):
    # todo rename this to a generic function name

    input = core._promote_bfloat16_to_float32(input)
    out_dtype: core.constexpr = _pick_sum_dtype(input.dtype, dtype)

    if out_dtype is not None:
        input = input.to(out_dtype)

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
def _indicator(n_dims: core.constexpr, j: core.constexpr):
    ar = core.arange(0, 2)
    ar = core.reshape(ar, [1] * (n_dims - j - 1) + [2] + [1] * j)
    return ar


@jit
def _compare_and_swap(x, flip, i: core.constexpr):
    # compare-and-swap on the ith *innermost* dimension
    n_dims: core.constexpr = _log2(x.numel)

    # flip along middle dimension (the bitwise XORs will be optimised away):
    idtype = core.get_int_dtype(bitwidth=x.dtype.primitive_bitwidth, signed=True)
    ix = x.to(idtype, bitcast=True)
    iy = ix ^ xor_sum(ix, n_dims - 1 - i, True)
    y = iy.to(x.dtype, bitcast=True)

    # determines whether we are in the right (rather than left) position along the axis:
    is_right = _indicator(n_dims, i)

    # conditional swap:
    ret = core.where((x > y) != (flip ^ is_right), y, x)
    return ret


@jit
def _bitonic_merge_hypercube(x, stage: core.constexpr, order: core.constexpr):
    '''
    order_type 0 == ascending
    order_type 1 == descending
    order_type 2 == alternating
    '''
    # flip denotes whether to re-arrange sub-sequences of elements in ascending or
    # descending order.
    # if flip = 00000000... then all elements will be re-arranged ascendingly at this stage
    # if flip = 00110011... then all the elements will be re-arranged alternatingly (with
    # a stride of 2) at this stage
    if order == 2:
        flip = _indicator(_log2(x.numel), stage)
    else:
        flip = order
    # perform `stage` rounds of `compare-and-swap`
    for i in core.static_range(stage):
        x = _compare_and_swap(x, flip, stage - 1 - i)
    return x


@jit
def _bitonic_merge(x, stage: core.constexpr, order: core.constexpr, n_dims: core.constexpr):
    h = core.reshape(x, [2] * _log2(x.numel))
    h = _bitonic_merge_hypercube(h, stage, order)
    x = core.reshape(h, x.shape)
    return x


@jit
def sort_impl(x, k: core.constexpr = None, dim: core.constexpr = None, descending: core.constexpr = core.CONSTEXPR_0):
    """
    Sorts a tensor along a specified dimension.

    :param x: The input tensor to be sorted.
    :type x: Tensor
    :param dim: The dimension along which to sort the tensor. If None, the tensor is sorted along the last dimension. Currently, only sorting along the last dimension is supported.
    :type dim: int, optional
    :param k: the number of top elements to select. If none, assume k = x.shape[dim]
    :type k: int, optional
    :param descending: If set to True, the tensor is sorted in descending order. If set to False, the tensor is sorted in ascending order.
    :type descending: bool, optional
    """
    # handle default dimension or check that it is the most minor dim
    _dim: core.constexpr = len(x.shape) - 1 if dim is None else dim
    core.static_assert(_dim == len(x.shape) - 1, "only minor dimension is currently supported")

    log_n: core.constexpr = _log2(x.shape[_dim])
    log_k: core.constexpr = log_n if k is None else _log2(k)

    n_dims: core.constexpr = _log2(x.numel)

    # reshape to hypercube:
    h = core.reshape(x, [2] * n_dims)

    # run first log_k bitonic sort iterations:
    for i in core.static_range(1, log_k + 1):
        h = _bitonic_merge_hypercube(h, i, 2 if i < log_n else descending)

    # select top k elements using bitonic top-k
    # https://www.doc.ic.ac.uk/~hlgr/pdfs/MassivelyParallelTopK.pdf
    for i in core.static_range(log_k + 1, log_n + 1):
        h = max(h, axis=(_log2(h.numel) - 1 - log_k)) if descending else min(h, axis=(_log2(h.numel) - 1 - log_k))
        h = _bitonic_merge_hypercube(h, log_k, 2 if i < log_n else descending)

    # reshape back:
    x = core.reshape(h, x.shape[:-1] + [2**log_k])
    return x


@jit
def sort(x, dim: core.constexpr = None, descending: core.constexpr = core.CONSTEXPR_0):
    return sort_impl(x, dim=dim, descending=descending)


@jit
def topk(x, k: core.constexpr, dim: core.constexpr = None):
    return sort_impl(x, k=k, dim=dim, descending=True)


@jit
def bitonic_merge(x, dim: core.constexpr = None, descending: core.constexpr = core.CONSTEXPR_0):
    # handle default dimension or check that it is the most minor dim
    _dim: core.constexpr = len(x.shape) - 1 if dim is None else dim
    core.static_assert(_dim == len(x.shape) - 1, "only minor dimension is currently supported")
    n_dims: core.constexpr = _log2(x.shape[-1])
    return _bitonic_merge(x, n_dims, descending, n_dims)


@constexpr_function
def _get_flip_dim(dim, shape):
    if dim is None:
        dim = len(shape) - 1
    if dim < 0:  # flip doesn't work if dim < 0 because the xor-swap for loop will start/end at the wrong index
        dim += len(shape)
    return dim


@core._tensor_member_fn
@jit
def flip(x, dim=None):
    """
    Flips a tensor `x` along the dimension `dim`.

    :param x: the first input tensor
    :type x: Block
    :param dim: the dimension to flip along
    :type dim: int
    """
    core.static_assert(-len(x.shape) <= dim and dim < len(x.shape))
    _dim: core.constexpr = _get_flip_dim(dim, x.shape)
    core.static_assert(_is_power_of_two(x.shape[_dim]))
    steps: core.constexpr = _log2(x.shape[_dim])

    # reshape the swap dimension to (2, 2, ..., 2)
    idtype = core.get_int_dtype(bitwidth=x.dtype.primitive_bitwidth, signed=True)
    y = core.reshape(x.to(idtype, bitcast=True), x.shape[:_dim] + [2] * steps + x.shape[_dim + 1:])
    for i in core.static_range(steps):
        y = y ^ xor_sum(y, _dim + i, True)
    x = core.reshape(y, x.shape).to(x.dtype, bitcast=True)
    return x


@jit
def interleave(a, b):
    """
    Interleaves the values of two tensors along their last dimension. The two tensors must have the same shape.
    Equivalent to `tl.join(a, b).reshape(a.shape[:-1] + [2 * a.shape[-1]])`

    :param a: The first input tensor.
    :type a: Tensor
    :param b: The second input tensor.
    :type b: Tensor
    """
    c = core.join(a, b)

    if len(c.shape) == 1:
        # We must have interleaved two scalars.
        return c
    else:
        # This `else` is necessary because Triton's AST parser doesn't
        # understand that if we take the `if` above we definitely don't run this
        # `else`.
        return core.reshape(c, c.shape[:-2] + [2 * c.shape[-2]])
