import torch
import triton
import triton.language as tl


@triton.jit
def _indicator(n_dims: tl.constexpr, idx: tl.constexpr, pos: tl.constexpr):
    tl.static_assert(idx < n_dims)
    tl.static_assert((pos == 0) or (pos == 1))
    y = tl.arange(0, 2)
    if pos == 0:
        y = 1 - y

    for n in tl.static_range(0, n_dims):
        if n != n_dims - 1 - idx:
            y = tl.expand_dims(y, n)
    return y


@triton.jit
def _cast_to_int(x):
    y = x
    if x.dtype.is_floating():
        if tl.constexpr(x.dtype.primitive_bitwidth) == 16:
            dtype_int = tl.int16
        elif tl.constexpr(x.dtype.primitive_bitwidth) == 32:
            dtype_int = tl.int32
        elif tl.constexpr(x.dtype.primitive_bitwidth) == 64:
            dtype_int = tl.int64
        else:
            raise ValueError("Unsupported dtype")
        y = x.to(dtype_int, bitcast=True)
    return y


@triton.jit
def _take_slice(x, n_dims: tl.constexpr, idx: tl.constexpr, pos: tl.constexpr, keep_dim: tl.constexpr = True):
    y = tl.sum(x * _indicator(n_dims, idx, pos).to(x.dtype), n_dims - 1 - idx)
    if keep_dim:
        y = tl.expand_dims(y, n_dims - 1 - idx)

    return y


@triton.jit
def _compare_and_swap(x, desc_mask, n_dims: tl.constexpr, idx: tl.constexpr):
    x_int = _cast_to_int(x)
    l_int = _take_slice(x_int, n_dims, idx, 0)
    r_int = _take_slice(x_int, n_dims, idx, 1)
    l = l_int.to(x.dtype, bitcast=True)
    r = r_int.to(x.dtype, bitcast=True)

    desc_mask = desc_mask.to(x_int.dtype)
    zero = tl.zeros_like(x_int)
    y = x_int ^ tl.where((l > r) ^ desc_mask, l_int ^ r_int, zero)
    y = y.to(x.dtype, bitcast=True)
    return y


@triton.jit
def _bitonic_merge(x, n_dims: tl.constexpr, active_dims: tl.constexpr, order_type: tl.constexpr):
    '''
    order_type 0 == ascending
    order_type 1 == descending
    order_type 2 == alternating
    '''
    tl.static_assert(active_dims <= n_dims)

    if order_type == 2:
        desc_mask = _indicator(n_dims, active_dims, 1)
    else:
        desc_mask = order_type

    for i in tl.static_range(active_dims):
        x = _compare_and_swap(x, desc_mask, n_dims, active_dims - 1 - i)

    return x


def _log2(i: tl.constexpr):
    log2 = 0
    n = i.value
    while n > 1:
        n >>= 1
        log2 += 1
    return tl.constexpr(log2)


def _is_power_of_two(i: tl.constexpr):
    n = i.value
    return tl.constexpr((n & (n - 1)) == 0 and n != 0)


def _unwrap_if_constexpr(o):
    return o.value if isinstance(o, tl.constexpr) else o


def _get_sort_dim(dim, shape):
    dim = _unwrap_if_constexpr(dim)
    shape = _unwrap_if_constexpr(shape)
    if dim is None:
        dim = len(shape) - 1
    assert dim == len(shape) - 1, "Currently only support sorting on the last dimension"
    return tl.constexpr(dim)


@triton.jit
def sort(x, dim=None, descending: tl.constexpr = 0):
    tl.static_assert(_is_power_of_two(x.shape[_get_sort_dim(dim, x.shape)]))
    tl.static_assert(_is_power_of_two(x.numel))
    # reshape the tensor to have all dimensions be 2.
    # TODO: We shouldn't have to change the dimensions not sorted.
    y = tl.reshape(x, [2] * _log2(x.numel))
    for i in tl.static_range(1, _log2(x.shape[_get_sort_dim(dim, x.shape)]) + 1):
        y = _bitonic_merge(y, _log2(x.numel), i, (descending if
                                                  (i == _log2(x.shape[_get_sort_dim(dim, x.shape)])) else 2))

    x = tl.reshape(y, x.shape)
    return x


@triton.jit
def sort_rows(X, Y, N: tl.constexpr):
    pid = tl.program_id(0)
    Xs = X + pid * N + tl.arange(0, N)
    Ys = Y + pid * N + tl.arange(0, N)
    tl.store(Ys, sort(tl.load(Xs)))


M, N = 1, 128
X = torch.randn((M, N), dtype=torch.float32, device="cuda")
Y = torch.empty_like(X)
h = sort_rows[(M, )](X, Y, N)
print(h.asm["ttgir"])
print((torch.sort(X).values - Y).abs())
