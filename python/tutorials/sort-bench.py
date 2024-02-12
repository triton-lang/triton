import torch
import triton
import triton.language as tl


@triton.jit
def _compare_and_swap(x, flip, i: tl.constexpr, n_dims: tl.constexpr):
    n_outer: tl.constexpr = x.numel >> n_dims
    shape: tl.constexpr = [n_outer * 2**i, 2, 2**(n_dims - i - 1)]
    y = tl.reshape(x, shape)
    # slice left/right with 'stride' 2**(n_dims - i - 1)
    mask = tl.arange(0, 2)[None, :, None]
    left = tl.broadcast_to(tl.sum(y * (1 - mask), 1)[:, None, :], shape)
    right = tl.broadcast_to(tl.sum(y * mask, 1)[:, None, :], shape)
    left = tl.reshape(left, x.shape)
    right = tl.reshape(right, x.shape)
    # actual compare-and-swap
    idtype = tl.dtype(f'int{tl.constexpr(x.dtype.primitive_bitwidth)}')
    ileft = left.to(idtype, bitcast=True)
    iright = right.to(idtype, bitcast=True)
    ix = x.to(idtype, bitcast=True)
    ret = ix ^ tl.where((left > right) ^ flip, ileft ^ iright, tl.zeros_like(ix))
    return ret.to(x.dtype, bitcast=True)


@triton.jit
def _bitonic_merge(x, stage: tl.constexpr, order: tl.constexpr, n_dims: tl.constexpr):
    '''
    order_type 0 == ascending
    order_type 1 == descending
    order_type 2 == alternating
    '''
    n_outer: tl.constexpr = x.numel >> n_dims
    tl.static_assert(stage <= n_dims)
    # flip denotes whether to re-arrange sub-sequences of elements in ascending or
    # descending order.
    # if flip = 00000000... then all elements will be re-arranged ascendingly at this stage
    # if flip = 00110011... then all the elements will be re-arranged alternatingly (with
    # a stride of 2) at this stage
    if order == 2:
        shape: tl.constexpr = [n_outer * 2**(n_dims - 1 - stage), 2, 2**stage]
        flip = tl.reshape(tl.broadcast_to(tl.arange(0, 2)[None, :, None], shape), x.shape)
    else:
        flip = order
    # perform `stage` rounds of `compare-and-swap`
    for i in tl.static_range(stage):
        x = _compare_and_swap(x, flip, i + (n_dims - stage), n_dims)
    return x


@triton.jit
def sort(x, dim: tl.constexpr = None, descending: tl.constexpr = 0):
    # handle default dimension or check that it is the most minor dim
    _dim: tl.constexpr = len(x.shape) - 1 if dim is None else dim
    tl.static_assert(_dim == len(x.shape) - 1, "only minor dimension is currently supported")
    # iteratively run bitonic merge-sort steps
    n_dims: tl.constexpr = tl.math.log2(x.shape[_dim])
    for i in tl.static_range(1, n_dims + 1):
        x = _bitonic_merge(x, i, 2 if i < n_dims else descending, n_dims)
    return x


@triton.jit
def sort_rows(X, Y, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, descending: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # load
    Xs = X + offs_m[:, None] * BLOCK_N + offs_n[None, :]
    x = tl.load(Xs)
    # sort
    y = sort(x, dim=None)
    # x = tl.sort(x)
    # write-back
    Ys = Y + offs_m[:, None] * BLOCK_N + offs_n[None, :]
    tl.store(Ys, y)


M = 16384
BLOCK_M = 16
BLOCK_N = 128
X = torch.randn((M, BLOCK_N), dtype=torch.float32, device="cuda")
R = torch.empty_like(X)
descending = False
fn = lambda: sort_rows[(triton.cdiv(M, BLOCK_M), )](X, R, BLOCK_M, BLOCK_N, descending, num_warps=4)
fn()
print((R - torch.sort(X, dim=1, descending=descending).values).abs().max())
# fn = lambda: torch.sort(X, dim=1)
print(triton.testing.do_bench(fn))
print(triton.testing.do_bench(lambda: torch.sort(X, dim=1)))

# x x
# x x
# x x
# x x

# we want
# 0 0
# 1 1
# 0 0
# 1 1
