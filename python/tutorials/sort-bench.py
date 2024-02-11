import torch
import math
import triton
import triton.language as tl


@triton.jit
def compare_and_swap(x, desc_mask, i: tl.constexpr, n_dims: tl.constexpr):
    shape: tl.constexpr = [2**i, 2, 2**(n_dims - i - 1)]
    y = tl.reshape(x, shape)
    mask = tl.arange(0, 2)[None, :, None]
    left = tl.broadcast_to(tl.sum(y * (1 - mask), 1)[:, None, :], shape)
    right = tl.broadcast_to(tl.sum(y * mask, 1)[:, None, :], shape)
    # reshapes
    left = tl.reshape(left, x.shape)
    right = tl.reshape(right, x.shape)
    _0 = tl.zeros_like(left)
    ret = x ^ tl.where((left > right) ^ desc_mask, left ^ right, _0)
    return ret


@triton.jit
def bitonic_merge(x, N_DIMS: tl.constexpr, ACTIVE_DIMS: tl.constexpr, ORDER_TYPE: tl.constexpr):
    '''
    order_type 0 == ascending
    order_type 1 == descending
    order_type 2 == alternating
    '''
    tl.static_assert(ACTIVE_DIMS <= N_DIMS)
    if ORDER_TYPE == 2:
        shape: tl.constexpr = [2**(N_DIMS - 1 - ACTIVE_DIMS), 2, 2**ACTIVE_DIMS]
        desc_mask = tl.reshape(tl.broadcast_to(tl.arange(0, 2)[None, :, None], shape), x.shape)
    else:
        desc_mask = ORDER_TYPE
    for i in tl.static_range(ACTIVE_DIMS):
        x = compare_and_swap(x, desc_mask, i + (N_DIMS - ACTIVE_DIMS), N_DIMS)
    return x


@triton.jit
def sort_rows(X, Y1, Y2, R, N: tl.constexpr, N_DIMS: tl.constexpr):
    pid = tl.program_id(0)
    Xs = X + pid * N + tl.arange(0, N)
    x = tl.load(Xs)
    for i in tl.static_range(1, N_DIMS + 1):
        x = bitonic_merge(x, N_DIMS, i, 2 if i < N_DIMS else 0)
    # x = tl.sort(x)
    tl.store(R + pid * N + tl.arange(0, N), x)


M = 16384
N = 512
X = torch.randint(10, (M, N), dtype=torch.int32, device="cuda")
Y1 = torch.empty_like(X)
Y2 = torch.empty_like(X)
R = torch.empty_like(X)
fn = lambda: sort_rows[(M, )](X, Y1, Y2, R, N, int(math.log2(N)))
fn()
print((R - torch.sort(X, dim=1).values).abs().max())
# fn = lambda: torch.sort(X, dim=1)
# print(triton.testing.do_bench(fn))

# x x
# x x
# x x
# x x

# we want
# 0 0
# 1 1
# 0 0
# 1 1
