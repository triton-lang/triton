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
    desc_mask = tl.reshape(desc_mask, x.shape)
    _0 = tl.zeros_like(left)
    ret = x ^ tl.where((left > right) ^ desc_mask, left ^ right, _0)
    return left, right, ret, desc_mask


@triton.jit
def sort_rows(X, Y1, Y2, R, N: tl.constexpr, N_DIMS: tl.constexpr):
    pid = tl.program_id(0)
    Xs = X + pid * N + tl.arange(0, N)
    x = tl.load(Xs)
    i: tl.constexpr = 2
    # desc_mask
    shape: tl.constexpr = [2**(i - 1), 2, 2**(N_DIMS - i)]
    desc_mask = tl.broadcast_to(tl.arange(0, 2)[None, :, None], shape)
    #
    y1, y2, r, m = compare_and_swap(x, desc_mask, i, N_DIMS)
    # tl.store(Y1 + pid * N + tl.arange(0, N), y1)
    # tl.store(Y2 + pid * N + tl.arange(0, N), y2)

    tl.store(R + pid * N + tl.arange(0, N), r)
    # tl.store(Y2 + pid * N + tl.arange(0, N), m)


N = 8
X = torch.tensor([3, 9, 1, 4, 2, 1, 9, 3], dtype=torch.int32, device="cuda")
Y1 = torch.empty_like(X)
Y2 = torch.empty_like(X)
R = torch.empty_like(X)
h = sort_rows[(1, )](X, Y1, Y2, R, N, int(math.log2(N)))
print(X)
print(Y1)
print(Y2)
print(R)

# x x
# x x
# x x
# x x

# we want
# 0 0
# 1 1
# 0 0
# 1 1
