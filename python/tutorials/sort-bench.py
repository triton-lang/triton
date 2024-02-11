import torch
import math
import triton
import triton.language as tl


@triton.jit
def compare_and_swap(x, i: tl.constexpr, n_dims: tl.constexpr):
    shape: tl.constexpr = [2**i, 2, 2**(n_dims - i - 1)]
    y = tl.reshape(x, shape)
    mask = tl.arange(0, 2)[None, :, None]
    left = tl.broadcast_to(tl.sum(y * (1 - mask), 1)[:, None, :], shape)
    right = tl.broadcast_to(tl.sum(y * mask, 1)[:, None, :], shape)
    left = tl.reshape(left, x.shape)
    right = tl.reshape(right, x.shape)
    ileft = left.to(tl.int32, bitcast=True)
    iright = right.to(tl.int32, bitcast=True)
    ix = x.to(tl.int32, bitcast=True)
    _0 = tl.zeros_like(ix)
    ret = ix ^ tl.where((ileft > iright), ileft ^ iright, _0)
    ret = ret.to(x.dtype, bitcast=True)
    return left, right, ret


@triton.jit
def sort_rows(X, Y1, Y2, R, N: tl.constexpr, N_DIMS: tl.constexpr):
    pid = tl.program_id(0)
    Xs = X + pid * N + tl.arange(0, N)
    x = tl.load(Xs)
    y1, y2, r = compare_and_swap(x, 2, N_DIMS)
    tl.store(Y1 + pid * N + tl.arange(0, N), y1)
    tl.store(Y2 + pid * N + tl.arange(0, N), y2)
    tl.store(R + pid * N + tl.arange(0, N), r)


N = 8
X = torch.randint(10, (N, ), dtype=torch.int32, device="cuda")
Y1 = torch.empty_like(X)
Y2 = torch.empty_like(X)
R = torch.empty_like(X)
h = sort_rows[(1, )](X, Y1, Y2, R, N, int(math.log2(N)))
print(X)
print(Y1)
print(Y2)
print(R)
