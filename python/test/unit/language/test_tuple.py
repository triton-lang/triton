import pytest
import triton
import triton.language as tl
from typing import NamedTuple
import torch


@triton.jit
def _tuple_increment(values):
    for i in tl.static_range(len(values)):
        values[i] = values[i] + 1
    return values


@triton.jit
def _tuple_index_func(Ptrs, values):
    for i in tl.static_range(len(values)):
        tl.store(Ptrs[i], values[i])


@triton.jit
def _tuple_index(_0, Ptrs, _1: tl.constexpr, values, _2, _3: tl.constexpr, _4):
    values = _tuple_increment(values)
    _tuple_index_func(Ptrs, values)


@pytest.mark.parametrize("size", [0, 1, 2, 3, 4])
def test_index(size, device):
    vals = tuple([i + 1 for i in range(size)])
    rets = tuple([torch.zeros((1, ), dtype=torch.float32, device=device) for _ in vals])
    _tuple_index[(1, )](0, rets, 0, vals, 0, 0, 0)
    assert vals == tuple([x.item() - 1 for x in rets])


# ----


@triton.jit
def _tuple_assign(XPtrs, YPtrs, values):
    # assign from tuple
    X0, X1 = XPtrs
    x0, x1 = values
    tl.store(X0, x0)
    tl.store(X1, x1)
    # assign to tuple
    Y0, Y1, Y2 = YPtrs
    Y = Y0, Y1, Y2
    y = x0, 10, x1
    tl.store(Y[0], y[0])
    tl.store(Y[1], y[1])
    tl.store(Y[2], y[2])


@pytest.mark.interpreter
def test_assign(device):
    vals = (2., 3.)
    x = tuple([torch.zeros((1, ), dtype=torch.float32, device=device) for _ in range(2)])
    y = tuple([torch.zeros((1, ), dtype=torch.float32, device=device) for _ in range(3)])
    _tuple_assign[(1, )](x, y, vals)
    assert x[0] == vals[0]
    assert x[1] == vals[1]
    assert y[0] == vals[0]
    assert y[1] == 10
    assert y[2] == vals[1]


# -------


@triton.jit
def _tuple_fn0(Ptr, cst2: tl.constexpr, tuple1):
    tl.static_assert(tuple1[1] is None)
    tl.store(Ptr + 5, cst2)
    tl.store(Ptr + 6, tuple1[0])
    tl.store(Ptr + 7, tl.load(tuple1[2][0]))
    tl.store(Ptr + 8, tuple1[2][1][0])
    tl.store(Ptr + 9, tl.load(tuple1[2][1][2]))


# test serialization/deserialization of tuple arguments in
# the frontend.
@triton.jit
def _tuple_serialize(Ptr, N1, tuple1, cst1: tl.constexpr, val1, tuple2):
    tl.static_assert(N1 is None)
    tl.static_assert(tuple1[1][1] is None)
    tl.static_assert(tuple1[1][3] == 4)
    tl.store(Ptr + 0, tl.load(tuple1[0]))
    tl.store(Ptr + 1, tuple1[1][0])
    tl.store(Ptr + 2, tl.load(tuple1[1][2]))
    tl.store(Ptr + 3, cst1 + val1)
    tl.store(Ptr + 4, tl.load(tuple2[0]))
    _tuple_fn0(Ptr, 15, (-1, None, tuple1))


@pytest.mark.interpreter
def test_serialize(device):
    x0 = torch.tensor([8], dtype=torch.int32, device=device)
    x1 = torch.tensor([12], dtype=torch.int32, device=device)
    y0 = torch.tensor([10], dtype=torch.int32, device=device)
    z = torch.empty((10, ), dtype=torch.int32, device=device)
    # we want to check that JIT specialization propagates to tuples:
    _tuple_serialize[(1, )](z, None, (x0, (1, None, x1, tl.constexpr(4))), 20, 1, (y0, ))
    ref = torch.tensor([8, 1, 12, 21, 10, 15, -1, 8, 1, 12], device=device)
    assert torch.equal(z, ref)


class Function(NamedTuple):
    fn: tl.constexpr
    captured: tuple


class Tensor(NamedTuple):
    ptr: any
    shape: tuple
    stride: tuple


@triton.jit
def _namedtuple_mask_func(Tensor, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < Tensor.shape[0]) & (offs_n[None, :] < Tensor.shape[1])
    return mask


@triton.jit
def _namedtuple_kernel(closure, _X, Y, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    X = Tensor(shape=_X.shape, ptr=_X.ptr, stride=_X.stride)
    Xs = X.ptr + offs_m[:, None] * X.stride[0] + offs_n[None, :] * X.stride[1]
    Ys = Y.ptr + offs_m[:, None] * Y.stride[0] + offs_n[None, :] * Y.stride[1]
    x = tl.load(Xs, mask=_namedtuple_mask_func(X, BLOCK_M, BLOCK_N), other=0)
    y = closure.fn(x, *closure.captured)
    tl.store(Ys, y, mask=_namedtuple_mask_func(Y, BLOCK_M, BLOCK_N))


@pytest.mark.interpreter
def test_namedtuple(device):
    x = torch.randn((32, 32), dtype=torch.float32, device=device)
    y = torch.empty((16, 16), dtype=torch.float32, device=device)
    a = torch.tensor([5.2], dtype=torch.float32, device=device)

    @triton.jit
    def mul(x, a):
        return x * tl.load(a)

    function = Function(mul, (a, ))
    tx = Tensor(x, x.shape, x.stride())
    ty = Tensor(y, y.shape, y.stride())
    _namedtuple_kernel[(1, )](function, tx, ty, 64, 64)
    assert torch.allclose(y, x[:16, :16] * a)
