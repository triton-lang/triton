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
    x0, x1, _ = values
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
    vals = (2., 3., None)
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
def _namedtuple_create_func0(shape, ptr, stride):
    return Tensor(shape=shape, ptr=ptr, stride=stride)


@triton.jit
def _namedtuple_create_func1(shape, ptr, stride):
    tensor = Tensor(shape=shape, ptr=ptr, stride=stride)
    return tensor


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
    X = _namedtuple_create_func0(_X.shape, _X.ptr, _X.stride)
    Y = _namedtuple_create_func1(Y.shape, Y.ptr, Y.stride)
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


@pytest.mark.interpreter
def test_eq(device):

    @triton.jit
    def fn(ret_ptrs):
        tl.store(ret_ptrs + 0, (1, 2) == (1, 2))
        tl.store(ret_ptrs + 1, (1, 2) == (1, 1))
        tl.store(ret_ptrs + 2, tl.tuple((1, 2)) == (1, 2))
        tl.store(ret_ptrs + 3, tl.tuple((1, 2)) == (1, 3))

    rets = torch.zeros((4, ), dtype=torch.int32, device=device)
    fn[(1, )](rets)
    assert rets[0].item() == 1
    assert rets[1].item() == 0
    assert rets[2].item() == 1
    assert rets[3].item() == 0


@pytest.mark.interpreter
def test_add(device):

    @triton.jit
    def fn(ret_ptrs):
        tuple0 = ((0, 1)) + (2, 3)
        for i in tl.static_range(4):
            tl.store(ret_ptrs + i, tuple0[i])
        tuple1 = tl.tuple((4, 5)) + (6, 7)
        for i in tl.static_range(4):
            tl.store(ret_ptrs + 4 + i, tuple1[i])

    rets = torch.zeros((8, ), dtype=torch.int32, device=device)
    fn[(1, )](rets)
    torch.testing.assert_close(rets.cpu(), torch.arange(8, dtype=torch.int32))


def test_passing_tuple_with_constexpr(device):

    @triton.jit
    def m_to_the_n(X, shape: tl.constexpr, strides, m_n):
        Xs = X + tl.arange(0, shape[0])[:, None] * strides[0] + tl.arange(0, shape[1])[None, :] * strides[1]
        # Include a for loop to ensure strides[1] is lifted into a constexpr
        # (otherwise cloning the local scope will fail).
        data = tl.load(Xs)
        for i in tl.range(0, m_n[1]):
            data = m_n[0] * data
        tl.store(Xs, data)

    x = torch.arange(0, 64, device=device).reshape(8, 8)
    expected_x = 8 * x.clone()
    m_to_the_n[(1, )](x, x.shape, x.stride(), (2, 3))
    torch.testing.assert_close(x, expected_x, rtol=0, atol=0)


def test_passing_tuple_to_make_tensor_descriptor(device, with_allocator):

    from triton.language.core import builtin

    @builtin
    def is_constexpr(v, _semantic=None):
        return isinstance(v, tl.constexpr)

    @triton.jit
    def m_to_the_n(X_base, shape, strides, m_n, BLOCK_DIM: tl.constexpr):
        tl.static_assert(is_constexpr(strides[1]))
        X = tl.make_tensor_descriptor(
            X_base,
            shape=shape,
            strides=strides,
            block_shape=[BLOCK_DIM, BLOCK_DIM],
        )
        # Make sure tl.make_tensor_descriptor didn't modify strides (i.e. didn't unwrap the constexpr)
        tl.static_assert(is_constexpr(strides[1]))
        data = X.load([0, 0])
        # Include a for loop to ensure strides[1] is lifted into a constexpr
        # (otherwise cloning the local scope will fail).
        for i in tl.range(0, m_n[1]):
            data = m_n[0] * data
        X.store([0, 0], data)

    x = torch.arange(0, 16, device=device).reshape(4, 4)
    expected_x = 8 * x.clone()
    m_to_the_n[(1, )](x, x.size(), x.stride(), (2, 3), x.size(0))
    torch.testing.assert_close(x, expected_x, rtol=0, atol=0)


@pytest.mark.parametrize("init_val, idx, new_val, expected_ty",
                         [((5, 6, 7), 0, 32.1, tl.float32),  # i32 > fp32
                          ((5, 6, 7), 2, tl.constexpr(20), tl.int32),  # i32 > constexpr which converts to dtype
                          ((5, tl.constexpr(6), 7), 1, 32.1, tl.float32),  # constexpr > fp32
                          ((5, 1, 7), 1, 20, tl.int32),  # constexpr > i32 but with specialization of 1
                          ])
def test_modifying_tuples(init_val, idx, new_val, expected_ty):

    @triton.jit
    def set_tuple_value_at_idx(tuple_value, idx: tl.constexpr, new_value, expected_type: tl.constexpr):
        before_tuple_type: tl.constexpr = tuple_value.type
        before_type: tl.constexpr = tuple_value[idx].type
        tl.static_print(before_type, expected_type)

        # Make sure the underlying tuple_type matches the tuple's value's types
        tl.static_assert(before_type == tuple_value.type[idx])
        # Update the value to have a different type
        tuple_value[idx] = new_value
        # Make sure the tuple's type and the tuple's value's type were updated
        tl.static_print(new_value, tuple_value[idx].type, expected_type)
        tl.static_assert(tuple_value[idx].type == expected_type)
        tl.static_assert(tuple_value.type[idx] == expected_type)

        # Make sure the tuple type updated when the type updated
        tl.static_assert(before_type == expected_type or before_tuple_type != tuple_value.type)

    set_tuple_value_at_idx[(1, )](init_val, idx, new_val, expected_ty)
