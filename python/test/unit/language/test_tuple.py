import pytest
import triton
import triton.language as tl
from typing import NamedTuple
import torch


@triton.jit
def _tuple_increment(values):
    return tl.tuple([v + 1 for v in values])


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


@triton.jit
def _tuple_ret(a, b):
    return a + b, \
        a - b, \
        a * b


@pytest.mark.interpreter
def test_assign_return(device):

    @triton.jit
    def with_fn(X, Y, A, B, C):
        x = tl.load(X)
        y = tl.load(Y)
        a, b, c = _tuple_ret(x, y)
        tl.store(A, a)
        tl.store(B, b)
        tl.store(C, c)

    @triton.jit
    def without_fn(X, Y, A, B, C):
        x = tl.load(X)
        y = tl.load(Y)
        a, b, c = x + y, x - y, x * y
        tl.store(A, a)
        tl.store(B, b)
        tl.store(C, c)

    x = torch.tensor([1.3], device=device, dtype=torch.float32)
    y = torch.tensor([1.9], device=device, dtype=torch.float32)
    a_tri = torch.tensor([0], device=device, dtype=torch.float32)
    b_tri = torch.tensor([0], device=device, dtype=torch.float32)
    c_tri = torch.tensor([0], device=device, dtype=torch.float32)
    for kernel in [with_fn, without_fn]:
        kernel[(1, )](x, y, a_tri, b_tri, c_tri, num_warps=1)
        a_ref, b_ref, c_ref = x + y, x - y, x * y
        assert a_tri == a_ref
        assert b_tri == b_ref
        assert c_tri == c_ref


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


def test_passing_tuple_with_mixed_constexpr_and_non_constexpr_values(device):

    @triton.jit
    def kernel(out_ptr, values):
        tl.static_assert(values[1].type == tl.constexpr_type(3))
        tl.store(out_ptr, tl.load(values[0]) + values[1])

    x = torch.tensor([8], dtype=torch.int32, device=device)
    out = torch.empty_like(x)
    kernel[(1, )](out, (x, tl.constexpr(3)))
    assert out.item() == 11


@triton.jit
def _nested_tuple_kernel(x):
    # This creates a new scope, which will force a copy of liveins. It's
    # important for this to happen as it forces IR flattening/unflattening,
    # which relies on the types being correct for the roundtrip to succeed.
    for _ in range(1):
        tl.static_assert(x[1][0] == 2)


def test_passing_nested_tuple_with_constexpr(device):
    _nested_tuple_kernel[(1, )](((1, ), (tl.constexpr(2), )))


def test_passing_nested_tuple_with_constexpr_and_jit_hook(device, fresh_knobs):
    # get the serialized specialization data
    specialization_data = None

    def cache_hook(*args, **kwargs):
        nonlocal specialization_data
        specialization_data = kwargs["compile"]["specialization_data"]

    fresh_knobs.runtime.jit_cache_hook = cache_hook

    device = getattr(torch, device).current_device()

    # Clear the existing cache for this device to ensure that the hook is called;
    # This is needed because the kernel is shared between multiple tests and may
    # already have been compiled for this device.
    _nested_tuple_kernel.device_caches[device][0].clear()

    warmup_run = _nested_tuple_kernel.warmup(((1, ), (tl.constexpr(2), )), grid=(1, ))
    assert warmup_run is not None

    assert specialization_data is not None

    preload_run = _nested_tuple_kernel.preload(specialization_data)
    assert preload_run is not None

    assert warmup_run.hash == preload_run.hash


def test_passing_tuple_to_make_tensor_descriptor(device, with_allocator):

    @triton.jit
    def m_to_the_n(X_base, shape, strides, m_n, BLOCK_DIM: tl.constexpr):
        tl.static_assert(isinstance(strides[1].type, tl.constexpr_type))
        X = tl.make_tensor_descriptor(
            X_base,
            shape=shape,
            strides=strides,
            block_shape=[BLOCK_DIM, BLOCK_DIM],
        )
        # Make sure tl.make_tensor_descriptor didn't modify strides (i.e. didn't unwrap the constexpr)
        tl.static_assert(isinstance(strides[1].type, tl.constexpr_type))
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


def test_modifying_tuples():

    @triton.jit
    def set_tuple_value_at_idx():
        t = tl.tuple([5, 6, 7])
        t[0] = 0

    with pytest.raises(triton.CompilationError):
        set_tuple_value_at_idx[(1, )]()


@pytest.mark.interpreter
def test_tuple_logic():

    @triton.jit
    def tuple_logic_kernel():

        # arity-2 BoolOps:
        tl.static_assert(((3, 4) or (5, 6)) == (3, 4))
        tl.static_assert(((3, 4) and (5, 6)) == (5, 6))
        tl.static_assert(((3, 4) and ()) == ())
        tl.static_assert((() or (5, 6)) == (5, 6))

        # arity-3 BoolOps:
        tl.static_assert(((1, 2) and (3, 4) and (5, 6)) == (5, 6))
        tl.static_assert(((1, 2) or (3, 4) or (5, 6)) == (1, 2))

        # constexpr short-circuiting over dynamic argument:
        tl.static_assert((() and tl.program_id(0)) == ())

    tuple_logic_kernel[(1, )]()


@pytest.mark.interpreter
def test_tuple_float():

    @triton.jit
    def _namedtuple_float_tuple_kernel():
        x, y = float("-inf"), float("inf")  # noqa: F841

    _namedtuple_float_tuple_kernel[(1, )]()


@triton.constexpr_function
def passthrough_constexpr(x):
    return x


class TrivialTuple(NamedTuple):
    foo: tl.constexpr


@pytest.mark.interpreter
def test_tuple_constexpr_function():

    @triton.jit
    def kernel():
        tl.static_assert(passthrough_constexpr(TrivialTuple(0)).foo == 0)

    kernel[(1, )]()


@pytest.mark.interpreter
def test_constexpr_tuple_arg_unpack(device):
    """A Python tuple passed as tl.constexpr must support unpacking
    (`d0, d1, d2 = shape`) in both value positions (arithmetic) and
    shape positions (tl.zeros / tl.full / tl.reshape)."""

    @triton.jit
    def kernel_value(out_ptr, shape: tl.constexpr):
        d0: tl.constexpr
        d1: tl.constexpr
        d2: tl.constexpr
        d0, d1, d2 = shape
        x = tl.full((1, ), d0 * 100 + d1 * 10 + d2, dtype=tl.int32)
        tl.store(out_ptr + tl.arange(0, 1), x)

    @triton.jit
    def kernel_shape(out_ptr, shape: tl.constexpr):
        d0: tl.constexpr
        d1: tl.constexpr
        d0, d1 = shape
        x = tl.full((d0, d1), 1.0, dtype=tl.float32)
        x = tl.reshape(x, (d0 * d1, ))
        tl.store(out_ptr + tl.arange(0, d0 * d1), x)

    out = torch.zeros(1, dtype=torch.int32, device=device)
    kernel_value[(1, )](out, shape=(8, 4, 2))
    assert out.item() == 842

    out = torch.zeros(4, dtype=torch.float32, device=device)
    kernel_shape[(1, )](out, shape=(2, 2))
    assert out.sum().item() == 4.0


@pytest.mark.interpreter
def test_constexpr_nested_tuple_arg_unpack(device):
    """A nested Python tuple passed as tl.constexpr must support nested unpacking
    (`(d0, d1), d2 = shape`) without crashing."""

    @triton.jit
    def kernel_nested(out_ptr, shape: tl.constexpr):
        d0: tl.constexpr
        d1: tl.constexpr
        d2: tl.constexpr
        (d0, d1), d2 = shape
        x = tl.full((1, ), d0 * 100 + d1 * 10 + d2, dtype=tl.int32)
        tl.store(out_ptr + tl.arange(0, 1), x)

    out = torch.zeros(1, dtype=torch.int32, device=device)
    kernel_nested[(1, )](out, shape=((8, 4), 2))
    assert out.item() == 842


@pytest.mark.interpreter
def test_constexpr_tuple_arg_subscript(device):
    """Subscripting a tl.constexpr Python-tuple argument must yield a
    constexpr value (not a stripped Python int), so it can be used where
    constexpr is required (e.g. tl.reshape shape, tl.full shape)."""

    @triton.jit
    def kernel(out_ptr, shape: tl.constexpr):
        v = shape[0] * 100 + shape[1] * 10 + shape[2]
        x = tl.full((1, ), v, dtype=tl.int32)
        # use subscript value as a constexpr shape element too
        x = tl.reshape(x, (shape[0] // shape[0], ))
        tl.store(out_ptr + tl.arange(0, 1), x)

    out = torch.zeros(1, dtype=torch.int32, device=device)
    kernel[(1, )](out, shape=(8, 4, 2))
    assert out.item() == 842


@triton.aggregate
class _Offset:
    """A host-constructible aggregate mixing a runtime and a compile-time member."""
    values: tl.tensor
    bump: tl.constexpr

    @triton.constexpr_function
    def __init__(self, values, bump):
        self.values = values
        self.bump = tl.constexpr(bump)

    @triton.jit
    def load(self, offs):
        return tl.load(self.values + offs) + self.bump


def test_aggregate_in_tuple_argument(device):
    """An aggregate may travel inside a tuple argument rather than as an argument of
    its own. Its runtime members are lifted to kernel parameters in place, so the
    tuple's other elements are unaffected and swapping a runtime member reuses the
    compiled kernel."""

    @triton.jit
    def kernel(out_ptr, packed, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        tl.store(out_ptr + offs, packed[0].load(offs) * packed[1])

    BLOCK = 16
    values = torch.arange(BLOCK, dtype=torch.float32, device=device)
    out = torch.zeros(BLOCK, dtype=torch.float32, device=device)

    kernel[(1, )](out, (_Offset(values, 1.0), 2.0), BLOCK)
    torch.testing.assert_close(out, (values + 1.0) * 2.0)

    # Only the aggregate's runtime member changes: no recompile.
    n_compiled = len(kernel.device_caches[out.device.index][0])
    other = torch.arange(BLOCK, dtype=torch.float32, device=device) * 3
    kernel[(1, )](out, (_Offset(other, 1.0), 2.0), BLOCK)
    torch.testing.assert_close(out, (other + 1.0) * 2.0)
    assert len(kernel.device_caches[out.device.index][0]) == n_compiled


def test_aggregate_in_namedtuple_argument(device):
    """Same, but the enclosing argument is a namedtuple: it keeps its own type, so
    the aggregate is still reachable by field name."""

    class Packed(NamedTuple):
        offset: object
        gain: object

    @triton.jit
    def kernel(out_ptr, packed, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        tl.store(out_ptr + offs, packed.offset.load(offs) * packed.gain)

    BLOCK = 16
    values = torch.arange(BLOCK, dtype=torch.float32, device=device)
    out = torch.zeros(BLOCK, dtype=torch.float32, device=device)

    kernel[(1, )](out, Packed(_Offset(values, 4.0), 0.5), BLOCK)
    torch.testing.assert_close(out, (values + 4.0) * 0.5)


def test_aggregate_in_constexpr_declared_argument_is_not_lifted(device):
    """A parameter declared `tl.constexpr` is baked whole. The binder gives it one
    flat compile-time slot without looking inside, so an aggregate passed that way
    (on its own or inside a tuple) is an ordinary constexpr object: nothing is
    lifted, and it specializes on object identity like any other constexpr."""
    from triton._C.libtriton import native_find_aggregate_arg_paths
    from triton.compiler import make_backend
    from triton.runtime.jit import create_function_from_signature

    BLOCK = 16
    values = torch.arange(BLOCK, dtype=torch.float32, device=device)

    @triton.jit
    def kernel(out_ptr, cfg: tl.constexpr, packed: tl.constexpr, BLOCK: tl.constexpr):
        pass

    backend = make_backend(triton.runtime.driver.active.get_current_target())
    binder = create_function_from_signature(kernel.signature, kernel.params, backend)
    _, specialization, _ = binder(values, _PureConfig(1.0, 4.0), (_Offset(values, 1.0), 2.0), BLOCK, debug=False,
                                  instrumentation_mode="", fpsan_homomorphic_casts=False)
    assert native_find_aggregate_arg_paths(specialization) is None
    assert specialization[1][0] == "constexpr" and specialization[2][0] == "constexpr"


@triton.aggregate
class _PureConfig:
    """An aggregate with no runtime members at all: purely a bundle of constants."""
    lo: tl.constexpr
    hi: tl.constexpr

    @triton.constexpr_function
    def __init__(self, lo, hi):
        self.lo = tl.constexpr(lo)
        self.hi = tl.constexpr(hi)

    @triton.jit
    def span(self):
        return self.hi - self.lo


def test_pure_constexpr_aggregate_keys_on_value(device):
    """An aggregate with nothing to lift is still keyed by its *values*, not by the
    identity of the instance -- otherwise a fresh (and equal) aggregate built per
    launch would miss the compilation cache every time."""

    @triton.jit
    def kernel(out_ptr, cfg, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        tl.store(out_ptr + offs, tl.full((BLOCK, ), cfg.span(), tl.float32))

    BLOCK = 16
    out = torch.zeros(BLOCK, dtype=torch.float32, device=device)

    kernel[(1, )](out, _PureConfig(1.0, 4.0), BLOCK)
    torch.testing.assert_close(out, torch.full_like(out, 3.0))
    n_compiled = len(kernel.device_caches[out.device.index][0])

    # A distinct object with equal members reuses the kernel.
    kernel[(1, )](out, _PureConfig(1.0, 4.0), BLOCK)
    assert len(kernel.device_caches[out.device.index][0]) == n_compiled

    # Different members specialize.
    kernel[(1, )](out, _PureConfig(1.0, 6.0), BLOCK)
    torch.testing.assert_close(out, torch.full_like(out, 5.0))
    assert len(kernel.device_caches[out.device.index][0]) == n_compiled + 1

    # So do members that are `==` in Python but bake to a different constant
    # type: `1 == 1.0`, yet an int member is not a float member.
    kernel[(1, )](out, _PureConfig(1, 4), BLOCK)
    torch.testing.assert_close(out, torch.full_like(out, 3.0))
    assert len(kernel.device_caches[out.device.index][0]) == n_compiled + 2


def test_aggregate_arg_wrappers_are_interned(device):
    """Equal-valued aggregates -- distinct instances, distinct runtime members --
    resolve to the *same* wrapper object at launch, so the in-memory kernel cache
    lookup compares by identity. The wrapper carries no runtime member; those come
    back separately, from the aggregate at hand."""
    from triton.runtime.jit import AggregateArg

    values = torch.arange(16, dtype=torch.float32, device=device)
    other = values * 3

    w1, rt1 = AggregateArg.for_value(_Offset(values, 1.0))
    w2, rt2 = AggregateArg.for_value(_Offset(other, 1.0))
    assert w1 is w2
    assert len(rt1) == 1 and rt1[0] is values
    assert len(rt2) == 1 and rt2[0] is other

    # A different baked member, or a different class, is a different wrapper.
    w3, _ = AggregateArg.for_value(_Offset(values, 2.0))
    assert w3 is not w1 and w3 != w1
    w4, rt4 = AggregateArg.for_value(_PureConfig(1.0, 4.0))
    assert w4 is not w1 and len(rt4) == 0

    # The lift is memoized on the (immutable) instance: a prebuilt aggregate
    # launched repeatedly is walked once, and the memo is per instance -- an
    # equal-valued sibling shares the wrapper but reports its own members.
    agg = _Offset(values, 1.0)
    first = AggregateArg.for_value(agg)
    assert AggregateArg.for_value(agg) is first
    assert first[0] is w1 and first[1][0] is values
    assert AggregateArg.for_value(_Offset(other, 1.0)) is not first
    # aggregate_replace() builds a fresh instance, so it does not inherit the memo.
    replaced = tl.aggregate_replace(agg, bump=2.0)
    assert AggregateArg.for_value(replaced)[0] is w3

    # The launch path hands the interned wrapper to the specialization, so a repeat
    # launch with a fresh equal aggregate hits the same kernel_key_cache entry.
    @triton.jit
    def kernel(out_ptr, off, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        tl.store(out_ptr + offs, off.load(offs))

    out = torch.zeros(16, dtype=torch.float32, device=device)
    kernel[(1, )](out, _Offset(values, 1.0), 16)
    kernel[(1, )](out, _Offset(other, 1.0), 16)
    torch.testing.assert_close(out, other + 1.0)
    kernel_cache, kernel_key_cache = kernel.device_caches[out.device.index][:2]
    assert len(kernel_cache) == 1 and len(kernel_key_cache) == 1
    (spec, _), = kernel_key_cache.keys()
    assert spec[1][1][0] is w1


def test_aggregate_arg_preload_round_trip(device, fresh_knobs):
    """The serialized specialization data of a kernel taking a host aggregate can be
    preloaded, and the preloaded kernel is the one a later launch uses."""

    @triton.jit
    def kernel(out_ptr, off, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        tl.store(out_ptr + offs, off.load(offs))

    specialization_data = None

    def cache_hook(*args, **kwargs):
        nonlocal specialization_data
        specialization_data = kwargs["compile"]["specialization_data"]

    fresh_knobs.runtime.jit_cache_hook = cache_hook

    BLOCK = 16
    values = torch.arange(BLOCK, dtype=torch.float32, device=device)
    out = torch.zeros(BLOCK, dtype=torch.float32, device=device)
    kernel[(1, )](out, _Offset(values, 3.0), BLOCK)
    torch.testing.assert_close(out, values + 3.0)
    assert specialization_data is not None
    fresh_knobs.runtime.jit_cache_hook = None

    kernel.device_caches.clear()
    assert kernel.preload(specialization_data) is not None
    kernel_cache = kernel.device_caches[out.device.index][0]
    assert len(kernel_cache) == 1

    out.zero_()
    other = values * 2
    kernel[(1, )](out, _Offset(other, 3.0), BLOCK)
    torch.testing.assert_close(out, other + 3.0)
    assert len(kernel_cache) == 1


def _make_scaling_epilogue(op):
    """A factory for a parameterized aggregate. Every class it returns has the same
    `__qualname__`, so the cache key cannot be the class name alone."""

    @triton.aggregate
    class _Epilogue:
        values: tl.tensor

        @triton.constexpr_function
        def __init__(self, values):
            self.values = values

        if op == "add":

            @triton.jit
            def apply(self, offs):
                return tl.load(self.values + offs) + 100.0
        else:

            @triton.jit
            def apply(self, offs):
                return tl.load(self.values + offs) * 100.0

    return _Epilogue


def test_aggregate_classes_sharing_a_qualname_do_not_share_a_kernel(device):
    """Two aggregate classes from the same factory differ only in the body of a
    `@triton.jit` method -- same module, same qualname, same field layout. Each must
    compile its own kernel."""
    added, multiplied = _make_scaling_epilogue("add"), _make_scaling_epilogue("mul")
    assert added.__qualname__ == multiplied.__qualname__

    @triton.jit
    def kernel(out_ptr, epilogue, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        tl.store(out_ptr + offs, epilogue.apply(offs))

    BLOCK = 16
    values = torch.full((BLOCK, ), 2.0, dtype=torch.float32, device=device)
    out = torch.zeros(BLOCK, dtype=torch.float32, device=device)

    kernel[(1, )](out, added(values), BLOCK)
    torch.testing.assert_close(out, values + 100.0)

    kernel[(1, )](out, multiplied(values), BLOCK)
    torch.testing.assert_close(out, values * 100.0)


@triton.aggregate
class _HostBias:
    """An aggregate with a plain Python (host-only) constructor."""
    values: tl.tensor
    n: tl.constexpr

    def __init__(self, values, n):
        self.values = values
        self.n = tl.constexpr(n)

    @triton.jit
    def apply(self, offs):
        return tl.load(self.values + offs, mask=offs < self.n, other=0.0)


@triton.aggregate
class _HostBiasScale(_HostBias):
    scale: tl.tensor

    def __init__(self, values, n, scale):
        self.values = values
        self.n = tl.constexpr(n)
        self.scale = scale

    @triton.jit
    def apply(self, offs):
        return _HostBias.apply(self, offs) * self.scale


def test_aggregate_derived_with_host_init_references_base(device):
    """A `@triton.jit` method that references an aggregate class whose `__init__`
    is a plain Python function hashes that class without compiling its `__init__`."""

    @triton.jit
    def kernel(out_ptr, epilogue, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        tl.store(out_ptr + offs, epilogue.apply(offs))

    BLOCK = 16
    values = torch.arange(BLOCK, dtype=torch.float32, device=device)
    out = torch.zeros(BLOCK, dtype=torch.float32, device=device)
    kernel[(1, )](out, _HostBiasScale(values, 12, 0.5), BLOCK)
    expected = torch.where(torch.arange(BLOCK, device=device) < 12, values, 0.0) * 0.5
    torch.testing.assert_close(out, expected)


@triton.aggregate
class _HostDesc:
    """An aggregate with a plain Python constructor and a descriptor member."""
    desc: tl.tensor_descriptor

    def __init__(self, desc):
        self.desc = desc


def test_aggregate_host_field_rejects_wrong_value_type():
    """A runtime member of a host-constructed aggregate accepts only values that can
    be lifted to a kernel parameter of the annotated type."""
    with pytest.raises(TypeError, match="attribute 'values'"):
        _HostBias("not a tensor", 4)
    with pytest.raises(TypeError, match="attribute 'desc'"):
        _HostDesc(torch.zeros(4))
