import pytest
import torch
import triton.language as tl
import triton
from triton._internal_testing import run_in_process


def _run_device_assert(cond, mask, opt_flag, jit_flag, device):
    triton.knobs.refresh_knobs()
    torch.zeros([1], dtype=torch.int32, device=device)

    @triton.jit(debug=jit_flag)
    def _kernel(COND: tl.constexpr, MASK: tl.constexpr):
        tl.device_assert(COND, 'test', mask=MASK)

    kwargs = {}
    if opt_flag is not None:
        kwargs["debug"] = opt_flag

    _kernel[(1, )](cond, mask, **kwargs)
    getattr(torch, device).synchronize()


def _run_device_assert_barrier(device):
    triton.knobs.refresh_knobs()
    tensor = torch.zeros([16], dtype=torch.int32, device=device)

    @triton.jit
    def _kernel(in_ptr0):
        xindex = tl.arange(0, 8)
        tmp0 = tl.load(in_ptr0 + xindex)
        tl.device_assert(tmp0 < 1)

    _kernel[(1, )](tensor)
    getattr(torch, device).synchronize()


def _run_expect_zero_device_assert(device):
    triton.knobs.refresh_knobs()
    x = torch.ones([16], dtype=torch.float32, device=device)
    out = torch.empty_like(x)

    @triton.jit
    def _kernel(x_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
        offsets = tl.arange(0, BLOCK_SIZE)
        y = tl.load(x_ptr + offsets)
        y = tl.expect_zero(y, offsets == 0)
        tl.store(out_ptr + offsets, y)

    _kernel[(1, )](x, out, BLOCK_SIZE=16)
    getattr(torch, device).synchronize()


def test_expect_zero_device_assert(device):
    result = run_in_process(_run_expect_zero_device_assert, (device, ), env={"TRITON_DEBUG": "1"})
    assert isinstance(result.exc, RuntimeError)


@pytest.mark.parametrize('cond', [True, False])
@pytest.mark.parametrize('mask', [True, False, None])
@pytest.mark.parametrize('opt_flag', [True, False, None])
@pytest.mark.parametrize('env_var', [True, False])
@pytest.mark.parametrize('jit_flag', [True, False])
def test_device_assert(cond, mask, opt_flag, env_var, jit_flag, device):
    is_debug = env_var or (opt_flag if opt_flag is not None else jit_flag)
    result = run_in_process(_run_device_assert, (cond, mask, opt_flag, jit_flag, device),
                            env={"TRITON_DEBUG": str(int(env_var))})

    if not cond and is_debug and mask is not False:
        assert isinstance(result.exc, RuntimeError)
        return

    assert result.exc is None, result.exc


def test_device_assert_barrier(device):
    result = run_in_process(_run_device_assert_barrier, (device, ), env={"TRITON_DEBUG": "1"})
    assert result.exc is None, result.exc


@pytest.mark.parametrize("cond", [False, True])
def test_static_assert(cond):

    @triton.jit
    def _kernel(COND: tl.constexpr):
        tl.static_assert(COND)

    if not cond:
        with pytest.raises(triton.compiler.errors.CompileTimeAssertionFailure):
            _kernel[(1, )](cond)
        return

    _kernel[(1, )](cond)


def _run_overflow(x, y, x_dtype, y_dtype, debug, op, device):
    if op == "add":

        @triton.jit
        def tri_func(X, Y, Z):
            tl.store(Z, tl.load(X) + tl.load(Y))

        ref_func = lambda lhs, rhs: lhs + rhs
    elif op == "mul":

        @triton.jit
        def tri_func(X, Y, Z):
            tl.store(Z, tl.load(X) * tl.load(Y))

        ref_func = lambda lhs, rhs: lhs * rhs
    else:
        assert op == "sub"

        @triton.jit
        def tri_func(X, Y, Z):
            tl.store(Z, tl.load(X) - tl.load(Y))

        ref_func = lambda lhs, rhs: lhs - rhs

    x = torch.tensor([x], dtype=getattr(torch, x_dtype), device=device)
    y = torch.tensor([y], dtype=getattr(torch, y_dtype), device=device)
    z = torch.empty_like(x)
    tri_func[(1, )](x, y, z, debug=debug)
    getattr(torch, device).synchronize()
    assert int(z) == int(ref_func(x, y))


def _assert_overflow_result(result, debug, should_overflow):
    if should_overflow and debug:
        assert isinstance(result.exc, RuntimeError)
        assert "device-side assert" in str(result.exc)
        return

    assert result.exc is None, result.exc


# integer overflow sanitization


@pytest.mark.parametrize("x, y, x_dtype, y_dtype, debug, should_overflow", [
    (-2**31, -1, 'int32', 'int32', False, False),
    (-2**31, -1, 'int32', 'int32', True, True),
    (2**31 - 1, 1, 'int32', 'int32', True, True),
    (2**31 - 1, 100, 'int32', 'int32', True, True),
    (-2**31, 0, 'int32', 'int32', True, False),
    (-2**31, 2, 'int32', 'int32', True, False),
    (0, -1, 'int32', 'int32', True, False),
    (-2**15, -1, 'int16', 'int16', True, True),
    (2**15 - 1, 1, 'int16', 'int16', True, True),
])
def test_sanitize_int_add_overflow(x, y, x_dtype, y_dtype, debug, should_overflow, device):
    result = run_in_process(_run_overflow, (x, y, x_dtype, y_dtype, debug, "add", device))
    _assert_overflow_result(result, debug, should_overflow)


# mul overflow


@pytest.mark.parametrize("x, y, x_dtype, y_dtype, debug, should_overflow", [
    (2**30, 4, 'int32', 'int32', False, False),
    (2**30, 4, 'int32', 'int32', True, True),
    (2**30, 2, 'int32', 'int32', True, True),
    (-2**30, -4, 'int32', 'int32', True, True),
    (-2**31, 1, 'int32', 'int32', True, False),
    (-2**30, 2, 'int32', 'int32', True, False),
])
def test_sanitize_int_mul_overflow(x, y, x_dtype, y_dtype, debug, should_overflow, device):
    result = run_in_process(_run_overflow, (x, y, x_dtype, y_dtype, debug, "mul", device))
    _assert_overflow_result(result, debug, should_overflow)


# sub overflow


@pytest.mark.parametrize("x, y, x_dtype, y_dtype, debug, should_overflow", [
    (-2**31, 1, 'int32', 'int32', False, False),
    (-2**31, 1, 'int32', 'int32', True, True),
    (2**31 - 1, -1, 'int32', 'int32', True, True),
    (2**31 - 1, 1, 'int32', 'int32', True, False),
    (-2**31, -1, 'int32', 'int32', True, False),
])
def test_sanitize_int_sub_overflow(x, y, x_dtype, y_dtype, debug, should_overflow, device):
    result = run_in_process(_run_overflow, (x, y, x_dtype, y_dtype, debug, "sub", device))
    _assert_overflow_result(result, debug, should_overflow)
