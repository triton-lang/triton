import os
import pytest
import torch
import triton.language as tl
import triton

@pytest.mark.parametrize('cond, opt_flag, env_var', [
    (cond, opt_flag, env_var) for cond in [True, False] \
                              for opt_flag in [True, False] \
                              for env_var in [True, False]\
])
@pytest.mark.forked
def test_device_assert(cond, opt_flag, env_var, device="cuda"):
    os.environ['TRITON_DEBUG'] = str(int(env_var))
    torch.zeros([1], dtype=torch.int32, device=device)

    @triton.jit
    def _kernel(COND: tl.constexpr):
        tl.device_assert(COND, 'test')

    if not cond and (opt_flag or env_var):
        with pytest.raises(RuntimeError):
            _kernel[(1, )](cond, debug=opt_flag)
            torch.cuda.synchronize()
        return

    _kernel[(1, )](cond, debug=opt_flag)
    torch.cuda.synchronize()


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


def _test_overflow(x, y, x_dtype, y_dtype, debug, should_overflow, tri_func, ref_func):
    device = "cuda"
    x = torch.tensor([x], dtype=getattr(torch, x_dtype), device=device)
    y = torch.tensor([y], dtype=getattr(torch, y_dtype), device=device)
    z = torch.empty_like(x)
    if should_overflow and debug:
        with pytest.raises(RuntimeError) as exc_info:
            tri_func[(1, )](x, y, z, debug=debug)
            torch.cuda.synchronize()
        assert "device-side assert" in str(exc_info.value)
    else:
        tri_func[(1, )](x, y, z, debug=debug)
        torch.cuda.synchronize()
        assert int(z) == int(ref_func(x, y))


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
@pytest.mark.forked
def test_sanitize_int_add_overflow(x, y, x_dtype, y_dtype, debug, should_overflow):

    @triton.jit
    def _kernel_add(X, Y, Z):
        tl.store(Z, tl.load(X) + tl.load(Y))

    _test_overflow(x, y, x_dtype, y_dtype, debug, should_overflow, _kernel_add, lambda x, y: x + y)


# mul overflow


@pytest.mark.parametrize("x, y, x_dtype, y_dtype, debug, should_overflow", [
    (2**30, 4, 'int32', 'int32', False, False),
    (2**30, 4, 'int32', 'int32', True, True),
    (2**30, 2, 'int32', 'int32', True, True),
    (-2**30, -4, 'int32', 'int32', True, True),
    (-2**31, 1, 'int32', 'int32', True, False),
    (-2**30, 2, 'int32', 'int32', True, False),
])
@pytest.mark.forked
def test_sanitize_int_mul_overflow(x, y, x_dtype, y_dtype, debug, should_overflow):

    @triton.jit
    def _kernel_mul(X, Y, Z):
        tl.store(Z, tl.load(X) * tl.load(Y))

    _test_overflow(x, y, x_dtype, y_dtype, debug, should_overflow, _kernel_mul, lambda x, y: x * y)


# sub overflow


@pytest.mark.parametrize("x, y, x_dtype, y_dtype, debug, should_overflow", [
    (-2**31, 1, 'int32', 'int32', False, False),
    (-2**31, 1, 'int32', 'int32', True, True),
    (2**31 - 1, -1, 'int32', 'int32', True, True),
    (2**31 - 1, 1, 'int32', 'int32', True, False),
    (-2**31, -1, 'int32', 'int32', True, False),
])
@pytest.mark.forked
def test_sanitize_int_sub_overflow(x, y, x_dtype, y_dtype, debug, should_overflow):

    @triton.jit
    def _kernel_sub(X, Y, Z):
        tl.store(Z, tl.load(X) - tl.load(Y))

    _test_overflow(x, y, x_dtype, y_dtype, should_overflow, debug, _kernel_sub, lambda x, y: x - y)
