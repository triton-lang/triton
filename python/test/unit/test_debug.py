import triton
import triton.language as tl
import os
import torch
import pytest


def test_fn_dump(capfd, device):
    N = 1024
    src = torch.zeros(N, device=device)

    grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE']), )

    @triton.jit
    def _kernel(src, N, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(src + offsets, mask=offsets < N) + 1
        tl.store(src + offsets, x, mask=offsets < N)

    os.environ['MLIR_ENABLE_DUMP'] = '1'
    BLOCK_SIZE = 16
    _kernel[grid](src, N, BLOCK_SIZE)
    captured = capfd.readouterr()
    assert "IR Dump Before" in captured.err
    assert "tt.func public @_kernel" in captured.err

    os.environ['MLIR_ENABLE_DUMP'] = '_kernel'
    BLOCK_SIZE = 32
    _kernel[grid](src, N, BLOCK_SIZE)
    captured = capfd.readouterr()
    assert "IR Dump Before" in captured.err
    assert "tt.func public @_kernel" in captured.err

    os.environ['MLIR_ENABLE_DUMP'] = '_kernel2'
    BLOCK_SIZE = 64
    _kernel[grid](src, N, BLOCK_SIZE)
    captured = capfd.readouterr()
    assert "IR Dump Before" not in captured.err

    os.environ['MLIR_ENABLE_DUMP'] = '0'


#


def _test_overflow(x, y, x_dtype, y_dtype, should_overflow, tri_func, ref_func):
    device = "cuda"
    x = torch.tensor([x], dtype=getattr(torch, x_dtype), device=device)
    y = torch.tensor([y], dtype=getattr(torch, y_dtype), device=device)
    z = torch.empty_like(x)
    if should_overflow:
        with pytest.raises(RuntimeError) as exc_info:
            tri_func[(1, )](x, y, z)
            torch.cuda.synchronize()
        assert "device-side assert" in str(exc_info.value)
    else:
        tri_func[(1, )](x, y, z)
        assert int(z) == int(ref_func(x, y))


# add overflow


@pytest.mark.parametrize("x, y, x_dtype, y_dtype, should_overflow", [
    (-2**31, -1, 'int32', 'int32', True),
    (2**31 - 1, 1, 'int32', 'int32', True),
    (2**31 - 1, 100, 'int32', 'int32', True),
    (-2**31, 0, 'int32', 'int32', False),
    (-2**31, 2, 'int32', 'int32', False),
    (0, -1, 'int32', 'int32', False),
    (-2**15, -1, 'int16', 'int16', True),
    (2**15 - 1, 1, 'int16', 'int16', True),
])
def test_sanitize_int_add_overflow(x, y, x_dtype, y_dtype, should_overflow):

    @triton.jit
    def _kernel_add(X, Y, Z):
        tl.store(Z, tl.load(X) + tl.load(Y))

    _test_overflow(x, y, x_dtype, y_dtype, should_overflow, _kernel_add, lambda x, y: x + y)


# mul overflow


@pytest.mark.parametrize("x, y, x_dtype, y_dtype, should_overflow", [
    (2**30, 4, 'int32', 'int32', True),
    (2**30, 2, 'int32', 'int32', True),
    (-2**30, -4, 'int32', 'int32', True),
    (-2**31, 1, 'int32', 'int32', False),
    (-2**30, 2, 'int32', 'int32', False),
])
def test_sanitize_int_mul_overflow(x, y, x_dtype, y_dtype, should_overflow):

    @triton.jit
    def _kernel_mul(X, Y, Z):
        tl.store(Z, tl.load(X) * tl.load(Y))

    _test_overflow(x, y, x_dtype, y_dtype, should_overflow, _kernel_mul, lambda x, y: x * y)


# sub overflow


@pytest.mark.parametrize("x, y, x_dtype, y_dtype, should_overflow", [
    (-2**31, 1, 'int32', 'int32', True),
    (2**31 - 1, -1, 'int32', 'int32', True),
    (2**31 - 1, 1, 'int32', 'int32', False),
    (-2**31, -1, 'int32', 'int32', False),
])
def test_sanitize_int_sub_overflow(x, y, x_dtype, y_dtype, should_overflow):

    @triton.jit
    def _kernel_sub(X, Y, Z):
        tl.store(Z, tl.load(X) - tl.load(Y))

    _test_overflow(x, y, x_dtype, y_dtype, should_overflow, _kernel_sub, lambda x, y: x - y)
