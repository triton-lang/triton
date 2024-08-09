import os
import pytest
import torch

import triton
import triton.language as tl
from triton.language.extra import libdevice


def is_interpreter():
    return os.environ.get('TRITON_INTERPRET', '0') == '1'


def is_cpu():
    return not is_interpreter() and \
        triton.runtime.driver.active.get_current_target().backend == "cpu"


def is_x86():
    return is_cpu() and \
        triton.runtime.driver.active.get_current_target().arch == "x86_64"


float_dtypes = ['bfloat16', 'float16', 'float32', 'float64']


@pytest.mark.parametrize("dtype_str", float_dtypes)
@pytest.mark.parametrize("math_fn", ["cos", "exp", "exp2", "log", "log2", "sin"])
@pytest.mark.parametrize("size", [1, 2, 4, 8, 16, 32, 64, 128])
def test_tensor_math_fn(dtype_str, math_fn, size, device):
    if not is_x86():
        pytest.skip("Vectorized libm calls are supported for x86 target only.")

    @triton.jit
    def kernel(src, dst, MATH_FN: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        idxs = tl.arange(0, BLOCK_SIZE)
        x = tl.load(src + idxs)
        y = getattr(x, MATH_FN)()
        tl.store(dst + idxs, y)

    src = torch.rand((size, ), dtype=getattr(torch, dtype_str), device=device)
    res = torch.empty(src.shape, dtype=getattr(torch, dtype_str), device=device)
    meta = kernel[(1, )](src, res, MATH_FN=math_fn, BLOCK_SIZE=size)
    ref = getattr(src, math_fn)()
    torch.testing.assert_close(ref, res)

    # Check generated code calls vector math function
    # FP16 and BF16 are casted to FP32 for math ops
    elem_size = 8 if dtype_str == "float64" else 4
    data_size = size * elem_size
    num_vec_calls = 0
    if data_size >= 16:
        num_vec_calls = 1
    if data_size > 64:
        num_vec_calls = data_size / 64
    prefix = "Sleef" if os.environ.get("TRITON_CPU_USE_SLEEF", "0") != "0" else "_ZGV"
    assert meta.asm["asm"].count(prefix) == num_vec_calls


@pytest.mark.parametrize("dtype_str", float_dtypes)
@pytest.mark.parametrize("math_fn", [
    "acos", "acosh", "asin", "asinh", "atan", "atanh", "cbrt", "cos", "cosh", "erf", "exp", "exp2", "log", "log2",
    "log10", "sin", "sinh", "tan", "tanh"
])
@pytest.mark.parametrize("size", [1, 2, 4, 8, 16, 32, 64, 128])
def test_libdevice_math_fn(dtype_str, math_fn, size, device):
    if not is_x86():
        pytest.skip("Vectorized libm calls are supported for x86 target only.")

    @triton.jit
    def kernel(src, dst, MATH_FN: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        idxs = tl.arange(0, BLOCK_SIZE)
        x = tl.load(src + idxs)
        y = getattr(libdevice, MATH_FN)(x)
        tl.store(dst + idxs, y)

    src = torch.rand((size, ), dtype=getattr(torch, dtype_str), device=device)
    if math_fn == "acosh":
        src = src.abs() + 1
    res = torch.empty(src.shape, dtype=getattr(torch, dtype_str), device=device)
    meta = kernel[(1, )](src, res, MATH_FN=math_fn, BLOCK_SIZE=size)
    if math_fn == "cbrt":
        ref = src.pow(1 / 3)
    else:
        ref = getattr(src, math_fn)()
    torch.testing.assert_close(ref, res)

    # Check generated code calls vector math function
    # FP16 and BF16 are casted to FP32 for math ops
    elem_size = 8 if dtype_str == "float64" else 4
    data_size = size * elem_size
    num_vec_calls = 0
    if data_size >= 16:
        num_vec_calls = 1
    if data_size > 64:
        num_vec_calls = data_size / 64
    prefix = "Sleef" if os.environ.get("TRITON_CPU_USE_SLEEF", "0") != "0" else "_ZGV"
    assert meta.asm["asm"].count(prefix) == num_vec_calls
