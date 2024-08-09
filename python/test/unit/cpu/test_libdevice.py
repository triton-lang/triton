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


float_dtypes = ['bfloat16', 'float16', 'float32', 'float64']


@pytest.mark.parametrize("dtype_str", float_dtypes)
@pytest.mark.parametrize("math_fn", [
    "acos", "acosh", "asin", "asinh", "atan", "atanh", "cbrt", "cos", "cosh", "erf", "exp", "exp2", "expm1", "floor",
    "log", "log1p", "log2", "log10", "rsqrt", "sin", "sinh", "sqrt", "tan", "tanh"
])
@pytest.mark.parametrize("size", [1, 4, 16, 64])
def test_libdevice(dtype_str, math_fn, size, device):
    if not is_cpu():
        pytest.skip("This test is CPU-specific")

    if dtype_str == "bfloat16":
        if math_fn == "floor" or math_fn == "rsqrt":
            pytest.skip("libgcc < 13 does not define __truncsfbf2, which this op needs")

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
    kernel[(1, )](src, res, MATH_FN=math_fn, BLOCK_SIZE=size)
    if math_fn == "cbrt":
        ref = src.pow(1 / 3)
    else:
        ref = getattr(src, math_fn)()
    torch.testing.assert_close(ref, res)
