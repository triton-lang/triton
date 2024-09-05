import os
import pytest
import torch

import triton
import triton.language as tl
from triton.language.extra import libdevice
from itertools import chain, product


def is_interpreter():
    return os.environ.get('TRITON_INTERPRET', '0') == '1'


def is_cpu():
    return not is_interpreter() and \
        triton.runtime.driver.active.get_current_target().backend == "cpu"


float_dtypes = ['bfloat16', 'float16', 'float32', 'float64']
lib_prefix = {
    "libsleef": "Sleef",
    "libmvec": "_ZGV",
}
arch = triton.runtime.driver.active.get_current_target().arch

vec_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
scalar_sizes = [1, 4, 16, 64]


def check_num_vec_calls(meta, vec_lib, dtype_str, size):
    # Check generated code calls vector math function
    # FP16 and BF16 are casted to FP32 for math ops
    elem_size = 8 if dtype_str == "float64" else 4
    data_size = size * elem_size
    if data_size > 64:
        num_vec_calls = data_size // 64
    elif data_size >= 16:
        num_vec_calls = 1
    else:
        num_vec_calls = 0
    assert meta.asm["asm"].count(lib_prefix[vec_lib]) == num_vec_calls


@pytest.mark.parametrize("vec_lib, size",
                         chain(product(["libsleef", "libmvec"], vec_sizes), product([None], scalar_sizes)))
@pytest.mark.parametrize("dtype_str", float_dtypes)
@pytest.mark.parametrize("math_fn", ["cos", "exp", "exp2", "log", "log2", "sin"])
def test_tensor_math_fn(vec_lib, dtype_str, math_fn, size, device):
    if not is_cpu():
        pytest.skip("This test is CPU-specific")
    if vec_lib == "libmvec" and arch != "x86_64":
        pytest.skip("Vectorized libm calls are supported for x86 target only.")

    @triton.jit
    def kernel(src, dst, MATH_FN: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        idxs = tl.arange(0, BLOCK_SIZE)
        x = tl.load(src + idxs)
        y = getattr(x, MATH_FN)()
        tl.store(dst + idxs, y)

    src = torch.rand((size, ), dtype=getattr(torch, dtype_str), device=device)
    res = torch.empty(src.shape, dtype=getattr(torch, dtype_str), device=device)
    meta = kernel[(1, )](src, res, MATH_FN=math_fn, BLOCK_SIZE=size, vec_lib=vec_lib)
    ref = getattr(src, math_fn)()
    torch.testing.assert_close(ref, res)

    if vec_lib is not None:
        check_num_vec_calls(meta, vec_lib, dtype_str, size)


@pytest.mark.parametrize("vec_lib, size",
                         chain(product(["libsleef", "libmvec"], vec_sizes), product([None], scalar_sizes)))
@pytest.mark.parametrize("dtype_str", float_dtypes)
@pytest.mark.parametrize("math_fn", [
    "acos", "acosh", "asin", "asinh", "atan", "atanh", "cbrt", "cos", "cosh", "erf", "exp", "exp2", "expm1", "floor",
    "isnan", "isinf", "log", "log1p", "log2", "log10", "rsqrt", "signbit", "sin", "sinh", "sqrt", "tan", "tanh", "trunc"
])
def test_libdevice_math_fn(vec_lib, dtype_str, math_fn, size, device):
    if not is_cpu():
        pytest.skip("This test is CPU-specific")
    if vec_lib == "libmvec" and arch != "x86_64":
        pytest.skip("Vectorized libm calls are supported for x86 target only.")

    @triton.jit
    def kernel(src, dst, MATH_FN: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        idxs = tl.arange(0, BLOCK_SIZE)
        x = tl.load(src + idxs)
        y = getattr(libdevice, MATH_FN)(x)
        tl.store(dst + idxs, y)

    src = torch.rand((size, ), dtype=getattr(torch, dtype_str), device=device)
    # Customize inputs
    if math_fn == "acosh":
        src = src.abs() + 1
    if math_fn == "isnan" or math_fn == "isinf":
        indices = torch.randint(low=0, high=size, size=(size // 2, ), device=device)
        for i in indices:
            if math_fn == "isnan":
                src[i] = float("nan")
            else:
                src[i] = float(("+" if i % 2 else "-") + "inf")

    # Generate reference output
    if math_fn == "cbrt":
        ref = src.pow(1 / 3)
    else:
        ref = getattr(src, math_fn)()

    res = torch.empty(src.shape, dtype=ref.dtype, device=device)
    meta = kernel[(1, )](src, res, MATH_FN=math_fn, BLOCK_SIZE=size, vec_lib=vec_lib)
    torch.testing.assert_close(ref, res)

    if vec_lib is None:
        return

    # These are not implemented via extern library calls
    native_impls = {
        "libmvec": {"expm1", "floor", "isnan", "isinf", "rsqrt", "signbit", "sqrt", "trunc"},
        "libsleef": {"isnan", "isinf", "rsqrt", "signbit"},
    }
    if math_fn not in native_impls[vec_lib]:
        check_num_vec_calls(meta, vec_lib, dtype_str, size)
    else:
        assert meta.asm["asm"].count(lib_prefix[vec_lib]) == 0
