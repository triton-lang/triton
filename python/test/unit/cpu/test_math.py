import inspect
import os
import pytest
import torch

import triton
import triton.language as tl
from triton._C.libtriton import llvm
from triton.language.extra import libdevice
from itertools import chain, product


def get_native_vector_size_in_bits():
    """
    Returns the native vector size of the CPU.
    Assuming x86 always uses "auto dispatch" with 512-bit vectors for Sleef.
    """
    cpu_features = llvm.get_cpu_features()
    # TODO support for arm sve w/ VLA
    if "neon" in cpu_features:
        return 128
    return 512


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


def check_num_vec_calls(meta, vec_lib, dtype_str, size, is_always_extern=False):
    # Check generated code calls vector math function
    # FP16 and BF16 are cast to FP32 for math ops
    elem_size = 8 if dtype_str == "float64" else 4
    data_size = size * elem_size

    vec_size = get_native_vector_size_in_bits() / 8  # bytes
    # 128-bit vector is the smallest supported by Sleef for both x86 and arm
    smallest_vec_size = 128 / 8  # bytes
    if data_size > vec_size:
        num_vec_calls = data_size // vec_size
    elif data_size >= smallest_vec_size:
        num_vec_calls = 1
    else:
        num_vec_calls = 1 if is_always_extern else 0
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
    "acos", "acosh", "asin", "asinh", "atan", "atanh", "cbrt", "ceil", "cos", "cosh", "erf", "exp", "exp2", "expm1",
    "floor", "fmod", "isnan", "isinf", "log", "log1p", "log2", "log10", "pow", "rsqrt", "signbit", "sin", "sinh",
    "sqrt", "tan", "tanh", "trunc"
])
def test_libdevice_math_fn(vec_lib, dtype_str, math_fn, size, device):
    if not is_cpu():
        pytest.skip("This test is CPU-specific")
    if vec_lib == "libmvec" and arch != "x86_64":
        pytest.skip("Vectorized libm calls are supported for x86 target only.")
    if math_fn in {"ceil", "fmod", "pow"}:
        if vec_lib != "libsleef":
            pytest.skip("extern_elementwise only supports libsleef")
        if dtype_str not in {"float32", "torch.float64"}:
            pytest.skip(f"{math_fn} only supports fp32, fp64")

    @triton.jit
    def unary_kernel(src, dst, MATH_FN: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        idxs = tl.arange(0, BLOCK_SIZE)
        x = tl.load(src + idxs)
        y = getattr(libdevice, MATH_FN)(x)
        tl.store(dst + idxs, y)

    @triton.jit
    def binary_kernel(x_ptr, y_ptr, out_ptr, MATH_FN: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        idxs = tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + idxs)
        y = tl.load(y_ptr + idxs)
        result = getattr(libdevice, MATH_FN)(x, y)
        tl.store(out_ptr + idxs, result)

    signature = inspect.signature(getattr(libdevice, math_fn))
    num_params = len(signature.parameters)
    inputs = [torch.rand((size, ), dtype=getattr(torch, dtype_str), device=device) for _ in range(num_params)]
    # Customize inputs
    if math_fn == "acosh":
        inputs[0] = inputs[0].abs() + 1
    if math_fn == "isnan" or math_fn == "isinf":
        indices = torch.randint(low=0, high=size, size=(size // 2, ), device=device)
        src = inputs[0]
        for i in indices:
            if math_fn == "isnan":
                src[i] = float("nan")
            else:
                src[i] = float(("+" if i % 2 else "-") + "inf")

    # Generate reference output
    if math_fn == "cbrt":
        ref = inputs[0].pow(1 / 3)
    else:
        ref = getattr(inputs[0], math_fn)(*inputs[1:])

    res = torch.empty(inputs[0].shape, dtype=ref.dtype, device=device)
    kernel = unary_kernel if num_params == 1 else binary_kernel
    meta = kernel[(1, )](*inputs, res, MATH_FN=math_fn, BLOCK_SIZE=size, vec_lib=vec_lib)
    torch.testing.assert_close(ref, res)

    if vec_lib is None:
        return

    # These are not implemented via extern library calls
    native_impls = {
        "libmvec": {"expm1", "floor", "isnan", "isinf", "rsqrt", "signbit", "sqrt", "trunc"},
        "libsleef": {"isnan", "isinf", "rsqrt", "signbit"},
    }
    # These are always implemented with extern library calls
    always_extern = {"ceil", "fmod", "pow"}
    if math_fn not in native_impls[vec_lib]:
        check_num_vec_calls(meta, vec_lib, dtype_str, size, is_always_extern=math_fn in always_extern)
    else:
        assert meta.asm["asm"].count(lib_prefix[vec_lib]) == 0
