import inspect
import os
import re
import pytest
import torch

import triton
import triton.language as tl
from triton._C.libtriton import cpu
from triton.language.extra import libdevice
from itertools import chain, product


def get_native_vector_size_in_bits():
    """
    Returns the fixed vector size used by external SLEEF calls and fallbacks.
    Assuming x86 always uses "auto dispatch" with 512-bit vectors for Sleef.
    """
    cpu_features = cpu.llvm.get_cpu_features()
    if "neon" in cpu_features:
        return 128
    return 512


def has_sve():
    return bool({"sve", "sve2"} & cpu.llvm.get_cpu_features())


def is_interpreter():
    return os.environ.get('TRITON_INTERPRET', '0') == '1'


def is_cpu():
    return not is_interpreter() and \
        triton.runtime.driver.active.get_current_target().backend == "cpu"


float_dtypes = ['float32', 'float64']
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
    if vec_lib == "libsleef" and has_sve() and not is_always_extern and data_size >= smallest_vec_size:
        # The scalable loop has one static call site, regardless of how many
        # hardware vectors it processes at runtime.
        num_vec_calls = 1
    elif data_size > vec_size:
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


@pytest.mark.parametrize("dtype_str", ["float16", "bfloat16", "float32", "float64"])
@pytest.mark.parametrize("size", [4, 16, 64, 256])
@pytest.mark.parametrize("math_fn, ulp_suffix", [("sin", "_u10"), ("sqrt", "_u05"), ("floor", "_")])
def test_sleef_sve_vscale_reshaped_math(dtype_str, size, math_fn, ulp_suffix, device):
    if not is_cpu() or not has_sve():
        pytest.skip("This test requires an SVE CPU")

    @triton.jit
    def kernel(src, dst, MATH_FN: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        idxs = tl.arange(0, BLOCK_SIZE)
        x = tl.reshape(tl.load(src + idxs), (2, BLOCK_SIZE // 2))
        y = getattr(libdevice, MATH_FN)(x)
        tl.store(dst + idxs, tl.reshape(y, (BLOCK_SIZE, )))

    # Distinct values across the entire block catch incorrect loop strides or
    # shape restoration, including a final partial hardware vector.
    src = torch.linspace(0.125, 3.125, size, device=device).to(getattr(torch, dtype_str))
    res = torch.empty_like(src)
    meta = kernel[(1, )](src, res, MATH_FN=math_fn, BLOCK_SIZE=size, vec_lib="libsleef")
    ref = getattr(src, math_fn)()
    torch.testing.assert_close(res, ref)

    # Half and bfloat16 math is promoted before calling the SVE library.
    is_double = dtype_str == "float64"
    lanes, llvm_dtype, precision = (2, "double", "dx") if is_double else (4, "float", "fx")
    llir = meta.asm["llir"]
    # With vscale_range's lower bound, LLVM can fold a minimum-sized block to
    # one scalable call and eliminate the runtime loop step.
    if size > lanes:
        assert "llvm.vscale" in llir
    assert f"<vscale x {lanes} x {llvm_dtype}>" in llir
    assert f"Sleef_{math_fn}{precision}{ulp_suffix}sve" in llir
    check_num_vec_calls(meta, "libsleef", dtype_str, size)


@pytest.mark.parametrize("dtype_str, size", [("float32", 1), ("float32", 2), ("float64", 1)])
def test_sleef_sve_small_vector_fallback(dtype_str, size, device):
    if not is_cpu() or not has_sve():
        pytest.skip("This test requires an SVE CPU")

    @triton.jit
    def kernel(src, dst, BLOCK_SIZE: tl.constexpr):
        idxs = tl.arange(0, BLOCK_SIZE)
        tl.store(dst + idxs, tl.sin(tl.load(src + idxs)))

    src = torch.linspace(0.125, 1.125, size, device=device).to(getattr(torch, dtype_str))
    res = torch.empty_like(src)
    meta = kernel[(1, )](src, res, BLOCK_SIZE=size, vec_lib="libsleef")
    torch.testing.assert_close(res, src.sin())
    assert "llvm.vscale" not in meta.asm["llir"]
    assert "Sleef_" not in meta.asm["llir"]


def test_sleef_sve_chained_math_stack_usage(device):
    if not is_cpu() or not has_sve():
        pytest.skip("This test requires an SVE CPU")

    @triton.jit
    def kernel(src, dst, BLOCK_SIZE: tl.constexpr, DEPTH: tl.constexpr):
        offsets = tl.arange(0, BLOCK_SIZE)
        first = tl.sin(tl.load(src + offsets))
        result = first
        for _ in tl.static_range(DEPTH - 1):
            result = tl.sin(result)
        # Keep the first result live across the remaining math operations.
        tl.store(dst + offsets, result + first)

    size = 256
    src = torch.linspace(0.125, 3.125, size, dtype=torch.float32, device=device)
    res = torch.empty_like(src)
    frames = []
    for depth in (1, 8):
        meta = kernel[(1, )](src, res, BLOCK_SIZE=size, DEPTH=depth, vec_lib="libsleef")
        first = src.sin()
        ref = first
        for _ in range(depth - 1):
            ref = ref.sin()
        torch.testing.assert_close(res, ref + first)

        llir = meta.asm["llir"]
        if depth > 1:
            # LLVM can turn the single entry-block scope into static allocas.
            assert "llvm.stacksave" in llir
            assert "llvm.stackrestore" in llir
        assert meta.asm["asm"].count("Sleef_sinfx_u10sve") == depth
        assembly = meta.asm["asm"]
        assert ".cfi_def_cfa" in assembly, "Expected AArch64 stack-frame unwind information"
        prologue = assembly.split(".cfi_def_cfa", 1)[0]
        frame_bytes = sum(map(int, re.findall(r"\[sp,\s*#-(\d+)\]!", prologue)))
        for amount, shift in re.findall(r"sub\s+sp,\s*sp,\s*#(\d+)(?:,\s*lsl\s*#(\d+))?", prologue):
            frame_bytes += int(amount) << int(shift or 0)
        assert frame_bytes > 0
        frames.append(frame_bytes)

    # Dynamic scopes add at most one scratch pair at a time. The fixed frame
    # can spill a live earlier result, but must not grow by a full input/output
    # buffer pair for every added math operation.
    buffer_pair_bytes = 2 * size * src.element_size()
    assert frames[1] <= frames[0] + buffer_pair_bytes, frames


@pytest.mark.parametrize("mode", [0, 1])
def test_sleef_sve_scratch_scopes_in_loop_branch(mode, device):
    if not is_cpu() or not has_sve():
        pytest.skip("This test requires an SVE CPU")

    @triton.jit(do_not_specialize=["count", "mode"])
    def kernel(src, dst, count, mode, BLOCK_SIZE: tl.constexpr):
        offsets = tl.arange(0, BLOCK_SIZE)
        initial = tl.load(src + offsets)
        result = initial
        for _ in range(count):
            if mode == 0:
                result = tl.sin(result)
            else:
                result = tl.cos(result)
            result = tl.sqrt(result * result + 1.0)
        tl.store(dst + offsets, result + initial)

    size = 256
    count = 5
    src = torch.linspace(0.125, 3.125, size, dtype=torch.float32, device=device)
    res = torch.empty_like(src)
    meta = kernel[(1, )](src, res, count, mode, BLOCK_SIZE=size, vec_lib="libsleef")
    ref = src
    for _ in range(count):
        ref = ref.sin() if mode == 0 else ref.cos()
        ref = (ref * ref + 1.0).sqrt()
    torch.testing.assert_close(res, ref + src)
    assert "llvm.stacksave" in meta.asm["llir"]
    assert "llvm.stackrestore" in meta.asm["llir"]
