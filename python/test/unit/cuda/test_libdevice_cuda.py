# fmt: off

import numpy as np
import pytest
import torch
import triton
import triton.language as tl

from triton._internal_testing import is_cuda, numpy_random, to_numpy, to_triton
from triton.language.extra import libdevice


# -----------------------
# test extern functions
# -----------------------


@triton.jit
def tanh_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    direct_import: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)

    if direct_import:
        y = libdevice.tanh(x)
    else:
        y = tl.extra.libdevice.tanh(x)

    tl.store(y_ptr + offsets, y, mask=mask)


@pytest.mark.parametrize("direct_import", [False, True])
@pytest.mark.parametrize("dtype_str", ['float32', 'float64'])
def test_math_extern(dtype_str, direct_import):

    if not torch.cuda.is_available():
        pytest.skip("Test requires CUDA target.")
        return

    torch.manual_seed(42)

    x = torch.randn((100,), dtype=getattr(torch, dtype_str), device="cuda")

    y_tri = torch.empty_like(x)
    tanh_kernel[(1, )](x, y_tri, x.shape[0], direct_import, BLOCK_SIZE=128)

    y_ref = torch.tanh(x)
    np.testing.assert_allclose(y_ref.cpu().numpy(), y_tri.cpu().numpy(), rtol=0, atol=1.0e-6)


@triton.jit
def builtin_libdevice_unary_kernel(
    x_ptr,
    builtin_ptr,
    libdevice_ptr,
    n_elements,
    OP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)

    if OP == "exp":
        builtin = tl.exp(x)
        external = libdevice.fast_expf(x)
    elif OP == "sqrt_rn":
        builtin = tl.sqrt_rn(x)
        external = libdevice.sqrt_rn(x)
    elif OP == "exp2":
        builtin = tl.exp2(x)
        external = libdevice.exp2(x)
    elif OP == "log":
        builtin = tl.log(x)
        external = libdevice.log(x)
    elif OP == "log2":
        builtin = tl.log2(x)
        external = libdevice.log2(x)
    elif OP == "sin":
        builtin = tl.sin(x)
        external = libdevice.sin(x)
    elif OP == "cos":
        builtin = tl.cos(x)
        external = libdevice.cos(x)
    elif OP == "rsqrt":
        builtin = tl.rsqrt(x)
        external = libdevice.rsqrt(x)
    elif OP == "erf":
        builtin = tl.erf(x)
        external = libdevice.erf(x)
    elif OP == "floor":
        builtin = tl.floor(x)
        external = libdevice.floor(x)
    elif OP == "ceil":
        builtin = tl.ceil(x)
        external = libdevice.ceil(x)
    elif OP == "fast_sin":
        builtin = tl.sin(x)
        external = libdevice.fast_sinf(x)
    elif OP == "fast_cos":
        builtin = tl.cos(x)
        external = libdevice.fast_cosf(x)
    elif OP == "fast_log":
        builtin = tl.log(x)
        external = libdevice.fast_logf(x)
    elif OP == "fast_log2":
        builtin = tl.log2(x)
        external = libdevice.fast_log2f(x)

    tl.store(builtin_ptr + offsets, builtin, mask=mask)
    tl.store(libdevice_ptr + offsets, external, mask=mask)


@triton.jit
def builtin_libdevice_div_rn_kernel(
    x_ptr,
    y_ptr,
    builtin_ptr,
    libdevice_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    tl.store(builtin_ptr + offsets, tl.div_rn(x, y), mask=mask)
    tl.store(libdevice_ptr + offsets, libdevice.div_rn(x, y), mask=mask)


@triton.jit
def builtin_libdevice_fma_kernel(
    x_ptr,
    y_ptr,
    z_ptr,
    builtin_ptr,
    libdevice_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    z = tl.load(z_ptr + offsets, mask=mask)

    tl.store(builtin_ptr + offsets, tl.fma(x, y, z), mask=mask)
    tl.store(libdevice_ptr + offsets, libdevice.fma(x, y, z), mask=mask)


def _random_payloads(generator, n_elements):
    return torch.randint(-(2**31), 2**31 - 1, (n_elements,), dtype=torch.int32, device="cuda", generator=generator)


@pytest.mark.parametrize(
    "op", ["exp", "exp2", "log", "log2", "sin", "cos", "sqrt_rn", "rsqrt", "erf", "floor", "ceil"]
)
def test_fpsan_libdevice_unary_equivalence(op, fresh_knobs):
    if not is_cuda() or not torch.cuda.is_available():
        pytest.skip("test requires CUDA")

    fresh_knobs.compilation.instrumentation_mode = "fpsan"
    n_elements = 1024
    block_size = 256
    generator = torch.Generator(device="cuda")
    generator.manual_seed(0)
    x = _random_payloads(generator, n_elements)
    builtin = torch.empty_like(x)
    external = torch.empty_like(x)

    builtin_libdevice_unary_kernel[(triton.cdiv(n_elements, block_size),)](
        triton.TensorWrapper(x, dtype=torch.float32),
        triton.TensorWrapper(builtin, dtype=torch.float32),
        triton.TensorWrapper(external, dtype=torch.float32),
        n_elements,
        OP=op,
        BLOCK_SIZE=block_size,
    )

    assert torch.equal(builtin, external)


def test_fpsan_libdevice_div_rn_equivalence(fresh_knobs):
    if not is_cuda() or not torch.cuda.is_available():
        pytest.skip("test requires CUDA")

    fresh_knobs.compilation.instrumentation_mode = "fpsan"
    n_elements = 1024
    block_size = 256
    generator = torch.Generator(device="cuda")
    generator.manual_seed(0)
    inputs = [_random_payloads(generator, n_elements) for _ in range(2)]
    builtin = torch.empty_like(inputs[0])
    external = torch.empty_like(inputs[0])

    builtin_libdevice_div_rn_kernel[(triton.cdiv(n_elements, block_size),)](
        *(triton.TensorWrapper(value, dtype=torch.float32) for value in inputs),
        triton.TensorWrapper(builtin, dtype=torch.float32),
        triton.TensorWrapper(external, dtype=torch.float32),
        n_elements,
        BLOCK_SIZE=block_size,
    )

    assert torch.equal(builtin, external)


def test_fpsan_libdevice_fma_equivalence(fresh_knobs):
    if not is_cuda() or not torch.cuda.is_available():
        pytest.skip("test requires CUDA")

    fresh_knobs.compilation.instrumentation_mode = "fpsan"
    n_elements = 1024
    block_size = 256
    generator = torch.Generator(device="cuda")
    generator.manual_seed(0)
    inputs = [_random_payloads(generator, n_elements) for _ in range(3)]
    builtin = torch.empty_like(inputs[0])
    external = torch.empty_like(inputs[0])

    builtin_libdevice_fma_kernel[(triton.cdiv(n_elements, block_size),)](
        *(triton.TensorWrapper(value, dtype=torch.float32) for value in inputs),
        triton.TensorWrapper(builtin, dtype=torch.float32),
        triton.TensorWrapper(external, dtype=torch.float32),
        n_elements,
        BLOCK_SIZE=block_size,
    )

    assert torch.equal(builtin, external)


@pytest.mark.parametrize("op", ["fast_sin", "fast_cos", "fast_log", "fast_log2"])
def test_fpsan_distinguishes_approximate_libdevice(op, fresh_knobs):
    if not is_cuda() or not torch.cuda.is_available():
        pytest.skip("test requires CUDA")

    fresh_knobs.compilation.instrumentation_mode = "fpsan"
    n_elements = 1024
    block_size = 256
    generator = torch.Generator(device="cuda")
    generator.manual_seed(0)
    x = _random_payloads(generator, n_elements)
    builtin = torch.empty_like(x)
    external = torch.empty_like(x)

    builtin_libdevice_unary_kernel[(triton.cdiv(n_elements, block_size),)](
        triton.TensorWrapper(x, dtype=torch.float32),
        triton.TensorWrapper(builtin, dtype=torch.float32),
        triton.TensorWrapper(external, dtype=torch.float32),
        n_elements,
        OP=op,
        BLOCK_SIZE=block_size,
    )

    assert not torch.equal(builtin, external)


# Integer min/max are NVIDIA-only libdevice builtins (__nv_min/__nv_umin/__nv_llmin/...),
# so they live here rather than in the cross-backend test_libdevice.py. Data is generated
# via numpy_random/to_triton because torch lacks randint ranges and maximum/minimum for
# the unsigned dtypes.
@pytest.mark.parametrize("dtype_str", ["int32", "int64", "uint32", "uint64"])
@pytest.mark.parametrize("fn", ["max", "min"])
def test_int_max_min(fn, dtype_str, device):
    if not is_cuda():
        pytest.skip("integer max/min are NVIDIA-only libdevice builtins")

    @triton.jit
    def kernel(x_ptr, y_ptr, out_ptr, fn: tl.constexpr, SIZE: tl.constexpr):
        off = tl.arange(0, SIZE)
        x = tl.load(x_ptr + off)
        y = tl.load(y_ptr + off)
        tl.store(out_ptr + off, getattr(libdevice, fn)(x, y))

    SIZE = 128
    rs = np.random.RandomState(seed=0)
    x = numpy_random(SIZE, dtype_str=dtype_str, rs=rs)
    y = numpy_random(SIZE, dtype_str=dtype_str, rs=rs)
    x_tri = to_triton(x, device=device, dst_type=dtype_str)
    y_tri = to_triton(y, device=device, dst_type=dtype_str)
    out_tri = to_triton(np.empty(SIZE, dtype=getattr(np, dtype_str)), device=device, dst_type=dtype_str)
    kernel[(1, )](x_tri, y_tri, out_tri, fn=fn, SIZE=SIZE)

    ref = np.maximum(x, y) if fn == "max" else np.minimum(x, y)
    np.testing.assert_array_equal(to_numpy(out_tri), ref)
