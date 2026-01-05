# ruff: noqa: F821
import numpy as np
import pytest
import torch

import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl
from triton._internal_testing import is_cuda, is_hip, is_interpreter


def _require_cuda_backend(device: str):
    # CUDA and HIP both use torch device 'cuda'. fpsan is currently plumbed through CUDAOptions.
    if device != "cuda":
        pytest.skip("fpsan tests require torch device 'cuda'")
    if is_interpreter():
        pytest.skip("fpsan tests require a real backend (not the interpreter)")
    if is_hip():
        pytest.skip("fpsan tests currently cover the CUDA backend only")
    if not is_cuda():
        pytest.skip("fpsan tests require CUDA")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")


def _as_u32(x_i32: np.ndarray) -> np.ndarray:
    assert x_i32.dtype == np.int32
    return x_i32.view(np.uint32)


def _u32_to_i32(x_u32: np.ndarray) -> np.ndarray:
    assert x_u32.dtype == np.uint32
    return x_u32.view(np.int32)


def _expected_add_i32(x_i32: np.ndarray, y_i32: np.ndarray) -> np.ndarray:
    x_u32 = _as_u32(x_i32).astype(np.uint64)
    y_u32 = _as_u32(y_i32).astype(np.uint64)
    out_u32 = ((x_u32 + y_u32) & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    return _u32_to_i32(out_u32)


def _expected_sub_i32(x_i32: np.ndarray, y_i32: np.ndarray) -> np.ndarray:
    x_u32 = _as_u32(x_i32).astype(np.uint64)
    y_u32 = _as_u32(y_i32).astype(np.uint64)
    out_u32 = ((x_u32 - y_u32) & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    return _u32_to_i32(out_u32)


def _expected_mul_i32(x_i32: np.ndarray, y_i32: np.ndarray) -> np.ndarray:
    x_u32 = _as_u32(x_i32).astype(np.uint64)
    y_u32 = _as_u32(y_i32).astype(np.uint64)
    out_u32 = ((x_u32 * y_u32) & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    return _u32_to_i32(out_u32)


def _expected_srem_i32(x_i32: np.ndarray, y_i32: np.ndarray) -> np.ndarray:
    # Match LLVM srem semantics: remainder after trunc-toward-zero division.
    # NOTE: Python/NumPy '%' uses floor division for negatives, so we implement explicitly.
    #
    # In fpsan mode we force denominator non-zero using `den | 1` in the *payload* domain.
    x = x_i32.astype(np.int64)
    y_safe_u32 = (_as_u32(y_i32) | np.uint32(1)).astype(np.uint32)
    y = _u32_to_i32(y_safe_u32).astype(np.int64)
    q = (np.sign(x) * np.sign(y) * (np.abs(x) // np.abs(y))).astype(np.int64)
    r = (x - q * y).astype(np.int32)
    return r


def _expected_div_payload_i32(x_i32: np.ndarray, y_i32: np.ndarray) -> np.ndarray:
    # fpsan division is defined as: num_bits * inv(den_bits | 1) mod 2^32
    # where inv is the multiplicative inverse in Z/(2^32).
    MOD = 1 << 32
    mask = np.uint64(0xFFFFFFFF)
    num = _as_u32(x_i32).astype(np.uint64)
    den = _as_u32(y_i32).astype(np.uint64)
    den_odd = (den | np.uint64(1)).astype(np.uint64)
    inv = np.array([pow(int(d), -1, MOD) for d in den_odd], dtype=np.uint64)
    out_u32 = ((num * inv) & mask).astype(np.uint32)
    return _u32_to_i32(out_u32)


@gluon.jit
def _binop_kernel(x_ptr, y_ptr, out_ptr, n_elements, OP: ttgl.constexpr, BLOCK: ttgl.constexpr):
    pid = ttgl.program_id(0)
    layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[2], threads_per_warp=[32], warps_per_cta=[4],
                                                order=[0])
    offs = pid * BLOCK + ttgl.arange(0, BLOCK, layout=layout)
    mask = offs < n_elements
    x = ttgl.load(x_ptr + offs, mask=mask, other=0.0)
    y = ttgl.load(y_ptr + offs, mask=mask, other=0.0)

    if OP == "add":
        z = x + y
    elif OP == "sub":
        z = x - y
    elif OP == "mul":
        z = x * y
    elif OP == "truediv":
        z = x / y
    elif OP == "fdiv":
        z = ttgl.fdiv(x, y)
    elif OP == "mod":
        z = x % y
    else:
        ttgl.static_assert(False, "unsupported OP")

    ttgl.store(out_ptr + offs, z, mask=mask)


@pytest.mark.parametrize(
    "op,expected_fn",
    [
        ("add", _expected_add_i32),
        ("sub", _expected_sub_i32),
        ("mul", _expected_mul_i32),
        ("truediv", _expected_div_payload_i32),
        ("fdiv", _expected_div_payload_i32),
        ("mod", _expected_srem_i32),
    ],
)
def test_fpsan_binops_payload_semantics(device, op, expected_fn):
    _require_cuda_backend(device)

    # Use int32 storage but treat it as float32 via TensorWrapper so fpsan operates on payload bits.
    n_elements = 1024
    BLOCK = 256

    g = torch.Generator(device="cuda")
    g.manual_seed(0)
    x = torch.randint(-(2**31), 2**31 - 1, (n_elements, ), dtype=torch.int32, device="cuda", generator=g)
    y = torch.randint(-(2**31), 2**31 - 1, (n_elements, ), dtype=torch.int32, device="cuda", generator=g)
    out = torch.empty((n_elements, ), dtype=torch.int32, device="cuda")

    xw = triton.TensorWrapper(x, dtype=torch.float32)
    yw = triton.TensorWrapper(y, dtype=torch.float32)
    outw = triton.TensorWrapper(out, dtype=torch.float32)

    grid = (triton.cdiv(n_elements, BLOCK), )
    _binop_kernel[grid](xw, yw, outw, n_elements, OP=op, BLOCK=BLOCK, fpsan=True, num_warps=4)

    out_np = out.cpu().numpy().astype(np.int32, copy=False)
    exp_np = expected_fn(x.cpu().numpy().astype(np.int32, copy=False), y.cpu().numpy().astype(np.int32, copy=False))
    np.testing.assert_array_equal(out_np, exp_np)


@gluon.jit
def _unary_math_kernel(x_ptr, out_ptr, n_elements, OP: ttgl.constexpr, BLOCK: ttgl.constexpr):
    pid = ttgl.program_id(0)
    layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[2], threads_per_warp=[32], warps_per_cta=[4],
                                                order=[0])
    offs = pid * BLOCK + ttgl.arange(0, BLOCK, layout=layout)
    mask = offs < n_elements
    x = ttgl.load(x_ptr + offs, mask=mask, other=0.0)
    z = getattr(ttgl, OP)(x)
    ttgl.store(out_ptr + offs, z, mask=mask)


@pytest.mark.parametrize(
    "op",
    [
        "exp",
        "exp2",
        "log",
        "log2",
        "cos",
        "sin",
        "sqrt",
        "sqrt_rn",
        "rsqrt",
        "erf",
        "floor",
        "ceil",
    ],
)
def test_fpsan_unary_math_identity(device, op):
    _require_cuda_backend(device)

    n_elements = 1024
    BLOCK = 256
    rs = np.random.RandomState(0)
    # Includes negative values for log/sqrt on purpose; fpsan treats them as identity.
    xf = rs.randn(n_elements).astype(np.float32)
    x_bits = xf.view(np.int32)

    x = torch.tensor(x_bits, dtype=torch.int32, device="cuda")
    out = torch.empty((n_elements, ), dtype=torch.int32, device="cuda")

    grid = (triton.cdiv(n_elements, BLOCK), )
    _unary_math_kernel[grid](
        triton.TensorWrapper(x, dtype=torch.float32),
        triton.TensorWrapper(out, dtype=torch.float32),
        n_elements,
        OP=op,
        BLOCK=BLOCK,
        fpsan=True,
        num_warps=4,
    )

    np.testing.assert_array_equal(out.cpu().numpy().astype(np.int32, copy=False), x_bits)
