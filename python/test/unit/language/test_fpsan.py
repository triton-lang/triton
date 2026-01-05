# ruff: noqa: F821
import numpy as np
import pytest
import torch

import triton
import triton.language as tl

from triton._internal_testing import is_cuda, is_hip, is_interpreter


def _require_cuda_backend():
    # CUDA and HIP both use torch device 'cuda'. fpsan is currently plumbed through CUDAOptions.
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


@triton.jit
def _binop_kernel(x_ptr, y_ptr, out_ptr, N: tl.constexpr, OP: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)

    if OP == "add":
        z = x + y
    elif OP == "sub":
        z = x - y
    elif OP == "mul":
        z = x * y
    elif OP == "truediv":
        z = x / y
    elif OP == "fdiv":
        z = tl.math.fdiv(x, y)
    elif OP == "mod":
        z = x % y
    else:
        tl.static_assert(False, "unsupported OP")

    tl.store(out_ptr + offs, z, mask=mask)


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
def test_fpsan_binops_payload_semantics(op, expected_fn):
    _require_cuda_backend()

    # Use int32 storage but treat it as float32 via TensorWrapper so fpsan operates on payload bits.
    N = 1024
    BLOCK = 256

    g = torch.Generator(device="cuda")
    g.manual_seed(0)
    x = torch.randint(-(2**31), 2**31 - 1, (N, ), dtype=torch.int32, device="cuda", generator=g)
    y = torch.randint(-(2**31), 2**31 - 1, (N, ), dtype=torch.int32, device="cuda", generator=g)

    out = torch.empty((N, ), dtype=torch.int32, device="cuda")

    xw = triton.TensorWrapper(x, dtype=torch.float32)
    yw = triton.TensorWrapper(y, dtype=torch.float32)
    outw = triton.TensorWrapper(out, dtype=torch.float32)

    grid = (triton.cdiv(N, BLOCK), )
    _binop_kernel[grid](xw, yw, outw, N=N, OP=op, BLOCK=BLOCK, fpsan=True)

    out_np = out.cpu().numpy().astype(np.int32, copy=False)
    exp_np = expected_fn(x.cpu().numpy().astype(np.int32, copy=False), y.cpu().numpy().astype(np.int32, copy=False))
    np.testing.assert_array_equal(out_np, exp_np)
