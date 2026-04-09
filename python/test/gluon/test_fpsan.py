# ruff: noqa: F821
import itertools
import numpy as np
import pytest
import torch

import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton import language as tl
from triton._internal_testing import is_blackwell, is_cuda, is_hip, is_hip_cdna3, is_hip_cdna4, is_hip_gfx1250, is_interpreter
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    TensorMemoryScalesLayout,
    allocate_tensor_memory,
    mbarrier,
    tcgen05_commit,
    tcgen05_mma,
    tcgen05_mma_scaled,
)

THREADS_PER_WARP = triton.runtime.driver.active.get_current_target().warp_size


def _hip_device_supports_fpsan():
    return is_hip_cdna3() or is_hip_cdna4() or is_hip_gfx1250()


def _require_cuda_backend(device: str):
    # CUDA and HIP both use torch device 'cuda'. fpsan is plumbed through both CUDAOptions and HIPOptions.
    if device != "cuda":
        pytest.skip("fpsan tests require torch device 'cuda'")
    if is_interpreter():
        pytest.skip("fpsan tests require a real backend (not the interpreter)")
    if not (is_cuda() or is_hip()):
        pytest.skip("fpsan tests require CUDA or HIP")
    if is_hip() and not _hip_device_supports_fpsan():
        pytest.skip("fpsan is not supported on this HIP device")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")


def _as_u32(x_i32: np.ndarray) -> np.ndarray:
    assert x_i32.dtype == np.int32
    return x_i32.view(np.uint32)


def _u32_to_i32(x_u32: np.ndarray) -> np.ndarray:
    assert x_u32.dtype == np.uint32
    return x_u32.view(np.int32)


def _low_mask_u64(bitwidth: int) -> np.uint64:
    return np.uint64(2**bitwidth - 1)


def _inv_odd_u64(a: np.uint64) -> np.uint64:
    with np.errstate(over="ignore"):
        x = np.uint64(2) - a
        for _ in range(5):
            x = np.uint64(x * (np.uint64(2) - np.uint64(a * x)))
        return x


def _mix_config(bitwidth: int, one_bits: int) -> tuple[np.uint64, np.uint64, np.uint64, int, np.uint64, np.uint64]:
    with np.errstate(over="ignore"):
        full_mask = _low_mask_u64(bitwidth)
        sign_mask = np.uint64(1 << (bitwidth - 1))
        mag_mask = sign_mask - np.uint64(1)
        shift = int((one_bits & -one_bits).bit_length() - 1)
        y = (np.uint64(one_bits) * np.uint64(922291)) & mag_mask
        z = y ^ (y >> np.uint64(shift))
        mul_b_pos = _inv_odd_u64(z) & full_mask
        mul_b_neg = (mul_b_pos * mag_mask) & full_mask
        return full_mask, sign_mask, mag_mask, shift, mul_b_pos, mul_b_neg


def _xor_shift_right_u64(x: np.ndarray, shift: int) -> np.ndarray:
    return x ^ (x >> np.uint64(shift))


def _inverse_xor_shift_right_u64(x: np.ndarray, shift: int, bitwidth: int) -> np.ndarray:
    while shift < bitwidth:
        x = _xor_shift_right_u64(x, shift)
        shift *= 2
    return x


def _mix_float_bits_to_payload_u64(bits, bitwidth: int, one_bits: int) -> np.ndarray:
    full_mask, sign_mask, mag_mask, shift, mul_b_pos, mul_b_neg = _mix_config(bitwidth, one_bits)
    x = bits.astype(np.uint64) & full_mask
    neg = (x & sign_mask) != 0
    sign = np.where(neg, sign_mask, np.uint64(0))
    y = (((x ^ sign) * np.uint64(922291)) & mag_mask)
    z = _xor_shift_right_u64(y, shift)
    factor = np.where(neg, mul_b_neg, mul_b_pos)
    return (((z * factor) & mag_mask) ^ sign) & full_mask


def _unmix_payload_u64_to_float_bits(payload, bitwidth: int, one_bits: int) -> np.ndarray:
    full_mask, sign_mask, mag_mask, shift, mul_b_pos, mul_b_neg = _mix_config(bitwidth, one_bits)
    v = payload.astype(np.uint64) & full_mask
    neg = (v & sign_mask) != 0
    sign = np.where(neg, sign_mask, np.uint64(0))
    factor = np.where(neg, _inv_odd_u64(mul_b_neg), _inv_odd_u64(mul_b_pos))
    z = (((v ^ sign) * factor) & mag_mask)
    y = _inverse_xor_shift_right_u64(z, shift, bitwidth)
    x = (y * _inv_odd_u64(np.uint64(922291))) & mag_mask
    return (x ^ sign) & full_mask


def _mix_f32_bits_to_payload_u32(x_i32: np.ndarray) -> np.ndarray:
    return _mix_float_bits_to_payload_u64(_as_u32(x_i32), 32, 0x3F800000).astype(np.uint32)


def _unmix_payload_u32_to_f32_bits_i32(v_u32: np.ndarray) -> np.ndarray:
    assert v_u32.dtype == np.uint32
    return _u32_to_i32(_unmix_payload_u64_to_float_bits(v_u32, 32, 0x3F800000).astype(np.uint32))


def _signed_cast_payload_u64(payload, src_bitwidth: int, dst_bitwidth: int) -> np.ndarray:
    x = payload.astype(np.uint64) & _low_mask_u64(src_bitwidth)
    if dst_bitwidth <= src_bitwidth:
        return x & _low_mask_u64(dst_bitwidth)

    sign = np.uint64(1 << (src_bitwidth - 1))
    extension = _low_mask_u64(dst_bitwidth) ^ _low_mask_u64(src_bitwidth)
    return np.where((x & sign) != 0, x | extension, x) & _low_mask_u64(dst_bitwidth)


def _payload_u32_to_f32_bits_i32(x_u64: np.ndarray) -> np.ndarray:
    return _unmix_payload_u32_to_f32_bits_i32((x_u64 & np.uint64(0xFFFFFFFF)).astype(np.uint32))


def _expected_add_i32(x_i32: np.ndarray, y_i32: np.ndarray) -> np.ndarray:
    x_u32 = _mix_f32_bits_to_payload_u32(x_i32).astype(np.uint64)
    y_u32 = _mix_f32_bits_to_payload_u32(y_i32).astype(np.uint64)
    return _payload_u32_to_f32_bits_i32(x_u32 + y_u32)


def _expected_sub_i32(x_i32: np.ndarray, y_i32: np.ndarray) -> np.ndarray:
    x_u32 = _mix_f32_bits_to_payload_u32(x_i32).astype(np.uint64)
    y_u32 = _mix_f32_bits_to_payload_u32(y_i32).astype(np.uint64)
    return _payload_u32_to_f32_bits_i32(x_u32 - y_u32)


def _expected_mul_i32(x_i32: np.ndarray, y_i32: np.ndarray) -> np.ndarray:
    x_u32 = _mix_f32_bits_to_payload_u32(x_i32).astype(np.uint64)
    y_u32 = _mix_f32_bits_to_payload_u32(y_i32).astype(np.uint64)
    return _payload_u32_to_f32_bits_i32(x_u32 * y_u32)


def _expected_srem_i32(x_i32: np.ndarray, y_i32: np.ndarray) -> np.ndarray:
    # Match LLVM srem semantics: remainder after trunc-toward-zero division.
    # NOTE: Python/NumPy '%' uses floor division for negatives, so we implement explicitly.
    #
    # In fpsan mode we force denominator non-zero using `den | 1` in the *payload* domain.
    x = _u32_to_i32(_mix_f32_bits_to_payload_u32(x_i32)).astype(np.int64)
    y_safe_u32 = (_mix_f32_bits_to_payload_u32(y_i32) | np.uint32(1)).astype(np.uint32)
    y = _u32_to_i32(y_safe_u32).astype(np.int64)
    q = (np.sign(x) * np.sign(y) * (np.abs(x) // np.abs(y))).astype(np.int64)
    r = (x - q * y).astype(np.int32).view(np.uint32)
    return _unmix_payload_u32_to_f32_bits_i32(r)


def murmur64Mixer(h: np.uint64) -> np.uint64:
    with np.errstate(over="ignore"):
        h = np.uint64(h)
        h ^= h >> np.uint64(33)
        h = np.uint64(h * np.uint64(0xff51afd7ed558ccd))
        h ^= h >> np.uint64(33)
        h = np.uint64(h * np.uint64(0xc4ceb9fe1a85ec53))
        h ^= h >> np.uint64(33)
        return np.uint64(h)


OP_TO_ID_U64 = {
    "exp": np.uint64(0),
    "log": np.uint64(1),
    "exp2": np.uint64(2),
    "log2": np.uint64(3),
    "cos": np.uint64(4),
    "sin": np.uint64(5),
    "sqrt": np.uint64(6),
    "rsqrt": np.uint64(7),
    "erf": np.uint64(8),
    "floor": np.uint64(9),
    "ceil": np.uint64(10),
    "sqrt_rn": np.uint64(11),
}

OP_TO_TAG_U32 = {name: np.uint32(murmur64Mixer(op_id) & np.uint64(0xFFFFFFFF)) for name, op_id in OP_TO_ID_U64.items()}
UNARY_TAG_MULTIPLIER_U64 = np.uint64(314159)


def _expected_unary_tag_payload_u32(x_u32: np.ndarray, op: str) -> np.ndarray:
    tag = OP_TO_TAG_U32[op].astype(np.uint64)
    x = x_u32.astype(np.uint64)
    out_u64 = (((x * UNARY_TAG_MULTIPLIER_U64) & np.uint64(0xFFFFFFFF)) ^ tag) * UNARY_TAG_MULTIPLIER_U64
    return (out_u64 & np.uint64(0xFFFFFFFF)).astype(np.uint32)


def _expected_u32_inv(x_u32: np.ndarray) -> np.ndarray:
    mask = np.uint64(0xFFFFFFFF)
    a = x_u32.astype(np.uint64) | np.uint64(1)
    x = (np.uint64(2) - a) & mask
    for _ in range(4):
        factor = (np.uint64(2) - ((a * x) & mask)) & mask
        x = (x * factor) & mask
    x = (x & np.uint64(0xFFFFFFFE)) | (x_u32.astype(np.uint64) & np.uint64(1))
    return x.astype(np.uint32)


def _expected_div_payload_i32(x_i32: np.ndarray, y_i32: np.ndarray) -> np.ndarray:
    # fpsan division is defined as num_bits * u32_inv(den_bits) mod 2^32.
    num = _mix_f32_bits_to_payload_u32(x_i32).astype(np.uint64)
    inv = _expected_u32_inv(_mix_f32_bits_to_payload_u32(y_i32)).astype(np.uint64)
    return _payload_u32_to_f32_bits_i32(num * inv)


def _expected_exp2_i32(x_i32: np.ndarray) -> np.ndarray:
    c = np.uint64(0xa343836d)
    mask = np.uint64(0xFFFFFFFF)
    x = _mix_f32_bits_to_payload_u32(x_i32).astype(np.uint64)
    y = np.ones_like(x, dtype=np.uint64)
    for i in range(32):
        y = (y * y) & mask
        factor = np.where((x & np.uint64(1 << (31 - i))) == 0, np.uint64(1), c)
        y = (y * factor) & mask
    return _unmix_payload_u32_to_f32_bits_i32(y.astype(np.uint32))


def _expected_exp_i32(x_i32: np.ndarray) -> np.ndarray:
    x = _mix_f32_bits_to_payload_u32(x_i32).astype(np.uint64)
    rcp_log2 = np.uint64(0x236ee9bf)
    scaled = ((x * rcp_log2) & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    return _expected_exp2_i32(_unmix_payload_u32_to_f32_bits_i32(scaled))


def _expected_unary_tag_i32(x_i32: np.ndarray, op: str) -> np.ndarray:
    # Keep this mapping in sync with UnaryOpId in FpSanitizer.cpp.
    out_u32 = _expected_unary_tag_payload_u32(_mix_f32_bits_to_payload_u32(x_i32), op)
    return _unmix_payload_u32_to_f32_bits_i32(out_u32)


def stable_string_hash_u64(s: str) -> np.uint64:
    with np.errstate(over="ignore"):
        h = np.uint64(14695981039346656037)
        for c in s.encode("utf-8"):
            h ^= np.uint64(c)
            h = np.uint64(h * np.uint64(1099511628211))
        return h


def _expected_extern_unary_tag_i32(x_i32: np.ndarray, symbol: str) -> np.ndarray:
    return _expected_extern_variadic_tag_i32([x_i32], symbol)


def _expected_extern_binary_tag_i32(x_i32: np.ndarray, y_i32: np.ndarray, symbol: str) -> np.ndarray:
    return _expected_extern_variadic_tag_i32([x_i32, y_i32], symbol)


def _rotl_u32(x_u32: np.ndarray, amount: int) -> np.ndarray:
    amount = amount & 31
    if amount == 0:
        return x_u32
    x = x_u32.astype(np.uint64)
    out_u32 = ((x << np.uint64(amount)) | (x >> np.uint64(32 - amount))) & np.uint64(0xFFFFFFFF)
    return out_u32.astype(np.uint32)


def _expected_extern_variadic_tag_i32(args_i32: list[np.ndarray], symbol: str, float_args=None) -> np.ndarray:
    if float_args is None:
        float_args = [True] * len(args_i32)
    tag = np.uint64(stable_string_hash_u64(symbol) & np.uint64(0xFFFFFFFF))
    total_u64 = np.zeros_like(_as_u32(args_i32[0]), dtype=np.uint64)
    for i, (arg, is_float) in enumerate(zip(args_i32, float_args)):
        arg_u32 = _mix_f32_bits_to_payload_u32(arg) if is_float else _as_u32(arg)
        rotated = _rotl_u32(arg_u32, i).astype(np.uint64)
        total_u64 = (total_u64 + rotated) & np.uint64(0xFFFFFFFF)
    out_u32 = (total_u64 ^ tag).astype(np.uint32)
    return _unmix_payload_u32_to_f32_bits_i32(out_u32)


UNARY_EXTERN_SYMBOLS = {
    "cuda": [
        ("tan", "__nv_tanf"),
        ("tanh", "__nv_tanhf"),
        ("log1p", "__nv_log1pf"),
        ("cbrt", "__nv_cbrtf"),
        ("round", "__nv_roundf"),
    ],
    "hip": [
        ("tan", "__ocml_tan_f32"),
        ("tanh", "__ocml_tanh_f32"),
        ("log1p", "__ocml_log1p_f32"),
        ("round", "__ocml_round_f32"),
    ],
}

BINARY_EXTERN_SYMBOLS = {
    "cuda": [
        ("atan2", "__nv_atan2f"),
        ("hypot", "__nv_hypotf"),
        ("pow", "__nv_powf"),
    ],
    "hip": [
        ("atan2", "__ocml_atan2_f32"),
        ("hypot", "__ocml_hypot_f32"),
        ("pow", "__ocml_pow_f32"),
    ],
}

TERNARY_EXTERN_SYMBOLS = {
    "cuda": [
        ("fma", "__nv_fmaf"),
    ],
    "hip": [
        ("fma", "__ocml_fma_f32"),
    ],
}

MIXED_EXTERN_SYMBOLS = {
    "cuda": [
        ("ldexp", "__nv_ldexpf"),
    ],
    "hip": [
        ("ldexp", "__ocml_ldexp_f32"),
    ],
}


def _extern_backend_name() -> str:
    if is_hip():
        return "hip"
    return "cuda"


EXTERN_UNARY_CASES = UNARY_EXTERN_SYMBOLS[_extern_backend_name()]
EXTERN_BINARY_CASES = BINARY_EXTERN_SYMBOLS[_extern_backend_name()]
EXTERN_TERNARY_CASES = TERNARY_EXTERN_SYMBOLS[_extern_backend_name()]
EXTERN_MIXED_CASES = MIXED_EXTERN_SYMBOLS[_extern_backend_name()]


def _as_payload_np_i32(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if not isinstance(x, np.ndarray):
        raise TypeError(f"unsupported input type: {type(x)}")
    if x.dtype == np.int32:
        return x.astype(np.int32, copy=False)
    if x.dtype == np.uint32:
        return x.view(np.int32)
    if x.dtype == np.float32:
        return x.view(np.int32)
    raise TypeError(f"unsupported dtype for payload comparison: {x.dtype}")


def _assert_payload_equal(actual, expected) -> None:
    np.testing.assert_array_equal(_as_payload_np_i32(actual), _as_payload_np_i32(expected))


def _payload_equal(a, b) -> bool:
    return np.array_equal(_as_payload_np_i32(a), _as_payload_np_i32(b))


@gluon.jit
def _binop_kernel(x_ptr, y_ptr, out_ptr, n_elements, OP: gl.constexpr, BLOCK: gl.constexpr,
                  THREADS_PER_WARP: gl.constexpr):
    pid = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout(size_per_thread=[2], threads_per_warp=[THREADS_PER_WARP], warps_per_cta=[4],
                                            order=[0])
    offs = pid * BLOCK + gl.arange(0, BLOCK, layout=layout)
    mask = offs < n_elements
    x = gl.load(x_ptr + offs, mask=mask, other=0.0)
    y = gl.load(y_ptr + offs, mask=mask, other=0.0)

    if OP == "add":
        z = x + y
    elif OP == "sub":
        z = x - y
    elif OP == "mul":
        z = x * y
    elif OP == "truediv":
        z = x / y
    elif OP == "fdiv":
        z = gl.fdiv(x, y)
    elif OP == "mod":
        z = x % y
    else:
        gl.static_assert(False, "unsupported OP")

    gl.store(out_ptr + offs, z, mask=mask)


@gluon.jit
def _constant_identity_kernel(x_ptr, out_ptr, n_elements, OP: gl.constexpr, BLOCK: gl.constexpr,
                              THREADS_PER_WARP: gl.constexpr):
    pid = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout(size_per_thread=[2], threads_per_warp=[THREADS_PER_WARP], warps_per_cta=[4],
                                            order=[0])
    offs = pid * BLOCK + gl.arange(0, BLOCK, layout=layout)
    mask = offs < n_elements
    x = gl.load(x_ptr + offs, mask=mask, other=0.0)

    if OP == "mul_one":
        z = x * 1.0
    elif OP == "add_zero":
        z = x + 0.0
    else:
        gl.static_assert(False, "unsupported OP")

    gl.store(out_ptr + offs, z, mask=mask)


@gluon.jit
def _reciprocal_involution_kernel(x_ptr, out_ptr, n_elements, BLOCK: gl.constexpr, THREADS_PER_WARP: gl.constexpr):
    pid = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout(size_per_thread=[2], threads_per_warp=[THREADS_PER_WARP], warps_per_cta=[4],
                                            order=[0])
    offs = pid * BLOCK + gl.arange(0, BLOCK, layout=layout)
    mask = offs < n_elements
    x = gl.load(x_ptr + offs, mask=mask, other=0.0)
    z = 1.0 / (1.0 / x)
    gl.store(out_ptr + offs, z, mask=mask)


@pytest.mark.parametrize("op", ["mul_one", "add_zero"])
def test_constant_identity_noop(device, op, fresh_knobs):
    _require_cuda_backend(device)

    fresh_knobs.compilation.instrumentation_mode = "fpsan"

    n_elements = 1024
    BLOCK = 256

    g = torch.Generator(device="cuda")
    g.manual_seed(2)
    x = torch.randint(-(2**31), 2**31 - 1, (n_elements, ), dtype=torch.int32, device="cuda", generator=g)
    out = torch.empty((n_elements, ), dtype=torch.int32, device="cuda")

    grid = (triton.cdiv(n_elements, BLOCK), )
    _constant_identity_kernel[grid](
        triton.TensorWrapper(x, dtype=torch.float32),
        triton.TensorWrapper(out, dtype=torch.float32),
        n_elements,
        OP=op,
        BLOCK=BLOCK,
        THREADS_PER_WARP=THREADS_PER_WARP,
    )

    _assert_payload_equal(out, x)


def test_reciprocal_involution(device, fresh_knobs):
    _require_cuda_backend(device)

    fresh_knobs.compilation.instrumentation_mode = "fpsan"

    n_elements = 1024
    BLOCK = 256

    g = torch.Generator(device="cuda")
    g.manual_seed(3)
    x = torch.randint(-(2**31), 2**31 - 1, (n_elements, ), dtype=torch.int32, device="cuda", generator=g)
    out = torch.empty((n_elements, ), dtype=torch.int32, device="cuda")

    grid = (triton.cdiv(n_elements, BLOCK), )
    _reciprocal_involution_kernel[grid](
        triton.TensorWrapper(x, dtype=torch.float32),
        triton.TensorWrapper(out, dtype=torch.float32),
        n_elements,
        BLOCK=BLOCK,
        THREADS_PER_WARP=THREADS_PER_WARP,
    )

    _assert_payload_equal(out, x)


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
def test_binops_payload_semantics(device, op, expected_fn, fresh_knobs):
    _require_cuda_backend(device)

    fresh_knobs.compilation.instrumentation_mode = "fpsan"

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
    _binop_kernel[grid](xw, yw, outw, n_elements, OP=op, BLOCK=BLOCK, THREADS_PER_WARP=THREADS_PER_WARP)

    out_np = out.cpu().numpy().astype(np.int32, copy=False)
    exp_np = expected_fn(x.cpu().numpy().astype(np.int32, copy=False), y.cpu().numpy().astype(np.int32, copy=False))
    _assert_payload_equal(out_np, exp_np)


@pytest.mark.parametrize(
    "op,expected_fn",
    [
        ("truediv", _expected_div_payload_i32),
        ("fdiv", _expected_div_payload_i32),
        ("mod", _expected_srem_i32),
    ],
)
def test_binops_payload_semantics_zero_denominator(device, op, expected_fn, fresh_knobs):
    _require_cuda_backend(device)

    fresh_knobs.compilation.instrumentation_mode = "fpsan"

    n_elements = 1024
    BLOCK = 256

    g = torch.Generator(device="cuda")
    g.manual_seed(123)
    x = torch.randint(-(2**31), 2**31 - 1, (n_elements, ), dtype=torch.int32, device="cuda", generator=g)
    y = torch.randint(-(2**31), 2**31 - 1, (n_elements, ), dtype=torch.int32, device="cuda", generator=g)
    y[::7] = 0

    out = torch.empty((n_elements, ), dtype=torch.int32, device="cuda")

    xw = triton.TensorWrapper(x, dtype=torch.float32)
    yw = triton.TensorWrapper(y, dtype=torch.float32)
    outw = triton.TensorWrapper(out, dtype=torch.float32)

    grid = (triton.cdiv(n_elements, BLOCK), )
    _binop_kernel[grid](xw, yw, outw, n_elements, OP=op, BLOCK=BLOCK, THREADS_PER_WARP=THREADS_PER_WARP)

    out_np = out.cpu().numpy().astype(np.int32, copy=False)
    exp_np = expected_fn(x.cpu().numpy().astype(np.int32, copy=False), y.cpu().numpy().astype(np.int32, copy=False))
    _assert_payload_equal(out_np, exp_np)


@gluon.jit
def _unary_math_kernel(x_ptr, out_ptr, n_elements, OP: gl.constexpr, BLOCK: gl.constexpr,
                       THREADS_PER_WARP: gl.constexpr):
    pid = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout(size_per_thread=[2], threads_per_warp=[THREADS_PER_WARP], warps_per_cta=[4],
                                            order=[0])
    offs = pid * BLOCK + gl.arange(0, BLOCK, layout=layout)
    mask = offs < n_elements
    x = gl.load(x_ptr + offs, mask=mask, other=0.0)
    z = getattr(gl, OP)(x)
    gl.store(out_ptr + offs, z, mask=mask)


@gluon.jit
def _exp_binary_identity_kernel(x_ptr, y_ptr, out_ptr, n_elements, MODE: gl.constexpr, BLOCK: gl.constexpr,
                                THREADS_PER_WARP: gl.constexpr):
    pid = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout(size_per_thread=[2], threads_per_warp=[THREADS_PER_WARP], warps_per_cta=[4],
                                            order=[0])
    offs = pid * BLOCK + gl.arange(0, BLOCK, layout=layout)
    mask = offs < n_elements
    x = gl.load(x_ptr + offs, mask=mask, other=0.0)
    y = gl.load(y_ptr + offs, mask=mask, other=0.0)
    if MODE == "exp_add":
        z = gl.exp(x + y)
    elif MODE == "exp_mul":
        z = gl.exp(x) * gl.exp(y)
    else:
        gl.static_assert(False, "unsupported MODE")
    gl.store(out_ptr + offs, z, mask=mask)


@gluon.jit
def _exp_scaled_identity_kernel(x_ptr, out_ptr, n_elements, MODE: gl.constexpr, BLOCK: gl.constexpr,
                                THREADS_PER_WARP: gl.constexpr):
    pid = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout(size_per_thread=[2], threads_per_warp=[THREADS_PER_WARP], warps_per_cta=[4],
                                            order=[0])
    offs = pid * BLOCK + gl.arange(0, BLOCK, layout=layout)
    mask = offs < n_elements
    x = gl.load(x_ptr + offs, mask=mask, other=0.0)
    if MODE == "exp":
        z = gl.exp(x)
    elif MODE == "exp2_scaled":
        z = gl.exp2(x * 1.44269504)
    else:
        gl.static_assert(False, "unsupported MODE")
    gl.store(out_ptr + offs, z, mask=mask)


@gluon.jit
def _exp_inverse_identity_kernel(x_ptr, out_ptr, n_elements, MODE: gl.constexpr, BLOCK: gl.constexpr,
                                 THREADS_PER_WARP: gl.constexpr):
    pid = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout(size_per_thread=[2], threads_per_warp=[THREADS_PER_WARP], warps_per_cta=[4],
                                            order=[0])
    offs = pid * BLOCK + gl.arange(0, BLOCK, layout=layout)
    mask = offs < n_elements
    x = gl.load(x_ptr + offs, mask=mask, other=0.0)
    if MODE == "exp_neg":
        z = gl.exp(-x)
    elif MODE == "exp_recip":
        z = 1.0 / gl.exp(x)
    else:
        gl.static_assert(False, "unsupported MODE")
    gl.store(out_ptr + offs, z, mask=mask)


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
def test_unary_math_identity(device, op, fresh_knobs):
    _require_cuda_backend(device)

    fresh_knobs.compilation.instrumentation_mode = "fpsan"

    n_elements = 1024
    BLOCK = 256
    rs = np.random.RandomState(0)
    # Includes negative values for log/sqrt on purpose; fpsan works on payload bits.
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
        THREADS_PER_WARP=THREADS_PER_WARP,
    )

    if op == "exp":
        exp_bits = _expected_exp_i32(x_bits)
    elif op == "exp2":
        exp_bits = _expected_exp2_i32(x_bits)
    else:
        exp_bits = _expected_unary_tag_i32(x_bits, op)
    _assert_payload_equal(out, exp_bits)


def test_exp_add_mul_identity(device, fresh_knobs):
    _require_cuda_backend(device)

    fresh_knobs.compilation.instrumentation_mode = "fpsan"

    n_elements = 1024
    BLOCK = 256

    g = torch.Generator(device="cuda")
    g.manual_seed(0)
    x = torch.randint(-(2**31), 2**31 - 1, (n_elements, ), dtype=torch.int32, device="cuda", generator=g)
    y = torch.randint(-(2**31), 2**31 - 1, (n_elements, ), dtype=torch.int32, device="cuda", generator=g)
    out_add = torch.empty((n_elements, ), dtype=torch.int32, device="cuda")
    out_mul = torch.empty((n_elements, ), dtype=torch.int32, device="cuda")

    xw = triton.TensorWrapper(x, dtype=torch.float32)
    yw = triton.TensorWrapper(y, dtype=torch.float32)
    out_add_w = triton.TensorWrapper(out_add, dtype=torch.float32)
    out_mul_w = triton.TensorWrapper(out_mul, dtype=torch.float32)

    grid = (triton.cdiv(n_elements, BLOCK), )
    _exp_binary_identity_kernel[grid](xw, yw, out_add_w, n_elements, MODE="exp_add", BLOCK=BLOCK,
                                      THREADS_PER_WARP=THREADS_PER_WARP)
    _exp_binary_identity_kernel[grid](xw, yw, out_mul_w, n_elements, MODE="exp_mul", BLOCK=BLOCK,
                                      THREADS_PER_WARP=THREADS_PER_WARP)

    _assert_payload_equal(out_add, out_mul)


def test_exp_exp2_scaled_identity(device, fresh_knobs):
    _require_cuda_backend(device)

    fresh_knobs.compilation.instrumentation_mode = "fpsan"

    n_elements = 1024
    BLOCK = 256

    g = torch.Generator(device="cuda")
    g.manual_seed(1)
    x = torch.randint(-(2**31), 2**31 - 1, (n_elements, ), dtype=torch.int32, device="cuda", generator=g)
    out_exp = torch.empty((n_elements, ), dtype=torch.int32, device="cuda")
    out_exp2 = torch.empty((n_elements, ), dtype=torch.int32, device="cuda")

    xw = triton.TensorWrapper(x, dtype=torch.float32)
    out_exp_w = triton.TensorWrapper(out_exp, dtype=torch.float32)
    out_exp2_w = triton.TensorWrapper(out_exp2, dtype=torch.float32)

    grid = (triton.cdiv(n_elements, BLOCK), )
    _exp_scaled_identity_kernel[grid](xw, out_exp_w, n_elements, MODE="exp", BLOCK=BLOCK,
                                      THREADS_PER_WARP=THREADS_PER_WARP)
    _exp_scaled_identity_kernel[grid](xw, out_exp2_w, n_elements, MODE="exp2_scaled", BLOCK=BLOCK,
                                      THREADS_PER_WARP=THREADS_PER_WARP)

    _assert_payload_equal(out_exp, out_exp2)


def test_exp_neg_reciprocal_identity(device, fresh_knobs):
    _require_cuda_backend(device)

    fresh_knobs.compilation.instrumentation_mode = "fpsan"

    n_elements = 1024
    BLOCK = 256

    g = torch.Generator(device="cuda")
    g.manual_seed(4)
    x = torch.randint(-(2**31), 2**31 - 1, (n_elements, ), dtype=torch.int32, device="cuda", generator=g)
    out_neg = torch.empty((n_elements, ), dtype=torch.int32, device="cuda")
    out_recip = torch.empty((n_elements, ), dtype=torch.int32, device="cuda")

    xw = triton.TensorWrapper(x, dtype=torch.float32)
    out_neg_w = triton.TensorWrapper(out_neg, dtype=torch.float32)
    out_recip_w = triton.TensorWrapper(out_recip, dtype=torch.float32)

    grid = (triton.cdiv(n_elements, BLOCK), )
    _exp_inverse_identity_kernel[grid](xw, out_neg_w, n_elements, MODE="exp_neg", BLOCK=BLOCK,
                                       THREADS_PER_WARP=THREADS_PER_WARP)
    _exp_inverse_identity_kernel[grid](xw, out_recip_w, n_elements, MODE="exp_recip", BLOCK=BLOCK,
                                       THREADS_PER_WARP=THREADS_PER_WARP)

    _assert_payload_equal(out_neg, out_recip)


@gluon.jit
def _extern_unary_math_kernel(x_ptr, out_ptr, n_elements, OP: gl.constexpr, BLOCK: gl.constexpr,
                              THREADS_PER_WARP: gl.constexpr):
    pid = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout(size_per_thread=[2], threads_per_warp=[THREADS_PER_WARP], warps_per_cta=[4],
                                            order=[0])
    offs = pid * BLOCK + gl.arange(0, BLOCK, layout=layout)
    mask = offs < n_elements
    x = gl.load(x_ptr + offs, mask=mask, other=0.0)
    if OP == "tan":
        z = gl.extra.libdevice.tan(x)
    elif OP == "tanh":
        z = gl.extra.libdevice.tanh(x)
    elif OP == "log1p":
        z = gl.extra.libdevice.log1p(x)
    elif OP == "cbrt":
        z = gl.extra.libdevice.cbrt(x)
    elif OP == "round":
        z = gl.extra.libdevice.round(x)
    else:
        gl.static_assert(False, "unsupported OP")
    gl.store(out_ptr + offs, z, mask=mask)


@pytest.mark.parametrize(
    "op,symbol",
    EXTERN_UNARY_CASES,
)
def test_extern_unary_payload_semantics(device, op, symbol, fresh_knobs):
    _require_cuda_backend(device)

    fresh_knobs.compilation.instrumentation_mode = "fpsan"

    n_elements = 1024
    BLOCK = 256
    rs = np.random.RandomState(11)
    xf = rs.randn(n_elements).astype(np.float32)
    x_bits = xf.view(np.int32)

    x = torch.tensor(x_bits, dtype=torch.int32, device="cuda")
    out = torch.empty((n_elements, ), dtype=torch.int32, device="cuda")

    grid = (triton.cdiv(n_elements, BLOCK), )
    _extern_unary_math_kernel[grid](
        triton.TensorWrapper(x, dtype=torch.float32),
        triton.TensorWrapper(out, dtype=torch.float32),
        n_elements,
        OP=op,
        BLOCK=BLOCK,
        THREADS_PER_WARP=THREADS_PER_WARP,
    )

    exp_bits = _expected_extern_unary_tag_i32(x_bits, symbol)
    _assert_payload_equal(out, exp_bits)


@gluon.jit
def _extern_binary_math_kernel(x_ptr, y_ptr, out_ptr, n_elements, OP: gl.constexpr, BLOCK: gl.constexpr,
                               THREADS_PER_WARP: gl.constexpr):
    pid = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout(size_per_thread=[2], threads_per_warp=[THREADS_PER_WARP], warps_per_cta=[4],
                                            order=[0])
    offs = pid * BLOCK + gl.arange(0, BLOCK, layout=layout)
    mask = offs < n_elements
    x = gl.load(x_ptr + offs, mask=mask, other=0.0)
    y = gl.load(y_ptr + offs, mask=mask, other=0.0)
    if OP == "atan2":
        z = gl.extra.libdevice.atan2(x, y)
    elif OP == "hypot":
        z = gl.extra.libdevice.hypot(x, y)
    elif OP == "pow":
        z = gl.extra.libdevice.pow(x, y)
    else:
        gl.static_assert(False, "unsupported OP")
    gl.store(out_ptr + offs, z, mask=mask)


@gluon.jit
def _extern_ternary_math_kernel(x_ptr, y_ptr, z_ptr, out_ptr, n_elements, OP: gl.constexpr, BLOCK: gl.constexpr,
                                THREADS_PER_WARP: gl.constexpr):
    pid = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout(size_per_thread=[2], threads_per_warp=[THREADS_PER_WARP], warps_per_cta=[4],
                                            order=[0])
    offs = pid * BLOCK + gl.arange(0, BLOCK, layout=layout)
    mask = offs < n_elements
    x = gl.load(x_ptr + offs, mask=mask, other=0.0)
    y = gl.load(y_ptr + offs, mask=mask, other=0.0)
    z = gl.load(z_ptr + offs, mask=mask, other=0.0)
    if OP == "fma":
        out = gl.extra.libdevice.fma(x, y, z)
    else:
        gl.static_assert(False, "unsupported OP")
    gl.store(out_ptr + offs, out, mask=mask)


@gluon.jit
def _extern_mixed_math_kernel(x_ptr, y_ptr, out_ptr, n_elements, OP: gl.constexpr, BLOCK: gl.constexpr,
                              THREADS_PER_WARP: gl.constexpr):
    pid = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout(size_per_thread=[2], threads_per_warp=[THREADS_PER_WARP], warps_per_cta=[4],
                                            order=[0])
    offs = pid * BLOCK + gl.arange(0, BLOCK, layout=layout)
    mask = offs < n_elements
    x = gl.load(x_ptr + offs, mask=mask, other=0.0)
    y = gl.load(y_ptr + offs, mask=mask, other=0)
    if OP == "ldexp":
        out = gl.extra.libdevice.ldexp(x, y)
    else:
        gl.static_assert(False, "unsupported OP")
    gl.store(out_ptr + offs, out, mask=mask)


@pytest.mark.parametrize(
    "op,symbol",
    EXTERN_BINARY_CASES,
)
def test_extern_binary_payload_semantics(device, op, symbol, fresh_knobs):
    _require_cuda_backend(device)

    fresh_knobs.compilation.instrumentation_mode = "fpsan"

    n_elements = 1024
    BLOCK = 256
    g = torch.Generator(device="cuda")
    g.manual_seed(23)
    x = torch.randint(-(2**31), 2**31 - 1, (n_elements, ), dtype=torch.int32, device="cuda", generator=g)
    y = torch.randint(-(2**31), 2**31 - 1, (n_elements, ), dtype=torch.int32, device="cuda", generator=g)
    out = torch.empty((n_elements, ), dtype=torch.int32, device="cuda")

    xw = triton.TensorWrapper(x, dtype=torch.float32)
    yw = triton.TensorWrapper(y, dtype=torch.float32)
    outw = triton.TensorWrapper(out, dtype=torch.float32)

    grid = (triton.cdiv(n_elements, BLOCK), )
    _extern_binary_math_kernel[grid](xw, yw, outw, n_elements, OP=op, BLOCK=BLOCK, THREADS_PER_WARP=THREADS_PER_WARP)

    exp_bits = _expected_extern_binary_tag_i32(
        x.cpu().numpy().astype(np.int32, copy=False),
        y.cpu().numpy().astype(np.int32, copy=False),
        symbol,
    )
    _assert_payload_equal(out, exp_bits)


@pytest.mark.parametrize(
    "op,symbol",
    EXTERN_TERNARY_CASES,
)
def test_extern_ternary_payload_semantics(device, op, symbol, fresh_knobs):
    _require_cuda_backend(device)

    fresh_knobs.compilation.instrumentation_mode = "fpsan"

    n_elements = 1024
    BLOCK = 256
    g = torch.Generator(device="cuda")
    g.manual_seed(29)
    x = torch.randint(-(2**31), 2**31 - 1, (n_elements, ), dtype=torch.int32, device="cuda", generator=g)
    y = torch.randint(-(2**31), 2**31 - 1, (n_elements, ), dtype=torch.int32, device="cuda", generator=g)
    z = torch.randint(-(2**31), 2**31 - 1, (n_elements, ), dtype=torch.int32, device="cuda", generator=g)
    out = torch.empty((n_elements, ), dtype=torch.int32, device="cuda")

    xw = triton.TensorWrapper(x, dtype=torch.float32)
    yw = triton.TensorWrapper(y, dtype=torch.float32)
    zw = triton.TensorWrapper(z, dtype=torch.float32)
    outw = triton.TensorWrapper(out, dtype=torch.float32)

    grid = (triton.cdiv(n_elements, BLOCK), )
    _extern_ternary_math_kernel[grid](xw, yw, zw, outw, n_elements, OP=op, BLOCK=BLOCK,
                                      THREADS_PER_WARP=THREADS_PER_WARP)

    exp_bits = _expected_extern_variadic_tag_i32(
        [
            x.cpu().numpy().astype(np.int32, copy=False),
            y.cpu().numpy().astype(np.int32, copy=False),
            z.cpu().numpy().astype(np.int32, copy=False),
        ],
        symbol,
    )
    _assert_payload_equal(out, exp_bits)


@pytest.mark.parametrize(
    "op,symbol",
    EXTERN_MIXED_CASES,
)
def test_extern_mixed_payload_semantics(device, op, symbol, fresh_knobs):
    _require_cuda_backend(device)

    fresh_knobs.compilation.instrumentation_mode = "fpsan"

    n_elements = 1024
    BLOCK = 256
    g = torch.Generator(device="cuda")
    g.manual_seed(31)
    x = torch.randint(-(2**31), 2**31 - 1, (n_elements, ), dtype=torch.int32, device="cuda", generator=g)
    y = torch.randint(-(2**31), 2**31 - 1, (n_elements, ), dtype=torch.int32, device="cuda", generator=g)
    out = torch.empty((n_elements, ), dtype=torch.int32, device="cuda")

    xw = triton.TensorWrapper(x, dtype=torch.float32)
    outw = triton.TensorWrapper(out, dtype=torch.float32)

    grid = (triton.cdiv(n_elements, BLOCK), )
    _extern_mixed_math_kernel[grid](xw, y, outw, n_elements, OP=op, BLOCK=BLOCK, THREADS_PER_WARP=THREADS_PER_WARP)

    exp_bits = _expected_extern_variadic_tag_i32(
        [
            x.cpu().numpy().astype(np.int32, copy=False),
            y.cpu().numpy().astype(np.int32, copy=False),
        ],
        symbol,
        float_args=[True, False],
    )
    _assert_payload_equal(out, exp_bits)


def _expected_fma_i32(x_i32: np.ndarray, y_i32: np.ndarray, z_i32: np.ndarray) -> np.ndarray:
    return _expected_add_i32(_expected_mul_i32(x_i32, y_i32), z_i32)


def _expected_trunc_ext_roundtrip_i32(x_i32: np.ndarray) -> np.ndarray:
    x_u32 = _mix_f32_bits_to_payload_u32(x_i32)
    trunc_u16 = _signed_cast_payload_u64(x_u32, 32, 16)
    out_u32 = _signed_cast_payload_u64(trunc_u16, 16, 32).astype(np.uint32)
    return _unmix_payload_u32_to_f32_bits_i32(out_u32)


def _expected_ext_f16_to_f32_i32(x_i16: np.ndarray) -> np.ndarray:
    payload_u16 = _mix_float_bits_to_payload_u64(x_i16.view(np.uint16), 16, 0x3C00)
    out_u32 = _signed_cast_payload_u64(payload_u16, 16, 32).astype(np.uint32)
    return _unmix_payload_u32_to_f32_bits_i32(out_u32)


@gluon.jit
def _fma_kernel(x_ptr, y_ptr, z_ptr, out_ptr, n_elements, BLOCK: gl.constexpr, THREADS_PER_WARP: gl.constexpr):
    pid = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout(size_per_thread=[2], threads_per_warp=[THREADS_PER_WARP], warps_per_cta=[4],
                                            order=[0])
    offs = pid * BLOCK + gl.arange(0, BLOCK, layout=layout)
    mask = offs < n_elements
    x = gl.load(x_ptr + offs, mask=mask, other=0.0)
    y = gl.load(y_ptr + offs, mask=mask, other=0.0)
    z = gl.load(z_ptr + offs, mask=mask, other=0.0)
    out = gl.fma(x, y, z)
    gl.store(out_ptr + offs, out, mask=mask)


def test_fma_payload_semantics(device, fresh_knobs):
    _require_cuda_backend(device)

    fresh_knobs.compilation.instrumentation_mode = "fpsan"

    n_elements = 1024
    BLOCK = 256

    g = torch.Generator(device="cuda")
    g.manual_seed(7)
    x = torch.randint(-(2**31), 2**31 - 1, (n_elements, ), dtype=torch.int32, device="cuda", generator=g)
    y = torch.randint(-(2**31), 2**31 - 1, (n_elements, ), dtype=torch.int32, device="cuda", generator=g)
    z = torch.randint(-(2**31), 2**31 - 1, (n_elements, ), dtype=torch.int32, device="cuda", generator=g)
    out = torch.empty((n_elements, ), dtype=torch.int32, device="cuda")

    xw = triton.TensorWrapper(x, dtype=torch.float32)
    yw = triton.TensorWrapper(y, dtype=torch.float32)
    zw = triton.TensorWrapper(z, dtype=torch.float32)
    outw = triton.TensorWrapper(out, dtype=torch.float32)

    grid = (triton.cdiv(n_elements, BLOCK), )
    _fma_kernel[grid](xw, yw, zw, outw, n_elements, BLOCK=BLOCK, THREADS_PER_WARP=THREADS_PER_WARP)

    out_np = out.cpu().numpy().astype(np.int32, copy=False)
    exp_np = _expected_fma_i32(
        x.cpu().numpy().astype(np.int32, copy=False),
        y.cpu().numpy().astype(np.int32, copy=False),
        z.cpu().numpy().astype(np.int32, copy=False),
    )
    _assert_payload_equal(out_np, exp_np)


@gluon.jit
def _cast_trunc_ext_kernel(x_ptr, out_ptr, n_elements, BLOCK: gl.constexpr, THREADS_PER_WARP: gl.constexpr):
    pid = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout(size_per_thread=[2], threads_per_warp=[THREADS_PER_WARP], warps_per_cta=[4],
                                            order=[0])
    offs = pid * BLOCK + gl.arange(0, BLOCK, layout=layout)
    mask = offs < n_elements
    x = gl.load(x_ptr + offs, mask=mask, other=0.0)
    y = x.to(gl.float16)
    z = y.to(gl.float32)
    gl.store(out_ptr + offs, z, mask=mask)


def test_cast_trunc_ext_payload_semantics(device, fresh_knobs):
    _require_cuda_backend(device)

    fresh_knobs.compilation.instrumentation_mode = "fpsan"

    n_elements = 1024
    BLOCK = 256

    g = torch.Generator(device="cuda")
    g.manual_seed(17)
    x = torch.randint(-(2**31), 2**31 - 1, (n_elements, ), dtype=torch.int32, device="cuda", generator=g)
    special_f32_bits = np.asarray([-1.0, 0.0, 1.0], dtype=np.float32).view(np.int32)
    x[:3] = torch.tensor(special_f32_bits, dtype=torch.int32, device="cuda")
    out = torch.empty((n_elements, ), dtype=torch.int32, device="cuda")

    xw = triton.TensorWrapper(x, dtype=torch.float32)
    outw = triton.TensorWrapper(out, dtype=torch.float32)

    grid = (triton.cdiv(n_elements, BLOCK), )
    _cast_trunc_ext_kernel[grid](xw, outw, n_elements, BLOCK=BLOCK, THREADS_PER_WARP=THREADS_PER_WARP)

    out_np = out.cpu().numpy().astype(np.int32, copy=False)
    exp_np = _expected_trunc_ext_roundtrip_i32(x.cpu().numpy().astype(np.int32, copy=False))
    _assert_payload_equal(out_np, exp_np)
    _assert_payload_equal(out_np[:3], special_f32_bits)


@gluon.jit
def _cast_ext_kernel(x_ptr, out_ptr, n_elements, BLOCK: gl.constexpr, THREADS_PER_WARP: gl.constexpr):
    pid = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout(size_per_thread=[2], threads_per_warp=[THREADS_PER_WARP], warps_per_cta=[4],
                                            order=[0])
    offs = pid * BLOCK + gl.arange(0, BLOCK, layout=layout)
    mask = offs < n_elements
    x = gl.load(x_ptr + offs, mask=mask, other=0.0)
    z = x.to(gl.float32)
    gl.store(out_ptr + offs, z, mask=mask)


def test_cast_ext_payload_semantics(device, fresh_knobs):
    _require_cuda_backend(device)

    fresh_knobs.compilation.instrumentation_mode = "fpsan"

    n_elements = 1024
    BLOCK = 256

    g = torch.Generator(device="cuda")
    g.manual_seed(19)
    x = torch.randint(-(2**15), 2**15 - 1, (n_elements, ), dtype=torch.int16, device="cuda", generator=g)
    special_f16_bits = np.asarray([-1.0, 0.0, 1.0], dtype=np.float16).view(np.int16)
    special_f32_bits = np.asarray([-1.0, 0.0, 1.0], dtype=np.float32).view(np.int32)
    x[:3] = torch.tensor(special_f16_bits, dtype=torch.int16, device="cuda")
    out = torch.empty((n_elements, ), dtype=torch.int32, device="cuda")

    xw = triton.TensorWrapper(x, dtype=torch.float16)
    outw = triton.TensorWrapper(out, dtype=torch.float32)

    grid = (triton.cdiv(n_elements, BLOCK), )
    _cast_ext_kernel[grid](xw, outw, n_elements, BLOCK=BLOCK, THREADS_PER_WARP=THREADS_PER_WARP)

    out_np = out.cpu().numpy().astype(np.int32, copy=False)
    exp_np = _expected_ext_f16_to_f32_i32(x.cpu().numpy().astype(np.int16, copy=False))
    _assert_payload_equal(out_np, exp_np)
    _assert_payload_equal(out_np[:3], special_f32_bits)


def _mm_payload_u32(a_i32: np.ndarray, b_i32: np.ndarray, c_i32: np.ndarray = None) -> np.ndarray:
    # Computes: c + a @ b in Z/(2^32) on mixed f32 payload bits.
    a_u = _mix_f32_bits_to_payload_u32(a_i32).astype(np.uint64)
    b_u = _mix_f32_bits_to_payload_u32(b_i32).astype(np.uint64)
    c_u = _mix_f32_bits_to_payload_u32(c_i32).astype(np.uint64) if c_i32 is not None else None
    m, k = a_u.shape
    k2, n = b_u.shape
    assert k == k2
    out = np.empty((m, n), dtype=np.uint64)
    mask = np.uint64(0xFFFFFFFF)
    for i in range(m):
        for j in range(n):
            s = c_u[i, j] if c_u is not None else 0
            for kk in range(k):
                s = (s + (a_u[i, kk] * b_u[kk, j])) & mask
            out[i, j] = s
    return _unmix_payload_u32_to_f32_bits_i32(out.astype(np.uint32))


def _unpack_element(data: np.ndarray, row: int, col: int, pack: int, pack_axis: int = 1) -> np.uint64:
    if pack_axis == 1:
        raw = np.uint64(data[row, col // pack])
        nibble_idx = col
    else:
        raw = np.uint64(data[row // pack, col])
        nibble_idx = row
    if pack == 2:
        return (raw >> np.uint64(4 * (nibble_idx % pack))) & np.uint64(0x0F)
    return raw


def _mix_float_scalar(val: np.uint64, bitwidth: int, one_bits: int) -> np.uint64:
    mixed = _mix_float_bits_to_payload_u64(np.asarray([val], dtype=np.uint64), bitwidth, one_bits)
    return np.uint64(mixed[0])


def _mix_dot_scaled_elem(val: np.uint64, elem_type: str) -> np.uint64:
    if elem_type in ("e4m3", "e5m2"):
        one_bits = 0x38 if elem_type == "e4m3" else 0x3C
        return _mix_float_scalar(val, 8, one_bits)
    if elem_type == "bf16":
        return _mix_float_scalar(val, 16, 0x3F80)
    return val


def _signed_cast_payload_scalar(payload: np.uint64, src_bitwidth: int, dst_bitwidth: int) -> np.uint64:
    casted = _signed_cast_payload_u64(np.asarray([payload], dtype=np.uint64), src_bitwidth, dst_bitwidth)
    return np.uint64(casted[0])


def _dot_scaled_compute_payload_elem(val: np.uint64, elem_type: str, compute_type: str) -> np.uint64:
    assert compute_type in ("bf16", "fp16")
    compute_width = 16
    if elem_type in ("e4m3", "e5m2"):
        payload = _mix_dot_scaled_elem(val, elem_type)
        return _signed_cast_payload_scalar(payload, 8, compute_width)
    if elem_type == "bf16":
        payload = _mix_float_scalar(val, 16, 0x3F80)
        return _signed_cast_payload_scalar(payload, 16, compute_width)
    if elem_type == "fp16":
        payload = _mix_float_scalar(val, 16, 0x3C00)
        return _signed_cast_payload_scalar(payload, 16, compute_width)

    # Match sanitized fp4_to_fp: unpacked e2m1 bits are zero-extended into the
    # destination floating-point payload.  Float6 formats use the same fallback
    # until the sanitizer has dtype-specific float6 mixing.
    return val & np.uint64(0xFFFF)


def _dot_scaled_scale_payload(raw_scale: np.uint64, compute_type: str) -> np.uint64:
    if compute_type == "bf16":
        raw_bf16 = (raw_scale & np.uint64(0xFF)) << np.uint64(7)
        return _mix_float_scalar(raw_bf16, 16, 0x3F80)
    if compute_type == "fp16":
        raw_f32 = (raw_scale & np.uint64(0xFF)) << np.uint64(23)
        payload_f32 = _mix_float_scalar(raw_f32, 32, 0x3F800000)
        return _signed_cast_payload_scalar(payload_f32, 32, 16)
    raise ValueError(f"unsupported dot_scaled compute type: {compute_type}")


def _dot_scaled_payload_u32(a_data: np.ndarray, b_data: np.ndarray, a_scale, b_scale, a_pack: int, b_pack: int,
                            type_a: str, type_b: str) -> np.ndarray:
    M, N = a_data.shape[0], b_data.shape[1]
    K = a_data.shape[1] * a_pack
    compute_type = "fp16" if "fp16" in (type_a, type_b) else "bf16"
    compute_mask = np.uint64(0xFFFF)
    mask = np.uint64(0xFFFFFFFF)
    out = np.zeros((M, N), dtype=np.uint64)
    for i, j in itertools.product(range(M), range(N)):
        s = np.uint64(0)
        for kk in range(K):
            a_val = _unpack_element(a_data, i, kk, a_pack, pack_axis=1)
            b_val = _unpack_element(b_data, kk, j, b_pack, pack_axis=0)
            a_val = _dot_scaled_compute_payload_elem(a_val, type_a, compute_type)
            b_val = _dot_scaled_compute_payload_elem(b_val, type_b, compute_type)
            if a_scale is not None:
                a_scale_val = _dot_scaled_scale_payload(np.uint64(a_scale[i, kk // 32]), compute_type)
                a_val = (a_val * a_scale_val) & compute_mask
            if b_scale is not None:
                b_scale_val = _dot_scaled_scale_payload(np.uint64(b_scale[j, kk // 32]), compute_type)
                b_val = (b_val * b_scale_val) & compute_mask
            a_val = _signed_cast_payload_scalar(a_val, 16, 32)
            b_val = _signed_cast_payload_scalar(b_val, 16, 32)
            s = (s + a_val * b_val) & mask
        out[i, j] = s
    return _unmix_payload_u32_to_f32_bits_i32(out.astype(np.uint32))


def _mm_scaled_payload_u32(a_u8: np.ndarray, b_u8: np.ndarray, a_scale_u8: np.ndarray, b_scale_u8: np.ndarray,
                           c_i32: np.ndarray = None, a_pack: int = 1, b_pack: int = 1,
                           elem_type: str = "e2m1") -> np.ndarray:
    a_scale = a_scale_u8.astype(np.uint64)
    b_scale = b_scale_u8.astype(np.uint64)
    c_u = _mix_f32_bits_to_payload_u32(c_i32).astype(np.uint64) if c_i32 is not None else None

    m = a_u8.shape[0]
    n = b_u8.shape[1]
    k = a_u8.shape[1] * a_pack
    assert k == b_u8.shape[0] * b_pack
    assert a_scale.shape == (m, k // 32)
    assert b_scale.shape == (n, k // 32)

    def unpack(data: np.ndarray, row: int, col: int, pack: int, pack_axis: int) -> np.uint16:
        if pack == 1:
            return np.uint16(data[row, col])
        return np.uint16(_unpack_element(data, row, col, pack, pack_axis=pack_axis))

    out = np.empty((m, n), dtype=np.uint64)
    compute_type = "bf16"
    compute_mask = np.uint64(0xFFFF)
    mask32 = np.uint64(0xFFFFFFFF)
    for i in range(m):
        for j in range(n):
            s = c_u[i, j] if c_u is not None else 0
            for kk in range(k):
                a_val = unpack(a_u8, i, kk, a_pack, pack_axis=1)
                b_val = unpack(b_u8, kk, j, b_pack, pack_axis=0)
                a_val = _dot_scaled_compute_payload_elem(np.uint64(a_val), elem_type, compute_type)
                b_val = _dot_scaled_compute_payload_elem(np.uint64(b_val), elem_type, compute_type)
                a_scale_val = _dot_scaled_scale_payload(a_scale[i, kk // 32], compute_type)
                b_scale_val = _dot_scaled_scale_payload(b_scale[j, kk // 32], compute_type)
                lhs = (a_val * a_scale_val) & compute_mask
                rhs = (b_val * b_scale_val) & compute_mask
                lhs = _signed_cast_payload_scalar(lhs, 16, 32)
                rhs = _signed_cast_payload_scalar(rhs, 16, 32)
                s = (s + ((np.uint64(lhs) * np.uint64(rhs)) & mask32)) & mask32
            out[i, j] = s
    return _unmix_payload_u32_to_f32_bits_i32(out.astype(np.uint32))


def test_dot_fma(device, fresh_knobs):
    _require_cuda_backend(device)

    B = 16
    BLOCK = gl.constexpr(B)

    fresh_knobs.compilation.instrumentation_mode = "fpsan"

    @gluon.jit
    def kernel(a_ptr, b_ptr, c_ptr, out_ptr, THREADS_PER_WARP: gl.constexpr):
        layout: gl.constexpr = gl.BlockedLayout([1, 1], [THREADS_PER_WARP, 1], [4, 1], [1, 0])
        lhs_layout: gl.constexpr = gl.DotOperandLayout(parent=layout, operand_index=0, k_width=0)
        rhs_layout: gl.constexpr = gl.DotOperandLayout(parent=layout, operand_index=1, k_width=0)

        offs_m = gl.arange(0, BLOCK, layout=gl.SliceLayout(1, layout))[:, None]
        offs_n = gl.arange(0, BLOCK, layout=gl.SliceLayout(0, layout))[None, :]
        # Important: build separate offsets for A and B.
        # dot_fma expects operands to represent A[M,K] and B[K,N]. Using the same
        # linearized (m,n) offsets for both makes B effectively transposed.
        offs_k = gl.arange(0, BLOCK, layout=gl.SliceLayout(0, layout))[None, :]
        a_offs = offs_m * BLOCK + offs_k
        b_offs = offs_n * BLOCK + offs_m  # load B^T so dot_fma produces A @ B
        out_offs = offs_m * BLOCK + offs_n

        a = gl.convert_layout(gl.load(a_ptr + a_offs), lhs_layout)
        b = gl.convert_layout(gl.load(b_ptr + b_offs), rhs_layout)
        c = gl.load(c_ptr + out_offs)
        out = gl.dot_fma(a, b, c)
        gl.store(out_ptr + out_offs, out)

    rs = np.random.RandomState(0)
    a_bits = rs.randint(-(2**31), 2**31 - 1, size=(B, B), dtype=np.int32)
    b_bits = rs.randint(-(2**31), 2**31 - 1, size=(B, B), dtype=np.int32)
    c_bits = rs.randint(-(2**31), 2**31 - 1, size=(B, B), dtype=np.int32)
    exp_bits = _mm_payload_u32(a_bits, b_bits.T, c_bits)

    a = torch.tensor(a_bits, device="cuda", dtype=torch.int32)
    b = torch.tensor(b_bits, device="cuda", dtype=torch.int32)
    c = torch.tensor(c_bits, device="cuda", dtype=torch.int32)
    out = torch.empty((B, B), device="cuda", dtype=torch.int32)

    # Wrap int storage as fp32 so fpsan operates on payload bits.
    aw = triton.TensorWrapper(a, dtype=torch.float32)
    bw = triton.TensorWrapper(b, dtype=torch.float32)
    cw = triton.TensorWrapper(c, dtype=torch.float32)
    outw = triton.TensorWrapper(out, dtype=torch.float32)

    kernel[(1, )](aw, bw, cw, outw, THREADS_PER_WARP=THREADS_PER_WARP)

    _assert_payload_equal(out, exp_bits)


@pytest.mark.skipif(not (is_hip_cdna4() or is_hip_gfx1250()), reason="Requires DotScaledOp support (CDNA4, or GFX1250)")
@pytest.mark.parametrize("type_a", ["e2m1", "e4m3", "e5m2"])
@pytest.mark.parametrize("type_b", ["e2m1", "e4m3", "e5m2", "bf16"])
def test_dot_scaled(device, type_a, type_b, fresh_knobs):
    _require_cuda_backend(device)

    B = 32
    K = 64
    SCALE_K = K // 32

    def allocator(size: int, alignment: int, stream):
        return torch.empty(size, device="cuda", dtype=torch.int32)

    triton.set_allocator(allocator)
    fresh_knobs.compilation.instrumentation_mode = "fpsan"

    @triton.jit
    def kernel(a_ptr, a_scale_ptr, b_ptr, b_scale_ptr, out_ptr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
               BLOCK_K: tl.constexpr, TYPE_A: tl.constexpr, TYPE_B: tl.constexpr):
        DIV_FACTOR_A: tl.constexpr = 2 if TYPE_A == "e2m1" else 1
        DIV_FACTOR_B: tl.constexpr = 2 if TYPE_B == "e2m1" else 1
        PACKED_BLOCK_K_A: tl.constexpr = BLOCK_K // DIV_FACTOR_A
        PACKED_BLOCK_K_B: tl.constexpr = BLOCK_K // DIV_FACTOR_B
        SCALE_BLOCK_K: tl.constexpr = BLOCK_K // 32

        offs_am = tl.arange(0, BLOCK_M)[:, None]
        offs_bn = tl.arange(0, BLOCK_N)[None, :]
        offs_ak = tl.arange(0, PACKED_BLOCK_K_A)[None, :]
        offs_bk = tl.arange(0, PACKED_BLOCK_K_B)[:, None]

        a = tl.load(a_ptr + offs_am * PACKED_BLOCK_K_A + offs_ak)
        b = tl.load(b_ptr + offs_bk * BLOCK_N + offs_bn)

        offs_scale_ak = tl.arange(0, SCALE_BLOCK_K)[None, :]
        offs_scale_bk = tl.arange(0, SCALE_BLOCK_K)[None, :]
        a_scale = tl.load(a_scale_ptr + offs_am * SCALE_BLOCK_K + offs_scale_ak)
        b_scale = tl.load(b_scale_ptr + tl.arange(0, BLOCK_N)[:, None] * SCALE_BLOCK_K + offs_scale_bk)

        c = tl.dot_scaled(a, a_scale, TYPE_A, b, b_scale, TYPE_B)
        tl.store(out_ptr + offs_am * BLOCK_N + offs_bn, c)

    a_pack = 2 if type_a == "e2m1" else 1
    b_pack = 2 if type_b == "e2m1" else 1
    packed_k_a = K // a_pack
    packed_k_b = K // b_pack

    rs = np.random.RandomState(1)
    a_bits = rs.randint(0, 256, size=(B, packed_k_a)).astype(np.uint8)
    b_bits = rs.randint(0, 256, size=(packed_k_b, B)).astype(np.uint8)
    a_scale_bits = rs.randint(0, 255, size=(B, SCALE_K)).astype(np.uint8)
    b_scale_bits = rs.randint(0, 255, size=(B, SCALE_K)).astype(np.uint8)

    a = torch.tensor(a_bits, device="cuda", dtype=torch.uint8)
    b = torch.tensor(b_bits, device="cuda", dtype=torch.uint8)
    a_scale = torch.tensor(a_scale_bits, device="cuda", dtype=torch.uint8)
    b_scale = torch.tensor(b_scale_bits, device="cuda", dtype=torch.uint8)

    if type_b == "bf16":
        b_bits = rs.randint(0, 65536, size=(packed_k_b, B)).astype(np.uint16)
        b = torch.tensor(b_bits, device="cuda", dtype=torch.uint16).view(torch.bfloat16)

    exp_bits = _dot_scaled_payload_u32(a_bits, b_bits, a_scale_bits, None if type_b == "bf16" else b_scale_bits, a_pack,
                                       b_pack, type_a, type_b)

    out = torch.empty((B, B), device="cuda", dtype=torch.int32)
    outw = triton.TensorWrapper(out, dtype=torch.float32)

    kernel[(1, )](a, a_scale, b, b_scale, outw, BLOCK_M=B, BLOCK_N=B, BLOCK_K=K, TYPE_A=type_a, TYPE_B=type_b)

    _assert_payload_equal(out, exp_bits)


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("use_acc", [False, True])
def test_tcgen05_mma(device, use_acc, fresh_knobs):
    _require_cuda_backend(device)

    B = 64
    BLOCK = gl.constexpr(B)

    fresh_knobs.compilation.instrumentation_mode = "fpsan"

    @gluon.jit
    def kernel(a_ptr, b_ptr, c_ptr, out_ptr, USE_ACC: gl.constexpr):
        layout: gl.constexpr = gl.BlockedLayout([1, 1], [32, 1], [gl.num_warps(), 1], [1, 0])

        offs_m = gl.arange(0, BLOCK, layout=gl.SliceLayout(1, layout))[:, None]
        offs_n = gl.arange(0, BLOCK, layout=gl.SliceLayout(0, layout))[None, :]
        offs_k_row = gl.arange(0, BLOCK, layout=gl.SliceLayout(1, layout))[:, None]
        offs_k_col = gl.arange(0, BLOCK, layout=gl.SliceLayout(0, layout))[None, :]

        a_offs = offs_m * BLOCK + offs_k_col
        b_offs = offs_k_row * BLOCK + offs_n
        out_offs = offs_m * BLOCK + offs_n

        a_tile = gl.load(a_ptr + a_offs)
        b_tile = gl.load(b_ptr + b_offs)

        smem_layout_a: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK, BLOCK], gl.float32)
        smem_layout_b: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK, BLOCK], gl.float32)
        smem_a = gl.allocate_shared_memory(gl.float32, [BLOCK, BLOCK], smem_layout_a)
        smem_b = gl.allocate_shared_memory(gl.float32, [BLOCK, BLOCK], smem_layout_b)
        smem_a.store(a_tile)
        smem_b.store(b_tile)

        tmem_layout: gl.constexpr = TensorMemoryLayout((BLOCK, BLOCK), col_stride=1)
        acc_tmem = allocate_tensor_memory(gl.float32, [BLOCK, BLOCK], layout=tmem_layout)
        acc_reg_layout: gl.constexpr = acc_tmem.get_reg_layout()
        if USE_ACC:
            c_tile = gl.load(c_ptr + out_offs)
            acc_init = gl.convert_layout(c_tile, acc_reg_layout)
            acc_tmem.store(acc_init)

        bar = gl.allocate_shared_memory(gl.int64, [1], gl.constexpr(mbarrier.MBarrierLayout()))
        mbarrier.init(bar, count=1)

        smem_b_T = smem_b.permute((1, 0))
        tcgen05_mma(smem_a, smem_b_T, acc_tmem, use_acc=USE_ACC, pred=True, mbarriers=[bar])

        mbarrier.wait(bar, phase=0, deps=[smem_a, smem_b])
        mbarrier.invalidate(bar)

        out = acc_tmem.load()
        out = gl.convert_layout(out, layout)
        gl.store(out_ptr + out_offs, out)

    rs = np.random.RandomState(0)
    a_bits = rs.randint(-(2**31), 2**31 - 1, size=(B, B), dtype=np.int32)
    b_bits = rs.randint(-(2**31), 2**31 - 1, size=(B, B), dtype=np.int32)
    c_bits = rs.randint(-(2**31), 2**31 - 1, size=(B, B), dtype=np.int32)
    exp_bits = _mm_payload_u32(a_bits, b_bits.T, c_bits if use_acc else None)

    a = torch.tensor(a_bits, device="cuda", dtype=torch.int32)
    b = torch.tensor(b_bits, device="cuda", dtype=torch.int32)
    c = torch.tensor(c_bits, device="cuda", dtype=torch.int32)
    out = torch.empty((B, B), device="cuda", dtype=torch.int32)

    aw = triton.TensorWrapper(a, dtype=torch.float32)
    bw = triton.TensorWrapper(b, dtype=torch.float32)
    cw = triton.TensorWrapper(c, dtype=torch.float32)
    outw = triton.TensorWrapper(out, dtype=torch.float32)

    kernel[(1, )](aw, bw, cw, outw, USE_ACC=use_acc)

    _assert_payload_equal(out, exp_bits)


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("elem_type", ["e2m1", "e4m3", "e5m2"])
def test_tcgen05_mma_scaled(device, elem_type, fresh_knobs):
    _require_cuda_backend(device)

    B = 128
    BLOCK = gl.constexpr(B)
    SCALE_K = gl.constexpr(B // 32)

    fresh_knobs.compilation.instrumentation_mode = "fpsan"

    @gluon.jit
    def kernel(a_ptr, b_ptr, a_scale_ptr, b_scale_ptr, c_ptr, out_ptr, TYPE: gl.constexpr):
        layout: gl.constexpr = gl.BlockedLayout([1, 1], [32, 1], [gl.num_warps(), 1], [1, 0])
        IS_FP4: gl.constexpr = TYPE == "e2m1"
        PACK_FACTOR: gl.constexpr = 2 if IS_FP4 else 1
        PACKED_K: gl.constexpr = BLOCK // PACK_FACTOR
        ELEM_DTYPE: gl.constexpr = gl.uint8 if IS_FP4 else (gl.float8e4nv if TYPE == "e4m3" else gl.float8e5)
        a_nvmma_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK, PACKED_K], ELEM_DTYPE)
        b_nvmma_layout: gl.constexpr = (gl.NVMMASharedLayout.get_default_for([BLOCK, PACKED_K], ELEM_DTYPE)
                                        if IS_FP4 else gl.NVMMASharedLayout(swizzle_byte_width=128, transposed=False,
                                                                            element_bitwidth=8, rank=2))
        scale_layout: gl.constexpr = TensorMemoryScalesLayout()

        offs_m = gl.arange(0, BLOCK, layout=gl.SliceLayout(1, layout))[:, None]
        offs_n = gl.arange(0, BLOCK, layout=gl.SliceLayout(0, layout))[None, :]
        offs_k_row = gl.arange(0, PACKED_K, layout=gl.SliceLayout(1, layout))[:, None]
        offs_k_col = gl.arange(0, PACKED_K, layout=gl.SliceLayout(0, layout))[None, :]

        a_tile = gl.load(a_ptr + offs_m * PACKED_K + offs_k_col)
        c_tile = gl.load(c_ptr + offs_m * BLOCK + offs_n)
        a_smem = gl.allocate_shared_memory(ELEM_DTYPE, [BLOCK, PACKED_K], a_nvmma_layout, a_tile)
        if IS_FP4:
            b_tile = gl.load(b_ptr + offs_m * PACKED_K + offs_k_col)
            b_smem = gl.allocate_shared_memory(ELEM_DTYPE, [BLOCK, PACKED_K], b_nvmma_layout, b_tile)
            b_mma = b_smem.permute((1, 0))
        else:
            b_tile = gl.load(b_ptr + offs_k_row * BLOCK + offs_n)
            b_smem = gl.allocate_shared_memory(ELEM_DTYPE, [PACKED_K, BLOCK], b_nvmma_layout, b_tile)
            b_mma = b_smem

        tmem_layout: gl.constexpr = TensorMemoryLayout((BLOCK, BLOCK), col_stride=1)
        acc_tmem = allocate_tensor_memory(gl.float32, [BLOCK, BLOCK], layout=tmem_layout)
        acc_tmem.store(gl.convert_layout(c_tile, acc_tmem.get_reg_layout()))

        a_scale_tmem = allocate_tensor_memory(gl.int8, [BLOCK, SCALE_K], layout=scale_layout)
        b_scale_tmem = allocate_tensor_memory(gl.int8, [BLOCK, SCALE_K], layout=scale_layout)
        a_scale_reg_layout: gl.constexpr = a_scale_tmem.get_reg_layout()
        b_scale_reg_layout: gl.constexpr = b_scale_tmem.get_reg_layout()
        scale_offs_k = gl.arange(0, SCALE_K, layout=gl.SliceLayout(0, a_scale_reg_layout))[None, :]
        scale_offs_m = gl.arange(0, BLOCK, layout=gl.SliceLayout(1, a_scale_reg_layout))[:, None]
        scale_offs_n = gl.arange(0, BLOCK, layout=gl.SliceLayout(1, b_scale_reg_layout))[:, None]
        a_scale_tmem.store(gl.load(a_scale_ptr + scale_offs_m * SCALE_K + scale_offs_k))
        b_scale_tmem.store(gl.load(b_scale_ptr + scale_offs_n * SCALE_K + scale_offs_k))

        bar = gl.allocate_shared_memory(gl.int64, [1], gl.constexpr(mbarrier.MBarrierLayout()))
        mbarrier.init(bar, count=1)
        tcgen05_mma_scaled(a_smem, b_mma, acc_tmem, a_scale_tmem, b_scale_tmem, TYPE, TYPE, use_acc=True)
        tcgen05_commit(bar)
        mbarrier.wait(bar, phase=0)
        mbarrier.invalidate(bar)

        out = gl.convert_layout(acc_tmem.load(), layout)
        gl.store(out_ptr + offs_m * BLOCK + offs_n, out)

    rs = np.random.RandomState(0)
    pack_factor = 2 if elem_type == "e2m1" else 1
    packed_k = B // pack_factor
    a_bits = rs.randint(0 if elem_type == "e2m1" else 20, 256 if elem_type == "e2m1" else 40, size=(B, packed_k),
                        dtype=np.uint8)
    if elem_type == "e2m1":
        b_bits = rs.randint(0, 256, size=(B, packed_k), dtype=np.uint8)
        b_ref_bits = b_bits.T
    else:
        b_bits = rs.randint(20, 40, size=(packed_k, B), dtype=np.uint8)
        b_ref_bits = b_bits
    a_scale_bits = rs.randint(1, 4, size=(B, B // 32), dtype=np.int8)
    b_scale_bits = rs.randint(1, 4, size=(B, B // 32), dtype=np.int8)
    c_bits = rs.randint(-(2**31), 2**31 - 1, size=(B, B), dtype=np.int32)
    exp_bits = _mm_scaled_payload_u32(a_bits, b_ref_bits, a_scale_bits.view(np.uint8), b_scale_bits.view(np.uint8),
                                      c_bits, a_pack=pack_factor, b_pack=pack_factor, elem_type=elem_type)

    if elem_type == "e2m1":
        a = torch.tensor(a_bits, device="cuda", dtype=torch.uint8)
        b = torch.tensor(b_bits, device="cuda", dtype=torch.uint8)
    else:
        torch_dtype = torch.float8_e4m3fn if elem_type == "e4m3" else torch.float8_e5m2
        a = torch.tensor(a_bits, device="cuda", dtype=torch.uint8).view(torch_dtype)
        b = torch.tensor(b_bits, device="cuda", dtype=torch.uint8).view(torch_dtype)
    a_scale = torch.tensor(a_scale_bits, device="cuda", dtype=torch.int8)
    b_scale = torch.tensor(b_scale_bits, device="cuda", dtype=torch.int8)
    c = torch.tensor(c_bits, device="cuda", dtype=torch.int32)
    out = torch.empty((B, B), device="cuda", dtype=torch.int32)

    cw = triton.TensorWrapper(c, dtype=torch.float32)
    outw = triton.TensorWrapper(out, dtype=torch.float32)

    kernel[(1, )](a, b, a_scale, b_scale, cw, outw, TYPE=elem_type)

    _assert_payload_equal(out, exp_bits)


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_tmem_index_subslice(device, fresh_knobs):
    _require_cuda_backend(device)

    B = 64
    BLOCK = gl.constexpr(B)
    SLICE_N = gl.constexpr(32)

    fresh_knobs.compilation.instrumentation_mode = "fpsan"

    @gluon.jit
    def kernel(x_ptr, out_ptr):
        layout: gl.constexpr = gl.BlockedLayout([1, 1], [32, 1], [gl.num_warps(), 1], [1, 0])
        offs_m = gl.arange(0, BLOCK, layout=gl.SliceLayout(1, layout))[:, None]
        offs_n = gl.arange(0, SLICE_N, layout=gl.SliceLayout(0, layout))[None, :]
        offs = offs_m * SLICE_N + offs_n

        x = gl.load(x_ptr + offs)

        tmem_layout: gl.constexpr = TensorMemoryLayout((BLOCK, BLOCK), col_stride=1)
        tmem = allocate_tensor_memory(gl.float32, [2, BLOCK, BLOCK], layout=tmem_layout)
        view = tmem.index(1)
        sub = view.slice(0, SLICE_N)

        sub_reg_layout: gl.constexpr = sub.get_reg_layout()
        x_reg = gl.convert_layout(x, sub_reg_layout)
        sub.store(x_reg)
        out = sub.load()
        out = gl.convert_layout(out, layout)
        gl.store(out_ptr + offs, out)

    rs = np.random.RandomState(0)
    x_bits = rs.randint(-(2**31), 2**31 - 1, size=(B, 32), dtype=np.int32)
    exp_bits = x_bits.copy()

    x = torch.tensor(x_bits, device="cuda", dtype=torch.int32)
    out = torch.empty((B, 32), device="cuda", dtype=torch.int32)

    xw = triton.TensorWrapper(x, dtype=torch.float32)
    outw = triton.TensorWrapper(out, dtype=torch.float32)

    kernel[(1, )](xw, outw)

    _assert_payload_equal(out, exp_bits)


def test_reduction(device, fresh_knobs):
    _require_cuda_backend(device)

    @triton.jit
    def reduce_kernel(a_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, stride_am: tl.constexpr, stride_ak: tl.constexpr,
                      ORDER: tl.constexpr):
        a_ptrs = a_ptr + (tl.arange(0, M)[:, None] * stride_am + (tl.arange(0, N)[None, :]) * stride_ak)
        a = tl.load(a_ptrs)
        r1 = tl.sum(a, axis=ORDER)
        r2 = tl.sum(r1, axis=ORDER - 1)
        tl.store(c_ptr, r2)

    M, N = 512, 512
    torch.manual_seed(0)
    a = torch.randn((M, N), dtype=torch.float32, device="cuda")
    # Make non-associativity visible and deterministic: large + tiny magnitudes.
    a[:, :64] *= 1e10
    a[:, 64:] *= 1e-10
    c1 = torch.empty((1, ), dtype=torch.float32).to('cuda')
    c2 = torch.empty((1, ), dtype=torch.float32).to('cuda')

    reduce_kernel[(1, )](a, c1, M=M, N=N, stride_am=a.stride(0), stride_ak=a.stride(1), ORDER=0)
    reduce_kernel[(1, )](a, c2, M=M, N=N, stride_am=a.stride(0), stride_ak=a.stride(1), ORDER=1)
    assert not _payload_equal(c1, c2)

    fresh_knobs.compilation.instrumentation_mode = "fpsan"

    reduce_kernel[(1, )](a, c1, M=M, N=N, stride_am=a.stride(0), stride_ak=a.stride(1), ORDER=0)
    reduce_kernel[(1, )](a, c2, M=M, N=N, stride_am=a.stride(0), stride_ak=a.stride(1), ORDER=1)
    assert _payload_equal(c1, c2)


def test_reduction_matches_loop(device, fresh_knobs):
    _require_cuda_backend(device)

    @triton.jit
    def reduce_sum_kernel(x_ptr, out_ptr, N: tl.constexpr):
        x = tl.load(x_ptr + tl.arange(0, N))
        tl.store(out_ptr, tl.sum(x, axis=0))

    @triton.jit
    def loop_sum_kernel(x_ptr, out_ptr, N: tl.constexpr):
        acc = tl.full([], 0.0, tl.float32)
        for i in tl.static_range(0, N):
            acc += tl.load(x_ptr + i)
        tl.store(out_ptr, acc)

    N = 256
    pattern = torch.tensor([1e20, 1.0, -1e20, 1.0], dtype=torch.float32, device="cuda")
    x = pattern.repeat(N // pattern.numel())
    reduce_out = torch.empty((1, ), dtype=torch.float32, device="cuda")
    loop_out = torch.empty((1, ), dtype=torch.float32, device="cuda")

    reduce_sum_kernel[(1, )](x, reduce_out, N=N)
    loop_sum_kernel[(1, )](x, loop_out, N=N)
    assert not _payload_equal(reduce_out, loop_out)

    fresh_knobs.compilation.instrumentation_mode = "fpsan"

    reduce_sum_kernel[(1, )](x, reduce_out, N=N)
    loop_sum_kernel[(1, )](x, loop_out, N=N)
    _assert_payload_equal(reduce_out, loop_out)


@pytest.mark.skipif(not (is_hip_cdna3() or is_hip_cdna4()), reason="Requires CDNA3 or CDNA4")
def test_mfma_dot(device, fresh_knobs):
    _require_cuda_backend(device)

    M, N, K = 16, 16, 32

    fresh_knobs.compilation.instrumentation_mode = "fpsan"

    cdna_version = 3 if is_hip_cdna3() else 4
    nonkdim = 32
    kdim = 8 if cdna_version == 3 else 16
    k_width_val = 4 if cdna_version == 3 else 8

    blocked = gl.BlockedLayout([4, 4], [4, 16], [4, 1], [1, 0])
    mfma_layout = gl.amd.AMDMFMALayout(cdna_version, [nonkdim, nonkdim, kdim], True, [4, 1])

    @gluon.jit
    def kernel(a_ptr, b_ptr, c_ptr, out_ptr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr,
               blocked: gl.constexpr, k_width: gl.constexpr, mfma_layout: gl.constexpr):
        dot_a_layout: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfma_layout, k_width=k_width)
        dot_b_layout: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfma_layout, k_width=k_width)

        offs_am = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, blocked))
        offs_bn = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, blocked))
        offs_ak = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(0, blocked))
        offs_bk = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(1, blocked))

        a = gl.load(a_ptr + offs_am[:, None] * BLOCK_K + offs_ak[None, :])
        b = gl.load(b_ptr + offs_bk[:, None] * BLOCK_N + offs_bn[None, :])
        c = gl.load(c_ptr + offs_am[:, None] * BLOCK_N + offs_bn[None, :])

        a1 = gl.convert_layout(a, layout=dot_a_layout)
        b1 = gl.convert_layout(b, layout=dot_b_layout)
        c_acc = gl.convert_layout(c, layout=mfma_layout)

        result = gl.amd.cdna3.mfma(a1, b1, c_acc)
        result = gl.convert_layout(result, layout=blocked)
        gl.store(out_ptr + offs_am[:, None] * BLOCK_N + offs_bn[None, :], result)

    rs = np.random.RandomState(0)
    a_bits = rs.randint(-(2**31), 2**31 - 1, size=(M, K), dtype=np.int32)
    b_bits = rs.randint(-(2**31), 2**31 - 1, size=(K, N), dtype=np.int32)
    c_bits = rs.randint(-(2**31), 2**31 - 1, size=(M, N), dtype=np.int32)
    exp_bits = _mm_payload_u32(a_bits, b_bits, c_bits)

    a = torch.tensor(a_bits, device="cuda", dtype=torch.int32)
    b = torch.tensor(b_bits, device="cuda", dtype=torch.int32)
    c = torch.tensor(c_bits, device="cuda", dtype=torch.int32)
    out = torch.empty((M, N), device="cuda", dtype=torch.int32)

    aw = triton.TensorWrapper(a, dtype=torch.float32)
    bw = triton.TensorWrapper(b, dtype=torch.float32)
    cw = triton.TensorWrapper(c, dtype=torch.float32)
    outw = triton.TensorWrapper(out, dtype=torch.float32)

    kernel[(1, )](aw, bw, cw, outw, BLOCK_M=M, BLOCK_N=N, BLOCK_K=K, blocked=blocked, k_width=k_width_val,
                  mfma_layout=mfma_layout)

    _assert_payload_equal(out, exp_bits)


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250")
def test_wmma_dot(device, fresh_knobs):
    _require_cuda_backend(device)

    B = 32
    fresh_knobs.compilation.instrumentation_mode = "fpsan"

    @gluon.jit
    def kernel(a_ptr, b_ptr, c_ptr, out_ptr, BLOCK: gl.constexpr, INSTR_SHAPE_K: gl.constexpr, K_WIDTH: gl.constexpr):
        blocked: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0])
        wmma: gl.constexpr = gl.amd.AMDWMMALayout(3, True, [[0, 1], [1, 0]], [], [16, 16, INSTR_SHAPE_K])

        offs_m = gl.arange(0, BLOCK, layout=gl.SliceLayout(1, blocked))[:, None]
        offs_k = gl.arange(0, BLOCK, layout=gl.SliceLayout(0, blocked))[None, :]
        offs_bk = gl.arange(0, BLOCK, layout=gl.SliceLayout(1, blocked))[:, None]
        offs_n = gl.arange(0, BLOCK, layout=gl.SliceLayout(0, blocked))[None, :]

        a = gl.load(a_ptr + offs_m * BLOCK + offs_k)
        b = gl.load(b_ptr + offs_bk * BLOCK + offs_n)
        c = gl.load(c_ptr + offs_m * BLOCK + offs_n)
        c = gl.convert_layout(c, wmma)

        a = gl.convert_layout(a, gl.DotOperandLayout(0, wmma, K_WIDTH))
        b = gl.convert_layout(b, gl.DotOperandLayout(1, wmma, K_WIDTH))
        acc = gl.amd.gfx1250.wmma(a, b, c)

        out_layout: gl.constexpr = gl.SliceLayout(1, wmma)
        offs_cm = gl.arange(0, BLOCK, layout=out_layout)[:, None]
        offs_cn = gl.arange(0, BLOCK, layout=gl.SliceLayout(0, wmma))[None, :]
        gl.store(out_ptr + offs_cm * BLOCK + offs_cn, acc)

    rs = np.random.RandomState(0)
    a_bits = rs.randint(-(2**31), 2**31 - 1, size=(B, B), dtype=np.int32)
    b_bits = rs.randint(-(2**31), 2**31 - 1, size=(B, B), dtype=np.int32)
    c_bits = rs.randint(-(2**31), 2**31 - 1, size=(B, B), dtype=np.int32)
    exp_bits = _mm_payload_u32(a_bits, b_bits, c_bits)

    a = torch.tensor(a_bits, device="cuda", dtype=torch.int32)
    b = torch.tensor(b_bits, device="cuda", dtype=torch.int32)
    c = torch.tensor(c_bits, device="cuda", dtype=torch.int32)
    out = torch.empty((B, B), device="cuda", dtype=torch.int32)

    aw = triton.TensorWrapper(a, dtype=torch.float32)
    bw = triton.TensorWrapper(b, dtype=torch.float32)
    cw = triton.TensorWrapper(c, dtype=torch.float32)
    outw = triton.TensorWrapper(out, dtype=torch.float32)

    kernel[(1, )](aw, bw, cw, outw, BLOCK=B, INSTR_SHAPE_K=4, K_WIDTH=2)

    _assert_payload_equal(out, exp_bits)
