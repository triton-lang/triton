# ruff: noqa: F821
import itertools
import numpy as np
import pytest
import torch

import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton import language as tl
from triton._internal_testing import is_blackwell, is_cuda, is_hip, is_hip_cdna3, is_hip_cdna4, is_hip_gfx1250, is_hopper, is_interpreter
from triton._internal_testing import is_blackwell_ultra
from triton.experimental.gluon.language.nvidia.ampere import mma_v2
from triton.experimental.gluon.language.nvidia import hopper
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    TensorMemoryScalesLayout,
    allocate_tensor_memory,
    mbarrier,
    tcgen05_commit,
    tcgen05_copy,
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


_FLOAT_DTYPE_INFO = {
    "f64": (64, 0x3FF0000000000000, np.int64, torch.int64, torch.float64, gl.float64),
    "f32": (32, 0x3F800000, np.int32, torch.int32, torch.float32, gl.float32),
    "f16": (16, 0x3C00, np.int16, torch.int16, torch.float16, gl.float16),
    "bf16": (16, 0x3F80, np.int16, torch.int16, torch.bfloat16, gl.bfloat16),
    "e4m3": (8, 0x38, np.int8, torch.int8, torch.float8_e4m3fn, gl.float8e4nv),
    "e5m2": (8, 0x3C, np.int8, torch.int8, torch.float8_e5m2, gl.float8e5),
    "e4m3fnuz": (8, 0x40, np.int8, torch.int8, torch.float8_e4m3fnuz, gl.float8e4b8),
    "e5m2fnuz": (8, 0x40, np.int8, torch.int8, torch.float8_e5m2fnuz, gl.float8e5b16),
}


def _float_dtype_info(dtype: str):
    return _FLOAT_DTYPE_INFO[dtype]


def _float_payload_edges(bitwidth: int) -> np.ndarray:
    assert bitwidth in (8, 16, 32, 64)
    boundaries = [8]
    if bitwidth > 16:
        boundaries.append(bitwidth // 2)
    edges = [0, 1]
    for boundary in boundaries:
        if boundary < bitwidth:
            edges.extend([(1 << boundary) - 1, 1 << boundary])
    sign = 1 << (bitwidth - 1)
    edges.extend([sign - 1, sign, sign + 1, (1 << bitwidth) - 1])
    return np.asarray(edges, dtype=np.uint64)


def _random_float_bits(rs: np.random.RandomState, shape, dtype: str) -> np.ndarray:
    bitwidth, one_bits, np_storage_dtype, _, _, _ = _float_dtype_info(dtype)
    high = np.iinfo(np.uint64).max if bitwidth == 64 else 1 << bitwidth
    payload = rs.randint(0, high, size=shape, dtype=np.uint64)
    edges = _float_payload_edges(bitwidth)
    edge_count = min(payload.size, len(edges))
    payload.reshape(-1)[:edge_count] = edges[:edge_count]
    bits = _unmix_payload_u64_to_float_bits(payload, bitwidth, one_bits)
    np_unsigned_dtype = np.dtype(f"u{bitwidth // 8}")
    return bits.astype(np_unsigned_dtype).view(np_storage_dtype)


def _as_float_bits_tensor(bits: np.ndarray, dtype: str):
    _, _, _, torch_storage_dtype, torch_dtype, _ = _float_dtype_info(dtype)
    storage = torch.tensor(bits, device="cuda", dtype=torch_storage_dtype)
    return storage, triton.TensorWrapper(storage, dtype=torch_dtype)


def _mix_float_bits(bits: np.ndarray, dtype: str) -> np.ndarray:
    bitwidth, one_bits, _, _, _, _ = _float_dtype_info(dtype)
    return _mix_float_bits_to_payload_u64(bits, bitwidth, one_bits)


def _unmix_payload_to_float_bits(payload: np.ndarray, dtype: str) -> np.ndarray:
    bitwidth, one_bits, np_storage_dtype, _, _, _ = _float_dtype_info(dtype)
    bits = _unmix_payload_u64_to_float_bits(payload, bitwidth, one_bits)
    np_unsigned_dtype = np.dtype(f"u{bitwidth // 8}")
    return bits.astype(np_unsigned_dtype).view(np_storage_dtype)


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


def _expected_neg_i32(x_i32: np.ndarray) -> np.ndarray:
    x_u32 = _mix_f32_bits_to_payload_u32(x_i32).astype(np.uint64)
    return _payload_u32_to_f32_bits_i32(np.uint64(0) - x_u32)


def _expected_mul_i32(x_i32: np.ndarray, y_i32: np.ndarray) -> np.ndarray:
    x_u32 = _mix_f32_bits_to_payload_u32(x_i32).astype(np.uint64)
    y_u32 = _mix_f32_bits_to_payload_u32(y_i32).astype(np.uint64)
    return _payload_u32_to_f32_bits_i32(x_u32 * y_u32)


def _expected_min_i32(x_i32: np.ndarray, y_i32: np.ndarray) -> np.ndarray:
    x = _u32_to_i32(_mix_f32_bits_to_payload_u32(x_i32))
    y = _u32_to_i32(_mix_f32_bits_to_payload_u32(y_i32))
    return _unmix_payload_u32_to_f32_bits_i32(np.minimum(x, y).astype(np.int32).view(np.uint32))


def _expected_max_i32(x_i32: np.ndarray, y_i32: np.ndarray) -> np.ndarray:
    x = _u32_to_i32(_mix_f32_bits_to_payload_u32(x_i32))
    y = _u32_to_i32(_mix_f32_bits_to_payload_u32(y_i32))
    return _unmix_payload_u32_to_f32_bits_i32(np.maximum(x, y).astype(np.int32).view(np.uint32))


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


def _expected_cossin_payload_u32(x_u32: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.uint64(0xFFFFFFFF)
    rcp5 = int(_inv_odd_u64(np.uint64(5)) & mask)
    a = np.uint64((-3 * rcp5) & 0xFFFFFFFF)
    b = np.uint64((4 * rcp5) & 0xFFFFFFFF)
    x = x_u32.astype(np.uint64)
    c = np.ones_like(x, dtype=np.uint64)
    s = np.zeros_like(x, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for i in range(32):
            c_double = (c * c - s * s) & mask
            s_double = (np.uint64(2) * c * s) & mask
            c_inc = (a * c_double - b * s_double) & mask
            s_inc = (a * s_double + b * c_double) & mask
            inc = (x & np.uint64(1 << (31 - i))) != 0
            c = np.where(inc, c_inc, c_double) & mask
            s = np.where(inc, s_inc, s_double) & mask
    return c.astype(np.uint32), s.astype(np.uint32)


def _expected_cos_i32(x_i32: np.ndarray) -> np.ndarray:
    c, _ = _expected_cossin_payload_u32(_mix_f32_bits_to_payload_u32(x_i32))
    return _unmix_payload_u32_to_f32_bits_i32(c)


def _expected_sin_i32(x_i32: np.ndarray) -> np.ndarray:
    _, s = _expected_cossin_payload_u32(_mix_f32_bits_to_payload_u32(x_i32))
    return _unmix_payload_u32_to_f32_bits_i32(s)


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


def _as_payload_np_unsigned(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if not isinstance(x, np.ndarray):
        raise TypeError(f"unsupported input type: {type(x)}")
    if x.dtype.kind in "iuf" and x.dtype.itemsize in (1, 2, 4, 8):
        return x.view(np.dtype(f"u{x.dtype.itemsize}"))
    raise TypeError(f"unsupported dtype for payload comparison: {x.dtype}")


def _assert_payload_equal(actual, expected) -> None:
    np.testing.assert_array_equal(_as_payload_np_unsigned(actual), _as_payload_np_unsigned(expected))


def _payload_equal(a, b) -> bool:
    return np.array_equal(_as_payload_np_unsigned(a), _as_payload_np_unsigned(b))


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
    elif OP == "min":
        z = gl.minimum(x, y)
    elif OP == "max":
        z = gl.maximum(x, y)
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


@gluon.jit
def _expect_zero_upper_triangle_kernel(x_ptr, out_ptr, N: gl.constexpr, THREADS_PER_WARP: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [THREADS_PER_WARP, 1], [4, 1], [1, 0])
    row = gl.arange(0, N, layout=gl.SliceLayout(1, layout))[:, None]
    col = gl.arange(0, N, layout=gl.SliceLayout(0, layout))[None, :]
    upper_triangle = col > row
    x = gl.load(x_ptr + row * N + col)
    x = gl.where(upper_triangle, x - 1.0e30, x)
    y = gl.exp(x)
    y = gl.expect_zero(y, upper_triangle)
    gl.store(out_ptr + row * N + col, y)


def test_expect_zero_upper_triangle_exp(device, fresh_knobs):
    _require_cuda_backend(device)

    N = 32
    torch.manual_seed(0)
    x = torch.randn((N, N), dtype=torch.float32, device="cuda")
    regular_out = torch.empty_like(x)
    fpsan_out = torch.empty_like(x)
    upper_triangle = torch.triu(torch.ones_like(x, dtype=torch.bool), diagonal=1)

    fresh_knobs.compilation.instrumentation_mode = ""
    _expect_zero_upper_triangle_kernel[(1, )](x, regular_out, N=N, THREADS_PER_WARP=THREADS_PER_WARP)

    fresh_knobs.compilation.instrumentation_mode = "fpsan"
    _expect_zero_upper_triangle_kernel[(1, )](x, fpsan_out, N=N, THREADS_PER_WARP=THREADS_PER_WARP)

    assert torch.equal(regular_out[upper_triangle], torch.zeros_like(regular_out[upper_triangle]))
    assert torch.equal(fpsan_out[upper_triangle], torch.zeros_like(fpsan_out[upper_triangle]))
    assert not torch.equal(regular_out, fpsan_out)


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
        ("min", _expected_min_i32),
        ("max", _expected_max_i32),
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
    if OP == "neg":
        z = -x
    else:
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


@gluon.jit
def _cossin_identity_kernel(x_ptr, y_ptr, lhs_ptr, rhs_ptr, n_elements, MODE: gl.constexpr, BLOCK: gl.constexpr,
                            THREADS_PER_WARP: gl.constexpr):
    pid = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout(size_per_thread=[2], threads_per_warp=[THREADS_PER_WARP], warps_per_cta=[4],
                                            order=[0])
    offs = pid * BLOCK + gl.arange(0, BLOCK, layout=layout)
    mask = offs < n_elements
    x = gl.load(x_ptr + offs, mask=mask, other=0.0)
    y = gl.load(y_ptr + offs, mask=mask, other=0.0)
    sx = gl.sin(x)
    sy = gl.sin(y)
    cx = gl.cos(x)
    cy = gl.cos(y)

    if MODE == "sin_add":
        lhs = gl.sin(x + y)
        rhs = sx * cy + cx * sy
    elif MODE == "sin_sub":
        lhs = gl.sin(x - y)
        rhs = sx * cy - cx * sy
    elif MODE == "cos_add":
        lhs = gl.cos(x + y)
        rhs = cx * cy - sx * sy
    elif MODE == "cos_sub":
        lhs = gl.cos(x - y)
        rhs = cx * cy + sx * sy
    elif MODE == "unit":
        lhs = cx * cx + sx * sx
        rhs = x * 0.0 + 1.0
    else:
        gl.static_assert(False, "unsupported MODE")

    gl.store(lhs_ptr + offs, lhs, mask=mask)
    gl.store(rhs_ptr + offs, rhs, mask=mask)


@pytest.mark.parametrize(
    "op",
    [
        "exp",
        "exp2",
        "neg",
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
    elif op == "neg":
        exp_bits = _expected_neg_i32(x_bits)
    elif op == "cos":
        exp_bits = _expected_cos_i32(x_bits)
    elif op == "sin":
        exp_bits = _expected_sin_i32(x_bits)
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


@pytest.mark.parametrize("mode", ["sin_add", "sin_sub", "cos_add", "cos_sub", "unit"])
def test_cossin_angle_identities(device, mode, fresh_knobs):
    _require_cuda_backend(device)

    fresh_knobs.compilation.instrumentation_mode = "fpsan"

    n_elements = 1024
    BLOCK = 256

    g = torch.Generator(device="cuda")
    g.manual_seed(5)
    x = torch.randint(-(2**31), 2**31 - 1, (n_elements, ), dtype=torch.int32, device="cuda", generator=g)
    y = torch.randint(-(2**31), 2**31 - 1, (n_elements, ), dtype=torch.int32, device="cuda", generator=g)
    lhs = torch.empty((n_elements, ), dtype=torch.int32, device="cuda")
    rhs = torch.empty((n_elements, ), dtype=torch.int32, device="cuda")

    xw = triton.TensorWrapper(x, dtype=torch.float32)
    yw = triton.TensorWrapper(y, dtype=torch.float32)
    lhsw = triton.TensorWrapper(lhs, dtype=torch.float32)
    rhsw = triton.TensorWrapper(rhs, dtype=torch.float32)

    grid = (triton.cdiv(n_elements, BLOCK), )
    _cossin_identity_kernel[grid](xw, yw, lhsw, rhsw, n_elements, MODE=mode, BLOCK=BLOCK,
                                  THREADS_PER_WARP=THREADS_PER_WARP)

    _assert_payload_equal(lhs, rhs)


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

    input_bits = [
        x.cpu().numpy().astype(np.int32, copy=False),
        y.cpu().numpy().astype(np.int32, copy=False),
        z.cpu().numpy().astype(np.int32, copy=False),
    ]
    if symbol == "__nv_fmaf":
        exp_bits = _expected_fma_i32(*input_bits)
    else:
        exp_bits = _expected_extern_variadic_tag_i32(input_bits, symbol)
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


def _mm_payload_bits(a_bits: np.ndarray, b_bits: np.ndarray, c_bits: np.ndarray, type_a: str, type_b: str,
                     acc_type: str) -> np.ndarray:
    # Computes: c + a @ b in Z/(2^acc_width) on mixed float payload bits.
    a_width = _float_dtype_info(type_a)[0]
    b_width = _float_dtype_info(type_b)[0]
    acc_width = _float_dtype_info(acc_type)[0]
    a_u = _signed_cast_payload_u64(_mix_float_bits(a_bits, type_a), a_width, acc_width)
    b_u = _signed_cast_payload_u64(_mix_float_bits(b_bits, type_b), b_width, acc_width)
    c_u = _mix_float_bits(c_bits, acc_type) if c_bits is not None else None
    assert a_u.shape[-1] == b_u.shape[-2]
    with np.errstate(over="ignore"):
        out = a_u @ b_u
        if c_u is not None:
            out += c_u
        out &= _low_mask_u64(acc_width)
    return _unmix_payload_to_float_bits(out, acc_type)


def _mm_payload_u32(a_i32: np.ndarray, b_i32: np.ndarray, c_i32: np.ndarray = None) -> np.ndarray:
    return _mm_payload_bits(a_i32, b_i32, c_i32, "f32", "f32", "f32")


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
    # CDNA4 converts raw E8M0 scale bytes to bf16 before scaled-upcast, even
    # when the scaled-upcast result uses fp16.
    scale_compute_type = "bf16" if is_hip_cdna4() else compute_type
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
                a_scale_val = _dot_scaled_scale_payload(np.uint64(a_scale[i, kk // 32]), scale_compute_type)
                a_val = (a_val * a_scale_val) & compute_mask
            if b_scale is not None:
                b_scale_val = _dot_scaled_scale_payload(np.uint64(b_scale[j, kk // 32]), scale_compute_type)
                b_val = (b_val * b_scale_val) & compute_mask
            a_val = _signed_cast_payload_scalar(a_val, 16, 32)
            b_val = _signed_cast_payload_scalar(b_val, 16, 32)
            s = (s + a_val * b_val) & mask
        out[i, j] = s
    return _unmix_payload_u32_to_f32_bits_i32(out.astype(np.uint32))


def _mm_scaled_payload_u32(a_u8: np.ndarray, b_u8: np.ndarray, a_scale_u8: np.ndarray, b_scale_u8: np.ndarray,
                           c_i32: np.ndarray = None, a_pack: int = 1, b_pack: int = 1, type_a: str = "e2m1",
                           type_b: str = "e2m1", scale_factor: int = 32, scale_type: str = "e8m0") -> np.ndarray:
    a_scale = a_scale_u8.astype(np.uint64)
    b_scale = b_scale_u8.astype(np.uint64)
    c_u = _mix_f32_bits_to_payload_u32(c_i32).astype(np.uint64) if c_i32 is not None else None

    m = a_u8.shape[0]
    n = b_u8.shape[1]
    k = a_u8.shape[1] * a_pack
    assert k == b_u8.shape[0] * b_pack
    assert a_scale.shape == (m, k // scale_factor)
    assert b_scale.shape == (n, k // scale_factor)

    def unpack_payload_matrix(data: np.ndarray, pack: int, pack_axis: int) -> np.ndarray:
        if pack == 1:
            return data.astype(np.uint64)
        assert pack == 2
        if pack_axis == 1:
            out = np.empty((data.shape[0], data.shape[1] * pack), dtype=np.uint64)
            out[:, 0::2] = data.astype(np.uint64) & np.uint64(0x0F)
            out[:, 1::2] = (data.astype(np.uint64) >> np.uint64(4)) & np.uint64(0x0F)
            return out
        out = np.empty((data.shape[0] * pack, data.shape[1]), dtype=np.uint64)
        out[0::2, :] = data.astype(np.uint64) & np.uint64(0x0F)
        out[1::2, :] = (data.astype(np.uint64) >> np.uint64(4)) & np.uint64(0x0F)
        return out

    def compute_payload_matrix(data: np.ndarray, elem_type: str) -> np.ndarray:
        if elem_type in ("e4m3", "e5m2"):
            one_bits = 0x38 if elem_type == "e4m3" else 0x3C
            payload = _mix_float_bits_to_payload_u64(data, 8, one_bits)
            return _signed_cast_payload_u64(payload, 8, 16)
        return data & np.uint64(0xFFFF)

    def scale_payload_matrix(raw_scale: np.ndarray) -> np.ndarray:
        if scale_type == "e4m3":
            payload = _mix_float_bits_to_payload_u64(raw_scale, 8, 0x38)
            return _signed_cast_payload_u64(payload, 8, 16)
        assert scale_type == "e8m0"
        raw_bf16 = (raw_scale & np.uint64(0xFF)) << np.uint64(7)
        return _mix_float_bits_to_payload_u64(raw_bf16, 16, 0x3F80)

    a_payload = compute_payload_matrix(unpack_payload_matrix(a_u8, a_pack, pack_axis=1), type_a)
    b_payload = compute_payload_matrix(unpack_payload_matrix(b_u8, b_pack, pack_axis=0), type_b)
    a_scale_payload = scale_payload_matrix(a_scale)
    b_scale_payload = scale_payload_matrix(b_scale)

    out = c_u.copy() if c_u is not None else np.zeros((m, n), dtype=np.uint64)
    compute_mask = np.uint64(0xFFFF)
    mask32 = np.uint64(0xFFFFFFFF)
    for group in range(k // scale_factor):
        start = group * scale_factor
        end = start + scale_factor
        lhs = (a_payload[:, start:end] * a_scale_payload[:, group:group + 1]) & compute_mask
        rhs = (b_payload[start:end, :] * b_scale_payload[:, group][None, :]) & compute_mask
        lhs = _signed_cast_payload_u64(lhs, 16, 32)
        rhs = _signed_cast_payload_u64(rhs, 16, 32)
        out = (out + (lhs @ rhs)) & mask32
    return _unmix_payload_u32_to_f32_bits_i32(out.astype(np.uint32))


_DOT_FLOAT_DTYPES = [
    ("f32", "f32", "f32"),
    ("bf16", "bf16", "f32"),
    ("f16", "f16", "f16"),
    ("f16", "f16", "f32"),
    *[(type_a, type_b, acc_type)
      for type_a, type_b, acc_type in itertools.product(("e4m3", "e5m2"), ("e4m3", "e5m2"), ("f16", "f32"))],
]

_DOT_FMA_DTYPES = [
    *_DOT_FLOAT_DTYPES,
    ("f64", "f64", "f64"),
]

_TCGEN05_FLOAT_DTYPES = [
    *_DOT_FLOAT_DTYPES,
    ("f16", "bf16", "f32"),
    ("bf16", "f16", "f32"),
]

_TCGEN05_SCALED_DTYPES = list(itertools.product(("e2m1", "e4m3", "e5m2"), repeat=2))

_MFMA_FP8_DTYPES = ("e4m3fnuz", "e5m2fnuz") if is_hip_cdna3() else ("e4m3", "e5m2")

_MFMA_DOT_CASES = [
    pytest.param("f32", "f32", "f32", 16, 16, 32, 32, 32, 8 if is_hip_cdna3() else 16, 4 if is_hip_cdna3() else 8,
                 id="f32-f32-f32-broad"),
    pytest.param("f64", "f64", "f64", 16, 16, 4, 16, 16, 4, 1, id="f64-f64-f64-minimum"),
    pytest.param("f32", "f32", "f32", 16, 16, 4, 16, 16, 4, 1, id="f32-f32-f32-minimum"),
    pytest.param("f16", "f16", "f32", 16, 16, 16, 16, 16, 16, 4, id="f16-f16-f32-minimum"),
    pytest.param("bf16", "bf16", "f32", 16, 16, 16, 16, 16, 16, 4, id="bf16-bf16-f32-minimum"),
    *[
        pytest.param(type_a, type_b, "f32", 16, 16, 32, 16, 16, 32, 8, id=f"{type_a}-{type_b}-f32-minimum")
        for type_a, type_b in itertools.product(_MFMA_FP8_DTYPES, repeat=2)
    ],
]

_WMMA_DOT_CASES = [
    pytest.param("f32", "f32", "f32", 32, 32, 32, 4, 2, id="f32-f32-f32-broad"),
    pytest.param("f32", "f32", "f32", 16, 16, 4, 4, 2, id="f32-f32-f32-minimum"),
    pytest.param("f16", "f16", "f32", 16, 16, 32, 32, 8, id="f16-f16-f32-minimum"),
    pytest.param("bf16", "bf16", "f32", 16, 16, 32, 32, 8, id="bf16-bf16-f32-minimum"),
    *[
        pytest.param(type_a, type_b, "f32", 16, 16, 64, 64, 8, id=f"{type_a}-{type_b}-f32-minimum")
        for type_a, type_b in itertools.product(("e4m3", "e5m2"), repeat=2)
    ],
]


def _native_mma_k(type_a: str) -> int:
    return 256 // _float_dtype_info(type_a)[0]


_DOT_FMA_CASES = [
    *[pytest.param(*dtypes, 32, 32, 32, id=f"{'-'.join(dtypes)}-broad") for dtypes in _DOT_FMA_DTYPES],
    *[
        pytest.param(*dtypes, 1, 1, _native_mma_k(dtypes[0]), id=f"{'-'.join(dtypes)}-minimum")
        for dtypes in _DOT_FMA_DTYPES
    ],
]

_MMA_V2_CASES = [
    pytest.param(*dtypes, 8 if dtypes[0] == "f64" else 16, 8, _native_mma_k(dtypes[0]), 8 if dtypes[0] == "f64" else 16,
                 id="-".join(dtypes)) for dtypes in _DOT_FMA_DTYPES
]

_WARP_GROUP_MMA_CASES = [
    *[pytest.param(*dtypes, 64, 64, 64, 32, id=f"{'-'.join(dtypes)}-broad") for dtypes in _DOT_FLOAT_DTYPES],
    *[
        pytest.param(*dtypes, 64, 8, _native_mma_k(dtypes[0]), 8, id=f"{'-'.join(dtypes)}-minimum")
        for dtypes in _DOT_FLOAT_DTYPES
    ],
]

_TCGEN05_MMA_CASES = [
    *[pytest.param(*dtypes, 64, 64, 64, id=f"{'-'.join(dtypes)}-broad") for dtypes in _TCGEN05_FLOAT_DTYPES],
    *[
        pytest.param(*dtypes, 64, 8, _native_mma_k(dtypes[0]), id=f"{'-'.join(dtypes)}-minimum")
        for dtypes in _TCGEN05_FLOAT_DTYPES
    ],
]

_TCGEN05_MMA_SCALED_CASES = [
    *[
        pytest.param(type_a, type_b, 128, 128, 128, 32, "e8m0", id=f"{type_a}-{type_b}-broad")
        for type_a, type_b in _TCGEN05_SCALED_DTYPES
    ],
    *[
        pytest.param(type_a, type_b, 128, 128, 64 if type_a == type_b == "e2m1" else 32, 32, "e8m0",
                     id=f"{type_a}-{type_b}-mxfp-minimum") for type_a, type_b in _TCGEN05_SCALED_DTYPES
    ],
    pytest.param("e2m1", "e2m1", 128, 128, 64, 16, "e4m3", id="e2m1-e2m1-nvfp4-minimum"),
]


@pytest.mark.parametrize(("type_a", "type_b", "acc_type", "m", "n", "k"), _DOT_FMA_CASES)
def test_dot_fma(device, type_a, type_b, acc_type, m, n, k, fresh_knobs):
    _require_cuda_backend(device)
    if is_cuda() and torch.cuda.get_device_capability()[0] < 9 and "e4m3" in (type_a, type_b):
        pytest.skip("E4M3 requires Hopper or newer")

    M = gl.constexpr(m)
    N = gl.constexpr(n)
    K = gl.constexpr(k)

    fresh_knobs.compilation.instrumentation_mode = "fpsan"

    @gluon.jit
    def kernel(a_ptr, b_ptr, c_ptr, out_ptr, THREADS_PER_WARP: gl.constexpr):
        layout: gl.constexpr = gl.BlockedLayout([1, 1], [THREADS_PER_WARP, 1], [4, 1], [1, 0])
        lhs_layout: gl.constexpr = gl.DotOperandLayout(parent=layout, operand_index=0, k_width=0)
        rhs_layout: gl.constexpr = gl.DotOperandLayout(parent=layout, operand_index=1, k_width=0)

        offs_m = gl.arange(0, M, layout=gl.SliceLayout(1, layout))[:, None]
        offs_n = gl.arange(0, N, layout=gl.SliceLayout(0, layout))[None, :]
        offs_k_row = gl.arange(0, K, layout=gl.SliceLayout(1, layout))[:, None]
        offs_k_col = gl.arange(0, K, layout=gl.SliceLayout(0, layout))[None, :]
        a_offs = offs_m * K + offs_k_col
        b_offs = offs_n * K + offs_k_row
        out_offs = offs_m * N + offs_n

        a = gl.convert_layout(gl.load(a_ptr + a_offs), lhs_layout)
        b = gl.convert_layout(gl.load(b_ptr + b_offs), rhs_layout)
        c = gl.load(c_ptr + out_offs)
        out = gl.dot_fma(a, b, c)
        gl.store(out_ptr + out_offs, out)

    rs = np.random.RandomState(0)
    a_bits = _random_float_bits(rs, (m, k), type_a)
    b_bits = _random_float_bits(rs, (n, k), type_b)
    c_bits = _random_float_bits(rs, (m, n), acc_type)
    exp_bits = _mm_payload_bits(a_bits, b_bits.T, c_bits, type_a, type_b, acc_type)

    _, aw = _as_float_bits_tensor(a_bits, type_a)
    _, bw = _as_float_bits_tensor(b_bits, type_b)
    _, cw = _as_float_bits_tensor(c_bits, acc_type)
    out, outw = _as_float_bits_tensor(np.empty((m, n), dtype=_float_dtype_info(acc_type)[2]), acc_type)

    compiled = kernel[(1, )](aw, bw, cw, outw, THREADS_PER_WARP=THREADS_PER_WARP)
    ttgir = compiled.asm["ttgir"]
    assert "ttng.tc_gen5_mma" not in ttgir
    assert "ttng.warp_group_dot" not in ttgir

    _assert_payload_equal(out, exp_bits)


@pytest.mark.skipif(not is_cuda(), reason="Requires NVIDIA MMA v2")
@pytest.mark.parametrize(("type_a", "type_b", "acc_type", "m", "n", "k", "instr_m"), _MMA_V2_CASES)
def test_mma_v2(device, type_a, type_b, acc_type, m, n, k, instr_m, fresh_knobs):
    _require_cuda_backend(device)
    if torch.cuda.get_device_capability()[0] < 9 and "e4m3" in (type_a, type_b):
        pytest.skip("E4M3 requires Hopper or newer")

    M = gl.constexpr(m)
    N = gl.constexpr(n)
    K = gl.constexpr(k)

    fresh_knobs.compilation.instrumentation_mode = "fpsan"

    @gluon.jit
    def kernel(a_ptr, b_ptr, c_ptr, out_ptr, A_K_WIDTH: gl.constexpr, B_K_WIDTH: gl.constexpr, INSTR_M: gl.constexpr,
               PRECISION: gl.constexpr, THREADS_PER_WARP: gl.constexpr):
        layout: gl.constexpr = gl.BlockedLayout([1, 1], [THREADS_PER_WARP, 1], [4, 1], [1, 0])
        acc_layout: gl.constexpr = gl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[4, 1],
                                                             instr_shape=[INSTR_M, 8])
        lhs_layout: gl.constexpr = gl.DotOperandLayout(parent=acc_layout, operand_index=0, k_width=A_K_WIDTH)
        rhs_layout: gl.constexpr = gl.DotOperandLayout(parent=acc_layout, operand_index=1, k_width=B_K_WIDTH)

        offs_m = gl.arange(0, M, layout=gl.SliceLayout(1, layout))[:, None]
        offs_n = gl.arange(0, N, layout=gl.SliceLayout(0, layout))[None, :]
        offs_k_row = gl.arange(0, K, layout=gl.SliceLayout(1, layout))[:, None]
        offs_k_col = gl.arange(0, K, layout=gl.SliceLayout(0, layout))[None, :]
        a_offs = offs_m * K + offs_k_col
        b_offs = offs_k_row * N + offs_n
        out_offs = offs_m * N + offs_n

        a = gl.convert_layout(gl.load(a_ptr + a_offs), lhs_layout)
        b = gl.convert_layout(gl.load(b_ptr + b_offs), rhs_layout)
        c = gl.convert_layout(gl.load(c_ptr + out_offs), acc_layout)
        out = mma_v2(a, b, c, input_precision=PRECISION)
        gl.store(out_ptr + out_offs, gl.convert_layout(out, layout))

    rs = np.random.RandomState(0)
    a_bits = _random_float_bits(rs, (m, k), type_a)
    b_bits = _random_float_bits(rs, (k, n), type_b)
    c_bits = _random_float_bits(rs, (m, n), acc_type)
    exp_bits = _mm_payload_bits(a_bits, b_bits, c_bits, type_a, type_b, acc_type)

    _, aw = _as_float_bits_tensor(a_bits, type_a)
    _, bw = _as_float_bits_tensor(b_bits, type_b)
    _, cw = _as_float_bits_tensor(c_bits, acc_type)
    out, outw = _as_float_bits_tensor(np.empty((m, n), dtype=_float_dtype_info(acc_type)[2]), acc_type)

    a_width = _float_dtype_info(type_a)[0]
    b_width = _float_dtype_info(type_b)[0]
    precision = "tf32" if type_a == "f32" else "ieee"
    kernel[(1, )](aw, bw, cw, outw, A_K_WIDTH=max(32 // a_width, 1), B_K_WIDTH=max(32 // b_width, 1), INSTR_M=instr_m,
                  PRECISION=precision, THREADS_PER_WARP=THREADS_PER_WARP)

    _assert_payload_equal(out, exp_bits)


def test_dot_fma_batched(device, fresh_knobs):
    _require_cuda_backend(device)

    BATCH_SIZE = 2
    B = 16
    BATCH = gl.constexpr(BATCH_SIZE)
    BLOCK = gl.constexpr(B)

    fresh_knobs.compilation.instrumentation_mode = "fpsan"

    @gluon.jit
    def kernel(a_ptr, b_ptr, c_ptr, out_ptr, THREADS_PER_WARP: gl.constexpr):
        layout: gl.constexpr = gl.BlockedLayout([1, 1, 1], [1, THREADS_PER_WARP, 1], [1, 4, 1], [2, 1, 0])
        lhs_layout: gl.constexpr = gl.DotOperandLayout(parent=layout, operand_index=0, k_width=0)
        rhs_layout: gl.constexpr = gl.DotOperandLayout(parent=layout, operand_index=1, k_width=0)

        offs_batch = gl.arange(0, BATCH, layout=gl.SliceLayout(1, parent=gl.SliceLayout(2, layout)))[:, None, None]
        offs_m = gl.arange(0, BLOCK, layout=gl.SliceLayout(0, parent=gl.SliceLayout(2, layout)))[None, :, None]
        offs_n = gl.arange(0, BLOCK, layout=gl.SliceLayout(0, parent=gl.SliceLayout(1, layout)))[None, None, :]
        offs_k_a = gl.arange(0, BLOCK, layout=gl.SliceLayout(0, parent=gl.SliceLayout(1, layout)))[None, None, :]
        offs_k_b = gl.arange(0, BLOCK, layout=gl.SliceLayout(0, parent=gl.SliceLayout(2, layout)))[None, :, None]

        a_offs = offs_batch * BLOCK * BLOCK + offs_m * BLOCK + offs_k_a
        b_offs = offs_batch * BLOCK * BLOCK + offs_k_b * BLOCK + offs_n
        out_offs = offs_batch * BLOCK * BLOCK + offs_m * BLOCK + offs_n

        a = gl.convert_layout(gl.load(a_ptr + a_offs), lhs_layout)
        b = gl.convert_layout(gl.load(b_ptr + b_offs), rhs_layout)
        c = gl.load(c_ptr + out_offs)
        out = gl.dot_fma(a, b, c)
        gl.store(out_ptr + out_offs, out)

    rs = np.random.RandomState(1)
    a_bits = rs.randint(-(2**31), 2**31 - 1, size=(BATCH_SIZE, B, B), dtype=np.int32)
    b_bits = rs.randint(-(2**31), 2**31 - 1, size=(BATCH_SIZE, B, B), dtype=np.int32)
    c_bits = rs.randint(-(2**31), 2**31 - 1, size=(BATCH_SIZE, B, B), dtype=np.int32)
    exp_bits = _mm_payload_u32(a_bits, b_bits, c_bits)

    a = torch.tensor(a_bits, device="cuda", dtype=torch.int32)
    b = torch.tensor(b_bits, device="cuda", dtype=torch.int32)
    c = torch.tensor(c_bits, device="cuda", dtype=torch.int32)
    out = torch.empty((BATCH_SIZE, B, B), device="cuda", dtype=torch.int32)

    aw = triton.TensorWrapper(a, dtype=torch.float32)
    bw = triton.TensorWrapper(b, dtype=torch.float32)
    cw = triton.TensorWrapper(c, dtype=torch.float32)
    outw = triton.TensorWrapper(out, dtype=torch.float32)

    compiled = kernel[(1, )](aw, bw, cw, outw, THREADS_PER_WARP=THREADS_PER_WARP)
    ttgir = compiled.asm["ttgir"]
    assert "ttng.tc_gen5_mma" not in ttgir
    assert "ttng.warp_group_dot" not in ttgir

    _assert_payload_equal(out, exp_bits)


@pytest.mark.skipif(not is_hopper(), reason="Requires Hopper")
@pytest.mark.parametrize(("use_acc", "is_async"), [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize(("type_a", "type_b", "acc_type", "m", "n", "k", "instr_n"), _WARP_GROUP_MMA_CASES)
def test_warpgroup_mma(device, use_acc, is_async, type_a, type_b, acc_type, m, n, k, instr_n, fresh_knobs):
    _require_cuda_backend(device)

    M = gl.constexpr(m)
    N = gl.constexpr(n)
    K = gl.constexpr(k)

    fresh_knobs.compilation.instrumentation_mode = "fpsan"

    @gluon.jit
    def kernel(a_ptr, b_ptr, c_ptr, out_ptr, USE_ACC: gl.constexpr, IS_ASYNC: gl.constexpr, A_DTYPE: gl.constexpr,
               B_DTYPE: gl.constexpr, INSTR_N: gl.constexpr, INSTR_K: gl.constexpr, PRECISION: gl.constexpr):
        layout: gl.constexpr = gl.BlockedLayout([1, 1], [32, 1], [gl.num_warps(), 1], [1, 0])
        acc_layout: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1],
                                                             instr_shape=[16, INSTR_N, INSTR_K])

        offs_m = gl.arange(0, M, layout=gl.SliceLayout(1, layout))[:, None]
        offs_n = gl.arange(0, N, layout=gl.SliceLayout(0, layout))[None, :]
        offs_k_row = gl.arange(0, K, layout=gl.SliceLayout(1, layout))[:, None]
        offs_k_col = gl.arange(0, K, layout=gl.SliceLayout(0, layout))[None, :]

        a_tile = gl.load(a_ptr + offs_m * K + offs_k_col)
        b_tile = gl.load(b_ptr + offs_k_row * N + offs_n)
        c_tile = gl.load(c_ptr + offs_m * N + offs_n)

        smem_layout_a: gl.constexpr = gl.NVMMASharedLayout.get_default_for([M, K], A_DTYPE)
        smem_layout_b: gl.constexpr = gl.NVMMASharedLayout.get_default_for([K, N], B_DTYPE)
        smem_a = gl.allocate_shared_memory(A_DTYPE, [M, K], smem_layout_a, a_tile)
        smem_b = gl.allocate_shared_memory(B_DTYPE, [K, N], smem_layout_b, b_tile)

        acc = gl.convert_layout(c_tile, acc_layout)
        acc = hopper.warpgroup_mma(smem_a, smem_b, acc, use_acc=USE_ACC, precision=PRECISION, is_async=IS_ASYNC)
        if IS_ASYNC:
            acc = hopper.warpgroup_mma_wait(num_outstanding=0, deps=[acc])
        out = gl.convert_layout(acc, layout)
        gl.store(out_ptr + offs_m * N + offs_n, out)

    rs = np.random.RandomState(0)
    a_bits = _random_float_bits(rs, (m, k), type_a)
    b_bits = _random_float_bits(rs, (k, n), type_b)
    c_bits = _random_float_bits(rs, (m, n), acc_type)
    exp_bits = _mm_payload_bits(a_bits, b_bits, c_bits if use_acc else None, type_a, type_b, acc_type)

    _, aw = _as_float_bits_tensor(a_bits, type_a)
    _, bw = _as_float_bits_tensor(b_bits, type_b)
    _, cw = _as_float_bits_tensor(c_bits, acc_type)
    out, outw = _as_float_bits_tensor(np.empty((m, n), dtype=_float_dtype_info(acc_type)[2]), acc_type)

    a_width, _, _, _, _, a_dtype = _float_dtype_info(type_a)
    _, _, _, _, _, b_dtype = _float_dtype_info(type_b)
    precision = "tf32" if type_a == "f32" else "ieee"
    kernel[(1, )](aw, bw, cw, outw, USE_ACC=use_acc, IS_ASYNC=is_async, A_DTYPE=a_dtype, B_DTYPE=b_dtype,
                  INSTR_N=instr_n, INSTR_K=256 // a_width, PRECISION=precision)

    _assert_payload_equal(out, exp_bits)


@pytest.mark.skipif(not (is_hip_cdna4() or is_hip_gfx1250()), reason="Requires DotScaledOp support (CDNA4, or GFX1250)")
@pytest.mark.parametrize("type_a", ["e2m1", "e4m3", "e5m2", "bf16", "fp16"])
@pytest.mark.parametrize("type_b", ["e2m1", "e4m3", "e5m2", "bf16", "fp16"])
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

    if type_a in ("bf16", "fp16"):
        a_bits = rs.randint(0, 65536, size=(B, packed_k_a)).astype(np.uint16)
        a = torch.tensor(a_bits, device="cuda",
                         dtype=torch.uint16).view(torch.bfloat16 if type_a == "bf16" else torch.float16)
    if type_b in ("bf16", "fp16"):
        b_bits = rs.randint(0, 65536, size=(packed_k_b, B)).astype(np.uint16)
        b = torch.tensor(b_bits, device="cuda",
                         dtype=torch.uint16).view(torch.bfloat16 if type_b == "bf16" else torch.float16)

    exp_bits = _dot_scaled_payload_u32(a_bits, b_bits, None if type_a in ("bf16", "fp16") else a_scale_bits,
                                       None if type_b in ("bf16", "fp16") else b_scale_bits, a_pack, b_pack, type_a,
                                       type_b)

    out = torch.empty((B, B), device="cuda", dtype=torch.int32)
    outw = triton.TensorWrapper(out, dtype=torch.float32)

    kernel[(1, )](a, a_scale, b, b_scale, outw, BLOCK_M=B, BLOCK_N=B, BLOCK_K=K, TYPE_A=type_a, TYPE_B=type_b)

    _assert_payload_equal(out, exp_bits)


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("use_acc", [False, True])
@pytest.mark.parametrize(("type_a", "type_b", "acc_type", "m", "n", "k"), _TCGEN05_MMA_CASES)
def test_tcgen05_mma(device, use_acc, type_a, type_b, acc_type, m, n, k, fresh_knobs):
    _require_cuda_backend(device)

    M = gl.constexpr(m)
    N = gl.constexpr(n)
    K = gl.constexpr(k)

    fresh_knobs.compilation.instrumentation_mode = "fpsan"

    @gluon.jit
    def kernel(a_ptr, b_ptr, c_ptr, out_ptr, USE_ACC: gl.constexpr, A_DTYPE: gl.constexpr, B_DTYPE: gl.constexpr,
               ACC_DTYPE: gl.constexpr, ACC_BITWIDTH: gl.constexpr):
        layout: gl.constexpr = gl.BlockedLayout([1, 1], [32, 1], [gl.num_warps(), 1], [1, 0])

        offs_m = gl.arange(0, M, layout=gl.SliceLayout(1, layout))[:, None]
        offs_n = gl.arange(0, N, layout=gl.SliceLayout(0, layout))[None, :]
        offs_n_row = gl.arange(0, N, layout=gl.SliceLayout(1, layout))[:, None]
        offs_k_col = gl.arange(0, K, layout=gl.SliceLayout(0, layout))[None, :]

        a_offs = offs_m * K + offs_k_col
        b_offs = offs_n_row * K + offs_k_col
        out_offs = offs_m * N + offs_n

        a_tile = gl.load(a_ptr + a_offs)
        b_tile = gl.load(b_ptr + b_offs)

        smem_layout_a: gl.constexpr = gl.NVMMASharedLayout.get_default_for([M, K], A_DTYPE)
        smem_layout_b: gl.constexpr = gl.NVMMASharedLayout.get_default_for([N, K], B_DTYPE)
        smem_a = gl.allocate_shared_memory(A_DTYPE, [M, K], smem_layout_a)
        smem_b = gl.allocate_shared_memory(B_DTYPE, [N, K], smem_layout_b)
        smem_a.store(a_tile)
        smem_b.store(b_tile)

        tmem_layout: gl.constexpr = TensorMemoryLayout((M, N), col_stride=32 // ACC_BITWIDTH)
        acc_tmem = allocate_tensor_memory(ACC_DTYPE, [M, N], layout=tmem_layout)
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
    a_bits = _random_float_bits(rs, (m, k), type_a)
    b_bits = _random_float_bits(rs, (n, k), type_b)
    c_bits = _random_float_bits(rs, (m, n), acc_type)
    exp_bits = _mm_payload_bits(a_bits, b_bits.T, c_bits if use_acc else None, type_a, type_b, acc_type)

    _, aw = _as_float_bits_tensor(a_bits, type_a)
    _, bw = _as_float_bits_tensor(b_bits, type_b)
    _, cw = _as_float_bits_tensor(c_bits, acc_type)
    out, outw = _as_float_bits_tensor(np.empty((m, n), dtype=_float_dtype_info(acc_type)[2]), acc_type)

    a_dtype = _float_dtype_info(type_a)[5]
    b_dtype = _float_dtype_info(type_b)[5]
    acc_bitwidth, _, _, _, _, acc_dtype = _float_dtype_info(acc_type)
    kernel[(1, )](aw, bw, cw, outw, USE_ACC=use_acc, A_DTYPE=a_dtype, B_DTYPE=b_dtype, ACC_DTYPE=acc_dtype,
                  ACC_BITWIDTH=acc_bitwidth)

    _assert_payload_equal(out, exp_bits)


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("partition_warps", [4, 2, 1])
def test_tcgen05_mma_warp_specialize_partition(device, partition_warps, fresh_knobs):
    _require_cuda_backend(device)

    M = gl.constexpr(64)
    N = gl.constexpr(32)
    K = gl.constexpr(32)
    PARTITION_WARPS = gl.constexpr(partition_warps)
    fresh_knobs.compilation.instrumentation_mode = "fpsan"

    @gluon.jit
    def default_partition():
        pass

    @gluon.jit
    def mma_partition(smem_a, smem_b, acc_tmem, bar):
        tcgen05_mma(smem_a, smem_b.permute((1, 0)), acc_tmem, use_acc=False, pred=True, mbarriers=[bar])

    @gluon.jit
    def kernel(a_ptr, b_ptr, out_ptr):
        layout: gl.constexpr = gl.BlockedLayout([1, 1], [32, 1], [gl.num_warps(), 1], [1, 0])
        offs_m = gl.arange(0, M, layout=gl.SliceLayout(1, layout))[:, None]
        offs_n = gl.arange(0, N, layout=gl.SliceLayout(1, layout))[:, None]
        offs_k = gl.arange(0, K, layout=gl.SliceLayout(0, layout))[None, :]
        out_offs_n = gl.arange(0, N, layout=gl.SliceLayout(0, layout))[None, :]

        a = gl.load(a_ptr + offs_m * K + offs_k)
        b = gl.load(b_ptr + offs_n * K + offs_k)
        smem_a = gl.allocate_shared_memory(gl.float32, [M, K], gl.NVMMASharedLayout.get_default_for([M, K], gl.float32),
                                           a)
        smem_b = gl.allocate_shared_memory(gl.float32, [N, K], gl.NVMMASharedLayout.get_default_for([N, K], gl.float32),
                                           b)
        acc_tmem = allocate_tensor_memory(gl.float32, [M, N], layout=TensorMemoryLayout((M, N), col_stride=1))
        bar = mbarrier.allocate_mbarrier()
        mbarrier.init(bar, count=1)

        gl.warp_specialize([
            (default_partition, ()),
            (mma_partition, (smem_a, smem_b, acc_tmem, bar)),
        ], [PARTITION_WARPS])

        mbarrier.wait(bar, phase=0, deps=[smem_a, smem_b])
        mbarrier.invalidate(bar)
        out = gl.convert_layout(acc_tmem.load(), layout)
        gl.store(out_ptr + offs_m * N + out_offs_n, out)

    rs = np.random.RandomState(0)
    a_bits = rs.randint(-(2**31), 2**31 - 1, size=(M.value, K.value), dtype=np.int32)
    b_bits = rs.randint(-(2**31), 2**31 - 1, size=(N.value, K.value), dtype=np.int32)
    exp_bits = _mm_payload_u32(a_bits, b_bits.T)
    a = torch.tensor(a_bits, device=device, dtype=torch.int32)
    b = torch.tensor(b_bits, device=device, dtype=torch.int32)
    out = torch.empty((M.value, N.value), device=device, dtype=torch.int32)

    kernel[(1, )](
        triton.TensorWrapper(a, dtype=torch.float32),
        triton.TensorWrapper(b, dtype=torch.float32),
        triton.TensorWrapper(out, dtype=torch.float32),
        num_warps=4,
    )
    _assert_payload_equal(out, exp_bits)


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("use_acc", [False, True])
def test_tcgen05_mma_two_ctas(device, use_acc, fresh_knobs):
    _require_cuda_backend(device)

    M = 256
    N = 128
    K = 64
    BLOCK_M = gl.constexpr(M)
    BLOCK_N = gl.constexpr(N)
    BLOCK_K = gl.constexpr(K)

    fresh_knobs.compilation.instrumentation_mode = "fpsan"

    @gluon.jit
    def kernel(a_ptr, b_ptr, c_ptr, out_ptr, USE_ACC: gl.constexpr):
        a_layout: gl.constexpr = gl.BlockedLayout([1, 1], [32, 1], [gl.num_warps(), 1], [1, 0], cga_layout=((1, 0), ))
        b_layout: gl.constexpr = gl.BlockedLayout([1, 1], [32, 1], [gl.num_warps(), 1], [1, 0], cga_layout=((0, 1), ))
        out_layout: gl.constexpr = gl.BlockedLayout([1, 1], [32, 1], [gl.num_warps(), 1], [1, 0], cga_layout=((1, 0), ))

        offs_m = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, a_layout))[:, None]
        offs_k_a = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(0, a_layout))[None, :]
        offs_k_b = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(1, b_layout))[:, None]
        offs_n = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, b_layout))[None, :]
        out_offs_m = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, out_layout))[:, None]
        out_offs_n = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, out_layout))[None, :]

        a_tile = gl.load(a_ptr + offs_m * BLOCK_K + offs_k_a)
        b_tile = gl.load(b_ptr + offs_k_b * BLOCK_N + offs_n)

        smem_layout_a: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_K], gl.float32,
                                                                           cga_layout=((1, 0), ))
        smem_layout_b: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK_K, BLOCK_N], gl.float32,
                                                                           cga_layout=((0, 1), ))
        smem_a = gl.allocate_shared_memory(gl.float32, [BLOCK_M, BLOCK_K], smem_layout_a, a_tile)
        smem_b = gl.allocate_shared_memory(gl.float32, [BLOCK_K, BLOCK_N], smem_layout_b, b_tile)

        tmem_layout: gl.constexpr = TensorMemoryLayout((128, BLOCK_N), col_stride=1, cga_layout=((1, 0), ),
                                                       two_ctas=True)
        acc_tmem = allocate_tensor_memory(gl.float32, [BLOCK_M, BLOCK_N], layout=tmem_layout)
        if USE_ACC:
            c_tile = gl.load(c_ptr + out_offs_m * BLOCK_N + out_offs_n)
            acc_tmem.store(gl.convert_layout(c_tile, acc_tmem.get_reg_layout()))

        bar = mbarrier.allocate_mbarrier()
        mbarrier.init(bar, count=1)
        tcgen05_mma(smem_a, smem_b, acc_tmem, use_acc=USE_ACC, pred=True, mbarriers=[bar])
        mbarrier.wait(bar, phase=0, deps=[smem_a, smem_b])
        mbarrier.invalidate(bar)

        out = gl.convert_layout(acc_tmem.load(), out_layout)
        store_offs_m = gl.arange(0, BLOCK_M)[:, None]
        store_offs_n = gl.arange(0, BLOCK_N)[None, :]
        gl.store(out_ptr + store_offs_m * BLOCK_N + store_offs_n, out)

    rs = np.random.RandomState(0)
    a_bits = rs.randint(-(2**31), 2**31 - 1, size=(M, K), dtype=np.int32)
    b_bits = rs.randint(-(2**31), 2**31 - 1, size=(K, N), dtype=np.int32)
    c_bits = rs.randint(-(2**31), 2**31 - 1, size=(M, N), dtype=np.int32)
    exp_bits = _mm_payload_u32(a_bits, b_bits, c_bits if use_acc else None)

    a = torch.tensor(a_bits, device="cuda", dtype=torch.int32)
    b = torch.tensor(b_bits, device="cuda", dtype=torch.int32)
    c = torch.tensor(c_bits, device="cuda", dtype=torch.int32)
    out = torch.empty((M, N), device="cuda", dtype=torch.int32)

    kernel[(1, )](
        triton.TensorWrapper(a, dtype=torch.float32),
        triton.TensorWrapper(b, dtype=torch.float32),
        triton.TensorWrapper(c, dtype=torch.float32),
        triton.TensorWrapper(out, dtype=torch.float32),
        USE_ACC=use_acc,
        num_warps=4,
        num_ctas=2,
    )

    _assert_payload_equal(out, exp_bits)


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize(("type_a", "type_b", "m", "n", "k", "scale_factor", "scale_type"), _TCGEN05_MMA_SCALED_CASES)
def test_tcgen05_mma_scaled(device, type_a, type_b, m, n, k, scale_factor, scale_type, fresh_knobs):
    _require_cuda_backend(device)

    M = gl.constexpr(m)
    N = gl.constexpr(n)
    K = gl.constexpr(k)
    SCALE_K = gl.constexpr(k // scale_factor)

    fresh_knobs.compilation.instrumentation_mode = "fpsan"

    @gluon.jit
    def kernel(a_ptr, b_ptr, a_scale_ptr, b_scale_ptr, c_ptr, out_ptr, TYPE_A: gl.constexpr, TYPE_B: gl.constexpr,
               SCALE_DTYPE: gl.constexpr):
        layout: gl.constexpr = gl.BlockedLayout([1, 1], [32, 1], [gl.num_warps(), 1], [1, 0])
        IS_FP4_A: gl.constexpr = TYPE_A == "e2m1"
        IS_FP4_B: gl.constexpr = TYPE_B == "e2m1"
        PACK_FACTOR_A: gl.constexpr = 2 if IS_FP4_A else 1
        PACK_FACTOR_B: gl.constexpr = 2 if IS_FP4_B else 1
        PACKED_K_A: gl.constexpr = K // PACK_FACTOR_A
        PACKED_K_B: gl.constexpr = K // PACK_FACTOR_B
        ELEM_DTYPE_A: gl.constexpr = gl.uint8 if IS_FP4_A else (gl.float8e4nv if TYPE_A == "e4m3" else gl.float8e5)
        ELEM_DTYPE_B: gl.constexpr = gl.uint8 if IS_FP4_B else (gl.float8e4nv if TYPE_B == "e4m3" else gl.float8e5)
        a_nvmma_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([M, PACKED_K_A], ELEM_DTYPE_A)
        b_nvmma_layout: gl.constexpr = (gl.NVMMASharedLayout.get_default_for([N, PACKED_K_B], ELEM_DTYPE_B)
                                        if IS_FP4_B else gl.NVMMASharedLayout(swizzle_byte_width=128, transposed=False,
                                                                              element_bitwidth=8, rank=2))
        scale_layout: gl.constexpr = TensorMemoryScalesLayout()

        offs_m = gl.arange(0, M, layout=gl.SliceLayout(1, layout))[:, None]
        offs_n = gl.arange(0, N, layout=gl.SliceLayout(0, layout))[None, :]
        offs_n_row = gl.arange(0, N, layout=gl.SliceLayout(1, layout))[:, None]
        offs_bk_row = gl.arange(0, PACKED_K_B, layout=gl.SliceLayout(1, layout))[:, None]
        offs_ak_col = gl.arange(0, PACKED_K_A, layout=gl.SliceLayout(0, layout))[None, :]
        offs_bk_col = gl.arange(0, PACKED_K_B, layout=gl.SliceLayout(0, layout))[None, :]

        a_tile = gl.load(a_ptr + offs_m * PACKED_K_A + offs_ak_col)
        c_tile = gl.load(c_ptr + offs_m * N + offs_n)
        a_smem = gl.allocate_shared_memory(ELEM_DTYPE_A, [M, PACKED_K_A], a_nvmma_layout, a_tile)
        if IS_FP4_B:
            b_tile = gl.load(b_ptr + offs_n_row * PACKED_K_B + offs_bk_col)
            b_smem = gl.allocate_shared_memory(ELEM_DTYPE_B, [N, PACKED_K_B], b_nvmma_layout, b_tile)
            b_mma = b_smem.permute((1, 0))
        else:
            b_tile = gl.load(b_ptr + offs_bk_row * N + offs_n)
            b_smem = gl.allocate_shared_memory(ELEM_DTYPE_B, [PACKED_K_B, N], b_nvmma_layout, b_tile)
            b_mma = b_smem

        tmem_layout: gl.constexpr = TensorMemoryLayout((M, N), col_stride=1)
        acc_tmem = allocate_tensor_memory(gl.float32, [M, N], layout=tmem_layout)
        acc_tmem.store(gl.convert_layout(c_tile, acc_tmem.get_reg_layout()))

        a_scale_tmem = allocate_tensor_memory(SCALE_DTYPE, [M, SCALE_K], layout=scale_layout)
        b_scale_tmem = allocate_tensor_memory(SCALE_DTYPE, [N, SCALE_K], layout=scale_layout)
        a_scale_reg_layout: gl.constexpr = a_scale_tmem.get_reg_layout()
        b_scale_reg_layout: gl.constexpr = b_scale_tmem.get_reg_layout()
        scale_offs_k = gl.arange(0, SCALE_K, layout=gl.SliceLayout(0, a_scale_reg_layout))[None, :]
        scale_offs_m = gl.arange(0, M, layout=gl.SliceLayout(1, a_scale_reg_layout))[:, None]
        scale_offs_n = gl.arange(0, N, layout=gl.SliceLayout(1, b_scale_reg_layout))[:, None]
        a_scale_tmem.store(gl.load(a_scale_ptr + scale_offs_m * SCALE_K + scale_offs_k))
        b_scale_tmem.store(gl.load(b_scale_ptr + scale_offs_n * SCALE_K + scale_offs_k))

        bar = gl.allocate_shared_memory(gl.int64, [1], gl.constexpr(mbarrier.MBarrierLayout()))
        mbarrier.init(bar, count=1)
        tcgen05_mma_scaled(a_smem, b_mma, acc_tmem, a_scale_tmem, b_scale_tmem, TYPE_A, TYPE_B, use_acc=True,
                           mbarriers=[bar])
        mbarrier.wait(bar, phase=0)
        mbarrier.invalidate(bar)

        out = gl.convert_layout(acc_tmem.load(), layout)
        gl.store(out_ptr + offs_m * N + offs_n, out)

    rs = np.random.RandomState(0)
    pack_factor_a = 2 if type_a == "e2m1" else 1
    pack_factor_b = 2 if type_b == "e2m1" else 1
    packed_k_a = k // pack_factor_a
    packed_k_b = k // pack_factor_b
    a_bits = rs.randint(0 if type_a == "e2m1" else 20, 256 if type_a == "e2m1" else 40, size=(m, packed_k_a),
                        dtype=np.uint8)
    if type_b == "e2m1":
        b_bits = rs.randint(0, 256, size=(n, packed_k_b), dtype=np.uint8)
        b_ref_bits = b_bits.T
    else:
        b_bits = rs.randint(20, 40, size=(packed_k_b, n), dtype=np.uint8)
        b_ref_bits = b_bits
    a_scale_bits = rs.randint(1, 4, size=(m, k // scale_factor), dtype=np.int8)
    b_scale_bits = rs.randint(1, 4, size=(n, k // scale_factor), dtype=np.int8)
    if scale_type == "e4m3":
        a_scale_bits = rs.randint(1, 0x40, size=(m, k // scale_factor), dtype=np.uint8)
        b_scale_bits = rs.randint(1, 0x40, size=(n, k // scale_factor), dtype=np.uint8)
    c_bits = rs.randint(-(2**31), 2**31 - 1, size=(m, n), dtype=np.int32)
    exp_bits = _mm_scaled_payload_u32(a_bits, b_ref_bits, a_scale_bits.view(np.uint8), b_scale_bits.view(np.uint8),
                                      c_bits, a_pack=pack_factor_a, b_pack=pack_factor_b, type_a=type_a, type_b=type_b,
                                      scale_factor=scale_factor, scale_type=scale_type)

    if type_a == "e2m1":
        a = torch.tensor(a_bits, device="cuda", dtype=torch.uint8)
    else:
        a = torch.tensor(a_bits, device="cuda", dtype=torch.uint8).view(_float_dtype_info(type_a)[4])
    if type_b == "e2m1":
        b = torch.tensor(b_bits, device="cuda", dtype=torch.uint8)
    else:
        b = torch.tensor(b_bits, device="cuda", dtype=torch.uint8).view(_float_dtype_info(type_b)[4])
    if scale_type == "e4m3":
        a_scale = torch.tensor(a_scale_bits, device="cuda", dtype=torch.uint8).view(torch.float8_e4m3fn)
        b_scale = torch.tensor(b_scale_bits, device="cuda", dtype=torch.uint8).view(torch.float8_e4m3fn)
        scale_dtype = gl.float8e4nv
    else:
        a_scale = torch.tensor(a_scale_bits, device="cuda", dtype=torch.int8)
        b_scale = torch.tensor(b_scale_bits, device="cuda", dtype=torch.int8)
        scale_dtype = gl.int8
    c = torch.tensor(c_bits, device="cuda", dtype=torch.int32)
    out = torch.empty((m, n), device="cuda", dtype=torch.int32)

    cw = triton.TensorWrapper(c, dtype=torch.float32)
    outw = triton.TensorWrapper(out, dtype=torch.float32)

    kernel[(1, )](a, b, a_scale, b_scale, cw, outw, TYPE_A=type_a, TYPE_B=type_b, SCALE_DTYPE=scale_dtype)

    _assert_payload_equal(out, exp_bits)


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_tcgen05_mma_scaled_two_ctas(device, fresh_knobs):
    _require_cuda_backend(device)

    M = 256
    N = 128
    K = 128
    BLOCK_M = gl.constexpr(M)
    BLOCK_N = gl.constexpr(N)
    BLOCK_K = gl.constexpr(K)
    SCALE_K = gl.constexpr(K // 32)

    fresh_knobs.compilation.instrumentation_mode = "fpsan"

    @gluon.jit
    def kernel(a_ptr, b_ptr, a_scale_ptr, b_scale_ptr, c_ptr, out_ptr):
        a_layout: gl.constexpr = gl.BlockedLayout([1, 1], [32, 1], [gl.num_warps(), 1], [1, 0], cga_layout=((1, 0), ))
        b_layout: gl.constexpr = gl.BlockedLayout([1, 1], [32, 1], [gl.num_warps(), 1], [1, 0], cga_layout=((1, 0), ))
        out_layout: gl.constexpr = gl.BlockedLayout([1, 1], [32, 1], [gl.num_warps(), 1], [1, 0], cga_layout=((1, 0), ))

        offs_m = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, a_layout))[:, None]
        offs_k_a = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(0, a_layout))[None, :]
        offs_n_b = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, b_layout))[:, None]
        offs_k_b = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(0, b_layout))[None, :]
        out_offs_m = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, out_layout))[:, None]
        out_offs_n = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, out_layout))[None, :]

        a_tile = gl.load(a_ptr + offs_m * BLOCK_K + offs_k_a)
        b_tile = gl.load(b_ptr + offs_n_b * BLOCK_K + offs_k_b)
        c_tile = gl.load(c_ptr + out_offs_m * BLOCK_N + out_offs_n)

        a_smem_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_K], gl.float8e5,
                                                                           cga_layout=((1, 0), ))
        b_smem_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK_N, BLOCK_K], gl.float8e5,
                                                                           cga_layout=((1, 0), ))
        a_smem = gl.allocate_shared_memory(gl.float8e5, [BLOCK_M, BLOCK_K], a_smem_layout, a_tile)
        b_smem = gl.allocate_shared_memory(gl.float8e5, [BLOCK_N, BLOCK_K], b_smem_layout, b_tile)

        acc_layout: gl.constexpr = TensorMemoryLayout((128, BLOCK_N), col_stride=1, cga_layout=((1, 0), ),
                                                      two_ctas=True)
        acc = allocate_tensor_memory(gl.float32, [BLOCK_M, BLOCK_N], acc_layout)
        acc.store(gl.convert_layout(c_tile, acc.get_reg_layout()))

        a_scale_layout: gl.constexpr = TensorMemoryScalesLayout(cga_layout=((1, 0), ))
        b_scale_layout: gl.constexpr = TensorMemoryScalesLayout(cga_layout=((0, 0), ))
        a_scale = allocate_tensor_memory(gl.int8, [BLOCK_M, SCALE_K], a_scale_layout)
        b_scale = allocate_tensor_memory(gl.int8, [BLOCK_N, SCALE_K], b_scale_layout)
        a_scale_reg_layout: gl.constexpr = a_scale.get_reg_layout()
        b_scale_reg_layout: gl.constexpr = b_scale.get_reg_layout()
        a_scale_offs_k = gl.arange(0, SCALE_K, layout=gl.SliceLayout(0, a_scale_reg_layout))[None, :]
        b_scale_offs_k = gl.arange(0, SCALE_K, layout=gl.SliceLayout(0, b_scale_reg_layout))[None, :]
        scale_offs_m = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, a_scale_reg_layout))[:, None]
        scale_offs_n = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, b_scale_reg_layout))[:, None]
        a_scale_values = gl.load(a_scale_ptr + scale_offs_m * SCALE_K + a_scale_offs_k)
        b_scale_values = gl.load(b_scale_ptr + scale_offs_n * SCALE_K + b_scale_offs_k)
        scale_smem_offset_bases: gl.constexpr = [
            [0, 1],
            [0, 2],
            [32, 0],
            [64, 0],
            [1, 0],
            [2, 0],
            [4, 0],
            [8, 0],
            [16, 0],
        ]
        a_scale_smem_layout: gl.constexpr = gl.SharedLinearLayout(
            offset_bases=scale_smem_offset_bases,
            block_bases=((128, 0), ),
        )
        b_scale_smem_layout: gl.constexpr = gl.SharedLinearLayout(
            offset_bases=scale_smem_offset_bases,
            block_bases=((0, 0), ),
        )
        a_scale_smem = gl.allocate_shared_memory(gl.int8, [BLOCK_M, SCALE_K], a_scale_smem_layout)
        b_scale_smem = gl.allocate_shared_memory(gl.int8, [BLOCK_N, SCALE_K], b_scale_smem_layout)
        a_scale_smem.store(a_scale_values)
        b_scale_smem.store(b_scale_values)
        tcgen05_copy(a_scale_smem, a_scale)
        tcgen05_copy(b_scale_smem, b_scale)

        bar = mbarrier.allocate_mbarrier()
        mbarrier.init(bar, count=1)
        tcgen05_mma_scaled(a_smem, b_smem.permute((1, 0)), acc, a_scale, b_scale, "e5m2", "e5m2", use_acc=True,
                           mbarriers=[bar])
        mbarrier.wait(bar, phase=0, deps=[a_smem, b_smem])
        mbarrier.invalidate(bar)

        out = gl.convert_layout(acc.load(), out_layout)
        store_offs_m = gl.arange(0, BLOCK_M)[:, None]
        store_offs_n = gl.arange(0, BLOCK_N)[None, :]
        gl.store(out_ptr + store_offs_m * BLOCK_N + store_offs_n, out)

    rs = np.random.RandomState(0)
    a_bits = rs.randint(20, 40, size=(M, K), dtype=np.uint8)
    b_bits = rs.randint(20, 40, size=(N, K), dtype=np.uint8)
    a_scale_bits = rs.randint(1, 4, size=(M, K // 32), dtype=np.int8)
    b_scale_bits = rs.randint(1, 4, size=(N, K // 32), dtype=np.int8)
    c_bits = rs.randint(-(2**31), 2**31 - 1, size=(M, N), dtype=np.int32)
    exp_bits = _mm_scaled_payload_u32(a_bits, b_bits.T, a_scale_bits.view(np.uint8), b_scale_bits.view(np.uint8),
                                      c_bits, a_pack=1, b_pack=1, type_a="e5m2", type_b="e5m2")

    a = torch.tensor(a_bits, device="cuda", dtype=torch.uint8).view(torch.float8_e5m2)
    b = torch.tensor(b_bits, device="cuda", dtype=torch.uint8).view(torch.float8_e5m2)
    a_scale = torch.tensor(a_scale_bits, device="cuda", dtype=torch.int8)
    b_scale = torch.tensor(b_scale_bits, device="cuda", dtype=torch.int8)
    c = torch.tensor(c_bits, device="cuda", dtype=torch.int32)
    out = torch.empty((M, N), device="cuda", dtype=torch.int32)

    kernel[(1, )](
        a,
        b,
        a_scale,
        b_scale,
        triton.TensorWrapper(c, dtype=torch.float32),
        triton.TensorWrapper(out, dtype=torch.float32),
        num_warps=4,
        num_ctas=2,
    )

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


@pytest.mark.skipif(not is_blackwell_ultra(), reason="Requires Blackwell Ultra")
@pytest.mark.parametrize("red_op,use_abs", [("min", False), ("max", True)])
def test_tmem_load_reduce(device, red_op, use_abs, fresh_knobs):
    _require_cuda_backend(device)

    m = 128
    n = 128
    M = gl.constexpr(m)
    N = gl.constexpr(n)

    fresh_knobs.compilation.instrumentation_mode = "fpsan"

    @gluon.jit
    def kernel(x_ptr, out_ptr, RED_OP: gl.constexpr, USE_ABS: gl.constexpr):
        layout: gl.constexpr = gl.BlockedLayout([1, 1], [32, 1], [gl.num_warps(), 1], [1, 0])
        red_layout: gl.constexpr = gl.SliceLayout(1, layout)
        offs_m = gl.arange(0, M, layout=red_layout)
        offs_n = gl.arange(0, N, layout=gl.SliceLayout(0, layout))
        offs = offs_m[:, None] * N + offs_n[None, :]

        value = gl.load(x_ptr + offs)
        tmem_layout: gl.constexpr = TensorMemoryLayout((M, N), col_stride=1)
        tmem = allocate_tensor_memory(gl.float32, [M, N], layout=tmem_layout)
        tmem.store(gl.convert_layout(value, tmem.get_reg_layout()))

        if RED_OP == "min":
            _, reduced = tmem.load_min(abs=USE_ABS)
        else:
            _, reduced = tmem.load_max(abs=USE_ABS)
        gl.store(out_ptr + offs_m, gl.convert_layout(reduced, red_layout))

    rs = np.random.RandomState(0)
    x_bits = rs.uniform(-100.0, 100.0, size=(m, n)).astype(np.float32).view(np.int32)
    reduction_bits = x_bits
    if use_abs:
        reduction_bits = _u32_to_i32(_as_u32(x_bits) & np.uint32(0x7FFFFFFF))
    payload = _u32_to_i32(_mix_f32_bits_to_payload_u32(reduction_bits))
    reduced_payload = (payload.min(axis=1) if red_op == "min" else payload.max(axis=1)).astype(np.int32)
    exp_bits = _unmix_payload_u32_to_f32_bits_i32(reduced_payload.view(np.uint32))

    x = torch.tensor(x_bits, device=device, dtype=torch.int32)
    out = torch.empty((m, ), device=device, dtype=torch.int32)
    kernel[(1, )](
        triton.TensorWrapper(x, dtype=torch.float32),
        triton.TensorWrapper(out, dtype=torch.float32),
        RED_OP=red_op,
        USE_ABS=use_abs,
        num_warps=4,
    )

    _assert_payload_equal(out, exp_bits)


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("scale_shape", [(64, 16), (256, 4)])
@pytest.mark.parametrize("two_ctas", [False, True])
def test_tmem_copy_scales_in_warp_specialize_partition(device, scale_shape, two_ctas, fresh_knobs):
    _require_cuda_backend(device)

    smem_h, smem_w = scale_shape
    SMEM_H = gl.constexpr(smem_h)
    SMEM_W = gl.constexpr(smem_w)
    tmem_rows = 128
    tmem_cols = 32
    TMEM_ROWS = gl.constexpr(tmem_rows)
    TMEM_COLS = gl.constexpr(tmem_cols)

    fresh_knobs.compilation.instrumentation_mode = "fpsan"

    @gluon.jit
    def copy_partition(smem, tmem, bar):
        tcgen05_copy(smem, tmem)
        tcgen05_commit(bar)

    @gluon.jit
    def load_partition(physical, bar, out_ptr):
        mbarrier.wait(bar, phase=0)
        mbarrier.invalidate(bar)
        physical_reg_layout: gl.constexpr = physical.get_reg_layout()
        copied = physical.load(physical_reg_layout)
        out_ptrs = out_ptr + gl.arange(0, TMEM_ROWS)[:, None] * TMEM_COLS + gl.arange(0, TMEM_COLS)[None, :]
        gl.store(gl.set_auto_layout(out_ptrs, physical_reg_layout), copied)

    @gluon.jit
    def default_partition():
        pass

    @gluon.jit
    def kernel(in_ptr, out_ptr, TWO_CTAS: gl.constexpr):
        if TWO_CTAS:
            mma_a_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([256, 64], gl.float32,
                                                                              cga_layout=((1, 0), ))
            mma_b_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([64, 128], gl.float32,
                                                                              cga_layout=((0, 1), ))
            mma_a = gl.allocate_shared_memory(gl.float32, [256, 64], mma_a_layout)
            mma_b = gl.allocate_shared_memory(gl.float32, [64, 128], mma_b_layout)
            mma_acc_layout: gl.constexpr = TensorMemoryLayout((128, 128), col_stride=1, cga_layout=((1, 0), ),
                                                              two_ctas=True)
            mma_acc = allocate_tensor_memory(gl.float32, [256, 128], mma_acc_layout)
            tcgen05_mma(mma_a, mma_b, mma_acc, use_acc=False)

        cga_layout: gl.constexpr = ((0, 0), ) if TWO_CTAS else ()
        blocked: gl.constexpr = gl.BlockedLayout([1, 4], [32, 1], [gl.num_warps(), 1], [1, 0], cga_layout=cga_layout)
        in_ptrs = (in_ptr + gl.arange(0, SMEM_H)[:, None] * SMEM_W + gl.arange(0, SMEM_W)[None, :])
        value = gl.load(gl.set_auto_layout(in_ptrs, blocked))

        if SMEM_H == 64:
            smem_offset_bases: gl.constexpr = [
                [0, 1],
                [0, 2],
                [32, 0],
                [0, 4],
                [1, 0],
                [2, 0],
                [4, 0],
                [8, 0],
                [16, 0],
                [0, 8],
            ]
        else:
            smem_offset_bases: gl.constexpr = [
                [0, 1],
                [0, 2],
                [32, 0],
                [64, 0],
                [1, 0],
                [2, 0],
                [4, 0],
                [8, 0],
                [16, 0],
                [128, 0],
            ]
        smem_layout: gl.constexpr = gl.SharedLinearLayout(
            offset_bases=smem_offset_bases,
            block_bases=cga_layout,
        )
        smem = gl.allocate_shared_memory(gl.int8, (SMEM_H, SMEM_W), layout=smem_layout)
        smem.store(value)

        tmem_layout: gl.constexpr = TensorMemoryScalesLayout(cga_layout=cga_layout)
        tmem = allocate_tensor_memory(gl.int8, (SMEM_H, SMEM_W), layout=tmem_layout)
        bar = mbarrier.allocate_mbarrier()
        mbarrier.init(bar, count=1)
        physical_layout: gl.constexpr = TensorMemoryLayout((TMEM_ROWS, TMEM_COLS), col_stride=1, cga_layout=cga_layout)
        physical = tmem._reinterpret(shape=(TMEM_ROWS, TMEM_COLS), layout=physical_layout)

        gl.warp_specialize(
            [
                (default_partition, ()),
                (copy_partition, (smem, tmem, bar)),
                (load_partition, (physical, bar, out_ptr)),
            ],
            [1, 4],
            [32, 32],
        )

    rs = np.random.RandomState(0)
    x_np = rs.randint(-100, 100, size=(smem_h, smem_w), dtype=np.int8)
    warp_tile = x_np.reshape(smem_h // 32, 32, smem_w // 4, 4).transpose(1, 2, 0, 3).reshape(32, -1)
    expected = np.tile(warp_tile, (4, 1))

    x = torch.tensor(x_np, device=device, dtype=torch.int8)
    out = torch.empty((tmem_rows, tmem_cols), device=device, dtype=torch.int8)
    kernel[(1, )](x, out, TWO_CTAS=two_ctas, num_warps=4, num_ctas=2 if two_ctas else 1)
    torch.testing.assert_close(out, torch.tensor(expected, device=device))


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_tmem_store_in_warp_specialize_partition_visible_to_parent(device, fresh_knobs):
    _require_cuda_backend(device)

    B = 64
    BLOCK = gl.constexpr(B)

    fresh_knobs.compilation.instrumentation_mode = "fpsan"

    @gluon.jit
    def store_one_partition(tmem, bar):
        reg_layout: gl.constexpr = tmem.get_reg_layout()
        one = gl.full((BLOCK, BLOCK), 1.0, gl.float32, reg_layout)
        tmem.store(one)
        mbarrier.arrive(bar, count=1)

    @gluon.jit
    def default_partition():
        pass

    @gluon.jit
    def kernel(out_ptr):
        layout: gl.constexpr = gl.BlockedLayout([1, 1], [32, 1], [gl.num_warps(), 1], [1, 0])
        offs_m = gl.arange(0, BLOCK, layout=gl.SliceLayout(1, layout))[:, None]
        offs_n = gl.arange(0, BLOCK, layout=gl.SliceLayout(0, layout))[None, :]
        offs = offs_m * BLOCK + offs_n

        tmem_layout: gl.constexpr = TensorMemoryLayout((BLOCK, BLOCK), col_stride=1)
        tmem = allocate_tensor_memory(gl.float32, [BLOCK, BLOCK], layout=tmem_layout)
        reg_layout: gl.constexpr = tmem.get_reg_layout()
        zero = gl.full((BLOCK, BLOCK), 0.0, gl.float32, reg_layout)
        tmem.store(zero)

        bar = gl.allocate_shared_memory(gl.int64, [1], gl.constexpr(mbarrier.MBarrierLayout()))
        mbarrier.init(bar, count=1)
        gl.warp_specialize([
            (default_partition, ()),
            (store_one_partition, (tmem, bar)),
        ], [4], [32])
        mbarrier.wait(bar, phase=0, deps=[tmem])
        mbarrier.invalidate(bar)

        out = tmem.load()
        out = gl.convert_layout(out, layout)
        gl.store(out_ptr + offs, out)

    out = torch.empty((B, B), device=device, dtype=torch.float32)
    kernel[(1, )](out, num_warps=4)

    torch.testing.assert_close(out, torch.ones_like(out), rtol=0, atol=0)


def test_reduction(device, fresh_knobs):
    _require_cuda_backend(device)

    @triton.jit
    def reduce_kernel(a_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, stride_ak: tl.constexpr, stride_am: tl.constexpr,
                      stride_an: tl.constexpr, ORDER: tl.constexpr):

        a_ptr += tl.program_id(0).to(tl.int64) * stride_ak
        c_ptr += tl.program_id(0).to(tl.int64)
        a_ptrs = a_ptr + (tl.arange(0, M)[:, None] * stride_am + (tl.arange(0, N)[None, :]) * stride_an)
        a = tl.load(a_ptrs)
        r1 = tl.sum(a, axis=ORDER)
        r2 = tl.sum(r1, axis=0)
        tl.store(c_ptr, r2)

    # we run K parallel tests so as to make non-associativity much more
    # likely to manifest:
    K, M, N = 100, 128, 128
    torch.manual_seed(0)
    a = torch.randn((K, M, N), dtype=torch.float32, device="cuda")
    c1 = torch.empty((K, ), dtype=torch.float32).to('cuda')
    c2 = torch.empty((K, ), dtype=torch.float32).to('cuda')

    reduce_kernel[(K, )](a, c1, M, N, a.stride(0), a.stride(1), a.stride(2), ORDER=0)
    reduce_kernel[(K, )](a, c2, M, N, a.stride(0), a.stride(1), a.stride(2), ORDER=1)
    assert not _payload_equal(c1, c2)

    fresh_knobs.compilation.instrumentation_mode = "fpsan"

    reduce_kernel[(K, )](a, c1, M, N, a.stride(0), a.stride(1), a.stride(2), ORDER=0)
    reduce_kernel[(K, )](a, c2, M, N, a.stride(0), a.stride(1), a.stride(2), ORDER=1)
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


def test_f32_loop_preserves_snan_payload(device, fresh_knobs):
    _require_cuda_backend(device)
    if not is_cuda():
        pytest.skip("regression is specific to NVPTX fabs lowering")

    @triton.jit
    def sum_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
        offsets = tl.arange(0, BLOCK)
        acc = tl.zeros((BLOCK, ), tl.float32)
        for i in range(3):
            acc += tl.load(x_ptr + i * BLOCK + offsets)
        tl.store(out_ptr + offsets, acc)

    fresh_knobs.compilation.instrumentation_mode = "fpsan"
    fresh_knobs.compilation.always_compile = True

    block = 128
    # The first two finite values sum to an sNaN; the zero row forces it through the next loop embed.
    input_bits = np.zeros((3, block), dtype=np.int32)
    input_bits[0].fill(0x1B0F577C)
    input_bits[1].fill(0x65E031B7)
    assert np.isfinite(input_bits.view(np.float32)).all()
    x = torch.tensor(input_bits, dtype=torch.int32, device="cuda")
    out = torch.empty((block, ), dtype=torch.int32, device="cuda")
    sum_kernel[(1, )](
        triton.TensorWrapper(x, dtype=torch.float32),
        triton.TensorWrapper(out, dtype=torch.float32),
        BLOCK=block,
        num_warps=1,
    )

    expected = _expected_add_i32(input_bits[0], input_bits[1])
    expected = _expected_add_i32(expected, input_bits[2])
    assert np.all(_as_u32(expected) == np.uint32(0x7FA12345))
    _assert_payload_equal(out, expected)


@pytest.mark.skipif(not (is_hip_cdna3() or is_hip_cdna4()), reason="Requires CDNA3 or CDNA4")
@pytest.mark.parametrize(("type_a", "type_b", "acc_type", "m", "n", "k", "instr_m", "instr_n", "instr_k", "k_width"),
                         _MFMA_DOT_CASES)
def test_mfma_dot(device, type_a, type_b, acc_type, m, n, k, instr_m, instr_n, instr_k, k_width, fresh_knobs):
    _require_cuda_backend(device)

    fresh_knobs.compilation.instrumentation_mode = "fpsan"

    cdna_version = 3 if is_hip_cdna3() else 4

    blocked = gl.BlockedLayout([4, 4], [4, 16], [4, 1], [1, 0])
    mfma_layout = gl.amd.AMDMFMALayout(cdna_version, [instr_m, instr_n, instr_k], True, [4, 1],
                                       element_bitwidth=_float_dtype_info(acc_type)[0])

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
    a_bits = _random_float_bits(rs, (m, k), type_a)
    b_bits = _random_float_bits(rs, (k, n), type_b)
    c_bits = _random_float_bits(rs, (m, n), acc_type)
    exp_bits = _mm_payload_bits(a_bits, b_bits, c_bits, type_a, type_b, acc_type)

    _, aw = _as_float_bits_tensor(a_bits, type_a)
    _, bw = _as_float_bits_tensor(b_bits, type_b)
    _, cw = _as_float_bits_tensor(c_bits, acc_type)
    out, outw = _as_float_bits_tensor(np.empty((m, n), dtype=_float_dtype_info(acc_type)[2]), acc_type)

    kernel[(1, )](aw, bw, cw, outw, BLOCK_M=m, BLOCK_N=n, BLOCK_K=k, blocked=blocked, k_width=k_width,
                  mfma_layout=mfma_layout)

    _assert_payload_equal(out, exp_bits)


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250")
@pytest.mark.parametrize(("type_a", "type_b", "acc_type", "m", "n", "k", "instr_k", "k_width"), _WMMA_DOT_CASES)
def test_wmma_dot(device, type_a, type_b, acc_type, m, n, k, instr_k, k_width, fresh_knobs):
    _require_cuda_backend(device)

    fresh_knobs.compilation.instrumentation_mode = "fpsan"

    @gluon.jit
    def kernel(a_ptr, b_ptr, c_ptr, out_ptr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr,
               INSTR_SHAPE_K: gl.constexpr, K_WIDTH: gl.constexpr):
        blocked: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0])
        wmma: gl.constexpr = gl.amd.AMDWMMALayout(3, True, [[0, 1], [1, 0]], [], [16, 16, INSTR_SHAPE_K])

        offs_m = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, blocked))[:, None]
        offs_k = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(0, blocked))[None, :]
        offs_bk = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(1, blocked))[:, None]
        offs_n = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, blocked))[None, :]

        a = gl.load(a_ptr + offs_m * BLOCK_K + offs_k)
        b = gl.load(b_ptr + offs_bk * BLOCK_N + offs_n)
        c = gl.load(c_ptr + offs_m * BLOCK_N + offs_n)
        c = gl.convert_layout(c, wmma)

        a = gl.convert_layout(a, gl.DotOperandLayout(0, wmma, K_WIDTH))
        b = gl.convert_layout(b, gl.DotOperandLayout(1, wmma, K_WIDTH))
        acc = gl.amd.gfx1250.wmma(a, b, c)

        out_layout: gl.constexpr = gl.SliceLayout(1, wmma)
        offs_cm = gl.arange(0, BLOCK_M, layout=out_layout)[:, None]
        offs_cn = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, wmma))[None, :]
        gl.store(out_ptr + offs_cm * BLOCK_N + offs_cn, acc)

    rs = np.random.RandomState(0)
    a_bits = _random_float_bits(rs, (m, k), type_a)
    b_bits = _random_float_bits(rs, (k, n), type_b)
    c_bits = _random_float_bits(rs, (m, n), acc_type)
    exp_bits = _mm_payload_bits(a_bits, b_bits, c_bits, type_a, type_b, acc_type)

    _, aw = _as_float_bits_tensor(a_bits, type_a)
    _, bw = _as_float_bits_tensor(b_bits, type_b)
    _, cw = _as_float_bits_tensor(c_bits, acc_type)
    out, outw = _as_float_bits_tensor(np.empty((m, n), dtype=_float_dtype_info(acc_type)[2]), acc_type)

    kernel[(1, )](aw, bw, cw, outw, BLOCK_M=m, BLOCK_N=n, BLOCK_K=k, INSTR_SHAPE_K=instr_k, K_WIDTH=k_width)

    _assert_payload_equal(out, exp_bits)
