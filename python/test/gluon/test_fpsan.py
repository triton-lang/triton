# ruff: noqa: F821
import numpy as np
import pytest
import torch

import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton import language as tl
from triton._internal_testing import is_blackwell, is_cuda, is_hip, is_hip_cdna3, is_hip_cdna4, is_hip_gfx1250, is_interpreter
from triton.experimental.gluon.language.nvidia.blackwell import (TensorMemoryLayout, allocate_tensor_memory, mbarrier,
                                                                 tcgen05_mma, get_tmem_reg_layout)

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
    "div_inv": np.uint64(12),
}

OP_TO_TAG_U32 = {name: np.uint32(murmur64Mixer(op_id) & np.uint64(0xFFFFFFFF)) for name, op_id in OP_TO_ID_U64.items()}


def _expected_div_payload_i32(x_i32: np.ndarray, y_i32: np.ndarray) -> np.ndarray:
    # fpsan division is defined as: num_bits * (den_bits xor DivInvOpId) mod 2^32.
    # Keep this in sync with UnaryOpId::DivInv in FpSanitizer.cpp.
    div_inv_tag = OP_TO_TAG_U32["div_inv"].astype(np.uint64)
    mask = np.uint64(0xFFFFFFFF)
    num = _as_u32(x_i32).astype(np.uint64)
    den = _as_u32(y_i32).astype(np.uint64)
    tagged = (den ^ div_inv_tag).astype(np.uint64)
    out_u32 = ((num * tagged) & mask).astype(np.uint32)
    return _u32_to_i32(out_u32)


def _expected_unary_tag_i32(x_i32: np.ndarray, op: str) -> np.ndarray:
    # Keep this mapping in sync with UnaryOpId in FpSanitizer.cpp.
    tag = OP_TO_TAG_U32[op]
    out_u32 = _as_u32(x_i32) ^ tag
    return _u32_to_i32(out_u32)


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

    exp_bits = _expected_unary_tag_i32(x_bits, op)
    _assert_payload_equal(out, exp_bits)


def _expected_fma_i32(x_i32: np.ndarray, y_i32: np.ndarray, z_i32: np.ndarray) -> np.ndarray:
    return _expected_add_i32(_expected_mul_i32(x_i32, y_i32), z_i32)


def _expected_trunc_ext_roundtrip_i32(x_i32: np.ndarray) -> np.ndarray:
    x_u32 = _as_u32(x_i32)
    out_u32 = x_u32 & np.uint32(0xFFFF0000)
    return _u32_to_i32(out_u32)


def _expected_ext_f16_to_f32_i32(x_i16: np.ndarray) -> np.ndarray:
    x_u16 = x_i16.view(np.uint16).astype(np.uint32)
    out_u32 = (x_u16 << np.uint32(16)).astype(np.uint32)
    return out_u32.view(np.int32)


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
    out = torch.empty((n_elements, ), dtype=torch.int32, device="cuda")

    xw = triton.TensorWrapper(x, dtype=torch.float32)
    outw = triton.TensorWrapper(out, dtype=torch.float32)

    grid = (triton.cdiv(n_elements, BLOCK), )
    _cast_trunc_ext_kernel[grid](xw, outw, n_elements, BLOCK=BLOCK, THREADS_PER_WARP=THREADS_PER_WARP)

    out_np = out.cpu().numpy().astype(np.int32, copy=False)
    exp_np = _expected_trunc_ext_roundtrip_i32(x.cpu().numpy().astype(np.int32, copy=False))
    _assert_payload_equal(out_np, exp_np)


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
    out = torch.empty((n_elements, ), dtype=torch.int32, device="cuda")

    xw = triton.TensorWrapper(x, dtype=torch.float16)
    outw = triton.TensorWrapper(out, dtype=torch.float32)

    grid = (triton.cdiv(n_elements, BLOCK), )
    _cast_ext_kernel[grid](xw, outw, n_elements, BLOCK=BLOCK, THREADS_PER_WARP=THREADS_PER_WARP)

    out_np = out.cpu().numpy().astype(np.int32, copy=False)
    exp_np = _expected_ext_f16_to_f32_i32(x.cpu().numpy().astype(np.int16, copy=False))
    _assert_payload_equal(out_np, exp_np)


def _mm_payload_u32(a_i32: np.ndarray, b_i32: np.ndarray, c_i32: np.ndarray = None) -> np.ndarray:
    # Computes: c + a @ b in Z/(2^32) on raw payload bits.
    a_u = a_i32.view(np.uint32).astype(np.uint64)
    b_u = b_i32.view(np.uint32).astype(np.uint64)
    c_u = c_i32.view(np.uint32).astype(np.uint64) if c_i32 is not None else None
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
    return out.astype(np.uint32).view(np.int32)


def test_dot_fma(device, fresh_knobs):
    _require_cuda_backend(device)

    B = 16
    BLOCK = gl.constexpr(B)

    def allocator(size: int, alignment: int, stream):
        return torch.empty(size, device="cuda", dtype=torch.int32)

    triton.set_allocator(allocator)

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


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("use_acc", [False, True])
def test_tcgen05_mma(device, use_acc, fresh_knobs):
    _require_cuda_backend(device)

    B = 64
    BLOCK = gl.constexpr(B)

    def allocator(size: int, alignment: int, stream):
        return torch.empty(size, device="cuda", dtype=torch.int32)

    triton.set_allocator(allocator)

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
        acc_reg_layout: gl.constexpr = get_tmem_reg_layout(gl.float32, (BLOCK, BLOCK), tmem_layout, gl.num_warps())
        acc_tmem = allocate_tensor_memory(gl.float32, [BLOCK, BLOCK], layout=tmem_layout)
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

        out = acc_tmem.load(acc_reg_layout)
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
def test_tmem_index_subslice(device, fresh_knobs):
    _require_cuda_backend(device)

    B = 64
    BLOCK = gl.constexpr(B)
    SLICE_N = gl.constexpr(32)

    def allocator(size: int, alignment: int, stream):
        return torch.empty(size, device="cuda", dtype=torch.int32)

    triton.set_allocator(allocator)

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

        sub_layout: gl.constexpr = TensorMemoryLayout((BLOCK, SLICE_N), col_stride=1)
        sub_reg_layout: gl.constexpr = get_tmem_reg_layout(gl.float32, (BLOCK, SLICE_N), sub_layout, gl.num_warps())
        x_reg = gl.convert_layout(x, sub_reg_layout)
        sub.store(x_reg)
        out = sub.load(sub_reg_layout)
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

    def alloc_fn(size: int, alignment: int, stream):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    reduce_kernel[(1, )](a, c1, M=M, N=N, stride_am=a.stride(0), stride_ak=a.stride(1), ORDER=0)
    reduce_kernel[(1, )](a, c2, M=M, N=N, stride_am=a.stride(0), stride_ak=a.stride(1), ORDER=1)
    assert not _payload_equal(c1, c2)

    fresh_knobs.compilation.instrumentation_mode = "fpsan"

    reduce_kernel[(1, )](a, c1, M=M, N=N, stride_am=a.stride(0), stride_ak=a.stride(1), ORDER=0)
    reduce_kernel[(1, )](a, c2, M=M, N=N, stride_am=a.stride(0), stride_ak=a.stride(1), ORDER=1)
    assert _payload_equal(c1, c2)
