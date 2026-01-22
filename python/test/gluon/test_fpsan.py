# ruff: noqa: F821
import numpy as np
import pytest
import torch

import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl
from triton._internal_testing import is_blackwell, is_cuda, is_hip, is_interpreter
from triton.experimental.gluon.language.nvidia.blackwell import (TensorMemoryLayout, allocate_tensor_memory, mbarrier,
                                                                 tcgen05_mma, get_tmem_reg_layout)


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


def test_fpsan_dot_fma_payload_semantics(device):
    _require_cuda_backend(device)

    B = 16
    BLOCK = ttgl.constexpr(B)

    @gluon.jit
    def kernel(a_ptr, b_ptr, c_ptr, out_ptr):
        layout: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [32, 1], [4, 1], [1, 0])
        lhs_layout: ttgl.constexpr = ttgl.DotOperandLayout(parent=layout, operand_index=0, k_width=0)
        rhs_layout: ttgl.constexpr = ttgl.DotOperandLayout(parent=layout, operand_index=1, k_width=0)

        offs_m = ttgl.arange(0, BLOCK, layout=ttgl.SliceLayout(1, layout))[:, None]
        offs_n = ttgl.arange(0, BLOCK, layout=ttgl.SliceLayout(0, layout))[None, :]
        # Important: build separate offsets for A and B.
        # dot_fma expects operands to represent A[M,K] and B[K,N]. Using the same
        # linearized (m,n) offsets for both makes B effectively transposed.
        offs_k = ttgl.arange(0, BLOCK, layout=ttgl.SliceLayout(0, layout))[None, :]
        a_offs = offs_m * BLOCK + offs_k
        b_offs = offs_n * BLOCK + offs_m  # load B^T so dot_fma produces A @ B
        out_offs = offs_m * BLOCK + offs_n

        a = ttgl.convert_layout(ttgl.load(a_ptr + a_offs), lhs_layout)
        b = ttgl.convert_layout(ttgl.load(b_ptr + b_offs), rhs_layout)
        c = ttgl.load(c_ptr + out_offs)
        out = ttgl.dot_fma(a, b, c)
        ttgl.store(out_ptr + out_offs, out)

    rs = np.random.RandomState(0)
    a_bits = rs.randint(-(2**31), 2**31 - 1, size=(B, B), dtype=np.int32)
    b_bits = rs.randint(-(2**31), 2**31 - 1, size=(B, B), dtype=np.int32)
    c_bits = rs.randint(-(2**31), 2**31 - 1, size=(B, B), dtype=np.int32)
    exp_bits = _mm_payload_u32(a_bits, b_bits, c_bits)

    a = torch.tensor(a_bits, device="cuda", dtype=torch.int32)
    b = torch.tensor(b_bits, device="cuda", dtype=torch.int32)
    c = torch.tensor(c_bits, device="cuda", dtype=torch.int32)
    out = torch.empty((B, B), device="cuda", dtype=torch.int32)

    # Wrap int storage as fp32 so fpsan operates on payload bits.
    aw = triton.TensorWrapper(a, dtype=torch.float32)
    bw = triton.TensorWrapper(b, dtype=torch.float32)
    cw = triton.TensorWrapper(c, dtype=torch.float32)
    outw = triton.TensorWrapper(out, dtype=torch.float32)

    kernel[(1, )](aw, bw, cw, outw, fpsan=True, num_warps=4)

    np.testing.assert_array_equal(out.cpu().numpy().astype(np.int32, copy=False), exp_bits)


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_fpsan_tcgen05_mma_payload_semantics(device):
    _require_cuda_backend(device)

    B = 64
    BLOCK = ttgl.constexpr(B)

    def allocator(size: int, alignment: int, stream):
        return torch.empty(size, device="cuda", dtype=torch.int32)

    triton.set_allocator(allocator)

    @gluon.jit
    def kernel(a_ptr, b_ptr, out_ptr):
        layout: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [32, 1], [ttgl.num_warps(), 1], [1, 0])

        offs_m = ttgl.arange(0, BLOCK, layout=ttgl.SliceLayout(1, layout))[:, None]
        offs_n = ttgl.arange(0, BLOCK, layout=ttgl.SliceLayout(0, layout))[None, :]
        offs_k_row = ttgl.arange(0, BLOCK, layout=ttgl.SliceLayout(1, layout))[:, None]
        offs_k_col = ttgl.arange(0, BLOCK, layout=ttgl.SliceLayout(0, layout))[None, :]

        a_offs = offs_m * BLOCK + offs_k_col
        b_offs = offs_k_row * BLOCK + offs_n
        out_offs = offs_m * BLOCK + offs_n

        a_tile = ttgl.load(a_ptr + a_offs)
        b_tile = ttgl.load(b_ptr + b_offs)

        smem_layout_a: ttgl.constexpr = ttgl.NVMMASharedLayout.get_default_for([BLOCK, BLOCK], ttgl.float32)
        smem_layout_b: ttgl.constexpr = ttgl.NVMMASharedLayout.get_default_for([BLOCK, BLOCK], ttgl.float32)
        smem_a = ttgl.allocate_shared_memory(ttgl.float32, [BLOCK, BLOCK], smem_layout_a)
        smem_b = ttgl.allocate_shared_memory(ttgl.float32, [BLOCK, BLOCK], smem_layout_b)
        smem_a.store(a_tile)
        smem_b.store(b_tile)

        tmem_layout: ttgl.constexpr = TensorMemoryLayout((BLOCK, BLOCK), col_stride=1)
        acc_reg_layout: ttgl.constexpr = get_tmem_reg_layout(ttgl.float32, (BLOCK, BLOCK), tmem_layout,
                                                             ttgl.num_warps())
        acc_tmem = allocate_tensor_memory(ttgl.float32, [BLOCK, BLOCK], layout=tmem_layout)

        bar = ttgl.allocate_shared_memory(ttgl.int64, [1], ttgl.constexpr(mbarrier.MBarrierLayout()))
        mbarrier.init(bar, count=1)

        smem_b_T = smem_b.permute((1, 0))
        tcgen05_mma(smem_a, smem_b_T, acc_tmem, use_acc=False, pred=True, mbarriers=[bar])

        mbarrier.wait(bar, phase=0, deps=[smem_a, smem_b])
        mbarrier.invalidate(bar)

        out = acc_tmem.load(acc_reg_layout)
        out = ttgl.convert_layout(out, layout)
        ttgl.store(out_ptr + out_offs, out)

    rs = np.random.RandomState(0)
    a_bits = rs.randint(-(2**31), 2**31 - 1, size=(B, B), dtype=np.int32)
    #b_bits = rs.randint(-(2**31), 2**31 - 1, size=(B, B), dtype=np.int32)
    b_bits = np.eye(B, dtype=np.int32)
    exp_bits = _mm_payload_u32(a_bits, b_bits)

    a = torch.tensor(a_bits, device="cuda", dtype=torch.int32)
    b = torch.tensor(b_bits, device="cuda", dtype=torch.int32)
    out = torch.empty((B, B), device="cuda", dtype=torch.int32)

    aw = triton.TensorWrapper(a, dtype=torch.float32)
    bw = triton.TensorWrapper(b, dtype=torch.float32)
    outw = triton.TensorWrapper(out, dtype=torch.float32)

    kernel[(1, )](aw, bw, outw, fpsan=True, num_warps=4)

    np.testing.assert_array_equal(out.cpu().numpy().astype(np.int32, copy=False), exp_bits)
