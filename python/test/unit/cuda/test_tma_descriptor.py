from contextlib import nullcontext
import pytest
import torch
import triton
import triton.language as tl
from triton.tools.ragged_tma import create_ragged_descriptor, atomic_add_ragged, load_ragged, store_ragged
from triton.tools.tensor_descriptor import TensorDescriptor


@pytest.mark.parametrize("M, BLOCK_M, expect_error", [(128, 32, False), (127, 32, False), (128, 31, True)])
def test_1d_tma_descriptor_exception(M, BLOCK_M, expect_error):
    if not torch.cuda.is_available() or not torch.cuda.get_device_capability()[0] >= 9:
        pytest.skip("Test requires Hopper or Blackwell target.")
        return

    device = "cuda"
    x = torch.randn(M, dtype=torch.float32, device=device)
    # globalAddress in the tma descriptor must be aligned to 16 bytes for CU_TENSOR_MAP_INTERLEAVE_NONE.
    # https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html#group__CUDA__TENSOR__MEMORY
    assert x.data_ptr() % 16 == 0

    ctx = pytest.raises(ValueError, match="Shape element 0 must be a power of 2") if expect_error else nullcontext()
    with ctx:
        _ = TensorDescriptor.from_tensor(x, [BLOCK_M])


@pytest.mark.parametrize("M, BLOCK_M, expect_error_m", [(128, 32, False), (125, 33, True), (0, 32, False)])
@pytest.mark.parametrize("N, BLOCK_N, expect_error_n", [(128, 32, False), (128, 30, True), (127, 32, False)])
def test_2d_tma_descriptor_exception(M, N, BLOCK_M, BLOCK_N, expect_error_n, expect_error_m):
    if not torch.cuda.is_available() or not torch.cuda.get_device_capability()[0] >= 9:
        pytest.skip("Test requires Hopper or Blackwell target.")
        return

    device = "cuda"
    torch.manual_seed(42)
    A = torch.randn((M, N), dtype=torch.float16, device=device)
    # globalAddress in the tma descriptor must be aligned to 16 bytes for CU_TENSOR_MAP_INTERLEAVE_NONE.
    # https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html#group__CUDA__TENSOR__MEMORY
    assert A.data_ptr() % 16 == 0

    shape_error = expect_error_n or expect_error_m
    error_alignment = (N % 16) != 0
    zero_shape_error = M <= 0 or N <= 0
    expect_error = shape_error or error_alignment or zero_shape_error

    exc_type = ValueError if shape_error else AssertionError
    match = "Shape element . must be a power of 2" if shape_error else "strides must be 16-byte aligned"
    if zero_shape_error and not shape_error and not error_alignment:
        match = "shape must be positive"
        exc_type = AssertionError
    ctx = pytest.raises(exc_type, match=match) if expect_error else nullcontext()
    with ctx:
        _ = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_N])


@triton.jit
def example_load_store_kernel(X, Y, x_off, y_off, x_size, y_size):

    data = load_ragged(X, x_off, x_size, [0, 0])
    store_ragged(Y, y_off, y_size, [0, 0], data)


@triton.jit
def example_load_atomic_add_kernel(X, Y, x_off, y_off, x_size, y_size):

    data = load_ragged(X, x_off, x_size, [0, 0])
    atomic_add_ragged(Y, y_off, y_size, [0, 0], data)


@pytest.mark.parametrize("dtype", [
    "bfloat16", "float16", "float32", "float64",  # floating-point
    "int8", "int16", "int32", "int64",  # signed integers
    "uint8", "uint16", "uint32", "uint64"  # unsigned integers
])
def test_ragged_tma(dtype):

    if not torch.cuda.is_available() or not torch.cuda.get_device_capability()[0] >= 9:
        pytest.skip("Test requires Hopper or Blackwell target.")
        return

    test_atomic_add = dtype in ["bfloat16", "float16", "float32", "int32"]
    dtype = getattr(torch, dtype)

    src1 = torch.randn((1024, 80), dtype=torch.float32, device="cuda").to(dtype)
    src2 = torch.randn((1024, 80), dtype=torch.float32, device="cuda").to(dtype)
    ref = torch.randn((1024, 80), dtype=torch.float32, device="cuda").to(dtype)
    dst = ref.clone()

    X1 = create_ragged_descriptor(src1, [32, 128])
    X2 = create_ragged_descriptor(src2, [32, 128])
    Y = create_ragged_descriptor(dst, [32, 128])

    x_off = 42
    y_off = 51
    x_size = 17
    y_size = 24

    example_load_store_kernel[(1, )](X1, Y, x_off, y_off, x_size, y_size)
    if test_atomic_add:
        example_load_atomic_add_kernel[(1, )](X2, Y, x_off, y_off, x_size, y_size)

    # the initial and final segments are unchanged:
    res0 = torch.equal(dst[:y_off], ref[:y_off])
    res1 = torch.equal(dst[y_off + y_size:], ref[y_off + y_size:])

    # this segment will be copied verbatim from src:
    ref_tensor = src1 + src2 if test_atomic_add else src1
    res2 = torch.equal(dst[y_off:y_off + x_size], ref_tensor[x_off:x_off + x_size])

    # this segment will have read OOB zeroes and written them here:
    res3 = torch.all(dst[y_off + x_size:y_off + y_size] == 0.0).item()

    assert [res0, res1, res2, res3] == [True, True, True, True]


def _fp6_values(fmt, codes):
    mantissa, bias = (3, 1) if fmt == "e2m3" else (2, 3)
    exp = (codes & 31) >> mantissa
    fraction = (codes & ((1 << mantissa) - 1)).float() / (1 << mantissa)
    return torch.ldexp(fraction + (exp != 0), exp.clamp_min(1) - bias) * (1 - 2 * (codes >> 5))


def _pack_fp6(codes):
    groups = codes.reshape(*codes.shape[:-1], -1, 4)
    words = groups[..., 0] | groups[..., 1] << 6 | groups[..., 2] << 12 | groups[..., 3] << 18
    return torch.stack([words, words >> 8, words >> 16], -1).byte().flatten(-2)


@triton.jit
def _fp6_desc_store(src, dst, desc, DTYPE: tl.constexpr, WIDTH: tl.constexpr, HOST: tl.constexpr,
                    ROW: tl.constexpr = 128):
    x = tl.load(src + tl.arange(0, WIDTH)).to(DTYPE)
    if not HOST:
        desc = tl.make_tensor_descriptor(dst, [3, ROW], [ROW * 3 // 4, 1], [1, WIDTH], dtype=DTYPE)
    desc.store([1, ROW - WIDTH], x[None, :])


@triton.jit
def _fp6_desc_load(src, desc, dst, DTYPE: tl.constexpr, HOST: tl.constexpr):
    if not HOST:
        desc = tl.make_tensor_descriptor(src, [3, 128], [96, 1], [1, 128], dtype=DTYPE)
    x = desc.load([1, 0])
    tl.static_assert(x.dtype == DTYPE)
    tl.store(dst + tl.arange(0, 128), tl.reshape(x.to(tl.float32), (128, )))


@pytest.mark.parametrize("fmt", ["e2m3", "e3m2"])
@pytest.mark.parametrize("width,host", [(64, False), (128, False), (128, True)])
def test_fp6_tma_roundtrip(fmt, width, host, with_allocator):
    sm = torch.cuda.get_device_capability() if torch.cuda.is_available() else None
    if sm not in ((10, 0), (10, 3)) or (width == 64 and sm != (10, 3)):
        pytest.skip("FP6 requires SM100 or SM103, and 64-value stores require SM103")
    dtype = tl.float6e2m3 if fmt == "e2m3" else tl.float6e3m2
    codes = torch.arange(width, device="cuda", dtype=torch.int32) % 64
    x = _fp6_values(fmt, codes)
    # Include an exact midpoint, saturation and a signed zero.
    x[0], codes[0] = x[1].item() / 2, 0
    x[1], codes[1] = 100, 31
    x[32] = -0.0
    initial = torch.randint(0, 64, (3, 128), device="cuda", dtype=torch.int32)
    buf = _pack_fp6(initial)
    initial[1, 128 - width:] = codes
    desc = TensorDescriptor.from_tensor(buf, [1, 128], dtype=dtype) if host else None
    store = _fp6_desc_store[(1, )](x, buf, desc, dtype, width, host)
    torch.testing.assert_close(buf, _pack_fp6(initial))
    out = torch.empty(128, dtype=torch.float32, device="cuda")
    load = _fp6_desc_load[(1, )](buf, desc, out, dtype, host)
    expected = _fp6_values(fmt, initial[1])
    torch.testing.assert_close(out, expected, atol=0, rtol=0)
    assert torch.signbit(out[128 - width + 32])
    assert "cp.async.bulk.tensor.2d.global.shared" in store.asm["ptx"]
    assert "cp.async.bulk.tensor.2d.shared" in load.asm["ptx"]


@triton.jit
def _fp6_desc_mma(a_desc, b_desc, sa, sb, out, FORMAT: tl.constexpr, TRANSPOSE_B: tl.constexpr,
                  REGISTER_B: tl.constexpr, DTYPE: tl.constexpr, A_FORMAT: tl.constexpr = "e4m3",
                  N: tl.constexpr = 128):
    a = a_desc.load([0, 0])
    b = b_desc.load([0, 0])
    if REGISTER_B:
        b = b.to(DTYPE)
    if TRANSPOSE_B:
        b = b.T
    r = tl.arange(0, 128)
    offsets = r[:, None] * 4 + tl.arange(0, 4)[None, :]
    cols = tl.arange(0, N)
    b_scale = None if sb is None else tl.load(sb + cols[:, None] * 4 + tl.arange(0, 4)[None, :])
    c = tl.dot_scaled(a, tl.load(sa + offsets), A_FORMAT, b, b_scale, FORMAT)
    tl.store(out + r[:, None] * N + cols[None, :], c)


@pytest.mark.parametrize("fmt", ["e2m3", "e3m2"])
@pytest.mark.parametrize("transpose_b", [False, True])
@pytest.mark.parametrize("register_b,scaled_b,n", [(False, True, 128), (False, False, 128), (True, True, 128),
                                                   (True, True, 32)],
                         ids=["tma_mxfp6", "tma_fp6", "cast_mxfp6", "cast_n32"])
def test_fp6_tma_dot_scaled(fmt, transpose_b, register_b, scaled_b, n):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("FP6 requires SM100 or SM103")
    dtype = tl.float6e2m3 if fmt == "e2m3" else tl.float6e3m2
    a = torch.randint(-2, 3, (128, 128), device="cuda").to(torch.float8_e4m3fn)
    shape = (n, 128) if transpose_b else (128, n)
    codes = torch.randint(0, 64, shape, device="cuda", dtype=torch.int32)
    b = _fp6_values(fmt, codes)
    sa, sb = (torch.randint(126, 129, (rows, 4), device="cuda", dtype=torch.uint8) for rows in (128, n))
    a_desc = TensorDescriptor.from_tensor(a, [128, 128])
    b_desc = (TensorDescriptor.from_tensor(b, list(shape)) if register_b else TensorDescriptor.from_tensor(
        _pack_fp6(codes), list(shape), dtype=dtype))
    out = torch.empty((128, n), device="cuda", dtype=torch.float32)
    kernel = _fp6_desc_mma[(1, )](a_desc, b_desc, sa, sb if scaled_b else None, out, fmt, transpose_b, register_b,
                                  dtype, N=n)
    a = a.float() * torch.exp2(sa.float() - 127).repeat_interleave(32, 1)
    b = b.T if transpose_b else b
    if scaled_b:
        b = b * torch.exp2(sb.float() - 127).repeat_interleave(32, 1).T
    torch.testing.assert_close(out, a @ b, atol=0, rtol=0)
    assert "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.block32" in kernel.asm["ptx"]
    if not register_b:
        assert not any("ttg.local_load" in line and "f6E" in line for line in kernel.asm["ttgir"].splitlines())


@pytest.mark.parametrize("fmt", ["e2m3", "e3m2"])
def test_fp6_tma_dot_scaled_lhs(fmt):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("FP6 requires SM100 or SM103")
    dtype = tl.float6e2m3 if fmt == "e2m3" else tl.float6e3m2
    codes = torch.randint(0, 64, (128, 128), device="cuda", dtype=torch.int32)
    a = _fp6_values(fmt, codes)
    b = torch.randint(-2, 3, (128, 128), device="cuda").to(torch.float8_e4m3fn)
    sa, sb = (torch.randint(126, 129, (128, 4), device="cuda", dtype=torch.uint8) for _ in range(2))
    out = torch.empty_like(a)
    kernel = _fp6_desc_mma[(1, )](TensorDescriptor.from_tensor(_pack_fp6(codes), [128, 128], dtype=dtype),
                                  TensorDescriptor.from_tensor(b, [128, 128]), sa, sb, out, "e4m3", False, False, dtype,
                                  fmt)
    a = a * torch.exp2(sa.float() - 127).repeat_interleave(32, 1)
    b = b.float() * torch.exp2(sb.float() - 127).repeat_interleave(32, 1).T
    torch.testing.assert_close(out, a @ b, atol=0, rtol=0)
    assert "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.block32" in kernel.asm["ptx"]
    assert not any("ttg.local_load" in line and "f6E" in line for line in kernel.asm["ttgir"].splitlines())


@triton.jit
def _fp6_pipelined_mma(a_desc, b_desc, scales, out, steps):
    r = tl.arange(0, 128)
    s = tl.load(scales + r[:, None] * 4 + tl.arange(0, 4)[None, :])
    acc = tl.full((128, 128), 0, tl.float32)
    for i in range(steps):
        a = a_desc.load([0, i * 128])
        b = b_desc.load([i * 128, 0])
        acc = tl.dot_scaled(a, s, "e2m3", b, s, "e3m2", acc)
    tl.store(out + r[:, None] * 128 + r[None, :], acc)


def test_fp6_tma_dot_scaled_pipeline():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("FP6 requires SM100 or SM103")
    steps = 4
    a = torch.randint(0, 64, (128, steps * 128), device="cuda", dtype=torch.int32)
    b = torch.randint(0, 64, (steps * 128, 128), device="cuda", dtype=torch.int32)
    scales = torch.full((128, 4), 127, device="cuda", dtype=torch.uint8)
    out = torch.empty((128, 128), device="cuda")
    kernel = _fp6_pipelined_mma[(1, )](TensorDescriptor.from_tensor(_pack_fp6(a), [128, 128], dtype=tl.float6e2m3),
                                       TensorDescriptor.from_tensor(_pack_fp6(b), [128, 128], dtype=tl.float6e3m2),
                                       scales, out, steps, num_stages=3)
    torch.testing.assert_close(out, _fp6_values("e2m3", a) @ _fp6_values("e3m2", b), rtol=0, atol=0)
    assert "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.block32" in kernel.asm["ptx"]
    assert not any("ttg.local_load" in line and "f6E" in line for line in kernel.asm["ttgir"].splitlines())


@triton.jit
def _copy_fp6(desc, DTYPE: tl.constexpr):
    desc.store([1, 0], desc.load([0, 0]))
    desc.store([2, 0], tl.full((1, 128), -0.0, DTYPE))


@pytest.mark.parametrize("fmt", ["e2m3", "e3m2"])
def test_fp6_tma_load_then_store(fmt):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("FP6 requires SM100 or SM103")
    dtype = tl.float6e2m3 if fmt == "e2m3" else tl.float6e3m2
    codes = torch.randint(0, 64, (3, 128), device="cuda", dtype=torch.int32)
    buf = _pack_fp6(codes)
    _copy_fp6[(1, )](TensorDescriptor.from_tensor(buf, [1, 128], dtype=dtype), dtype)
    codes[1] = codes[0]
    codes[2] = 32
    torch.testing.assert_close(buf, _pack_fp6(codes), rtol=0, atol=0)


def test_fp6_tma_store_48_byte_stride(with_allocator):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("48-byte FP6 rows require SM103")
    codes = torch.arange(64, device="cuda", dtype=torch.int32)
    buf = torch.full((3, 48), 0xff, device="cuda", dtype=torch.uint8)
    _fp6_desc_store[(1, )](_fp6_values("e2m3", codes), buf, None, tl.float6e2m3, 64, False, 64)
    expected = torch.full_like(buf, 0xff)
    expected[1] = _pack_fp6(codes)
    torch.testing.assert_close(buf, expected, rtol=0, atol=0)
