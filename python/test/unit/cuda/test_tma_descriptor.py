from contextlib import nullcontext
import pytest
import torch
import triton
import triton.language as tl
from triton._internal_testing import fp6_values, pack_fp6, is_cuda, is_blackwell_ultra
from triton.language.extra.cuda import to_fp6
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


@triton.jit
def _fp6_desc_store(src, dst, raw, desc, FORMAT: tl.constexpr, WIDTH: tl.constexpr, ROW: tl.constexpr):
    codes = to_fp6(tl.load(src + tl.arange(0, WIDTH)), FORMAT)
    tl.static_assert(codes.dtype == tl.uint8)
    tl.store(raw + tl.arange(0, WIDTH), codes)
    if desc is None:
        desc = tl.make_tensor_descriptor(dst, [3, ROW], [ROW * 3 // 4, 1], [1, WIDTH], fp6_format=FORMAT)
    desc.store([1, ROW - WIDTH], codes[None, :])


@pytest.mark.parametrize("fmt", ["e2m3", "e3m2"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("width,row,host", [(64, 64, False), (64, 128, False), (128, 128, False), (128, 128, True)])
def test_fp6_tma_store_padded_bytes(fmt, dtype, width, row, host, with_allocator):
    if not is_cuda() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("FP6 stores require SM100 or SM103")
    if width == 64 and not is_blackwell_ultra():
        pytest.skip("64-value FP6 stores require SM103")
    codes = torch.arange(width, device="cuda", dtype=torch.int32) % 64
    x = fp6_values(fmt, codes).to(dtype)
    # Check the halfway tie and the next representable input above it.
    midpoint = x[1] / 2
    x[0], x[2] = midpoint, torch.nextafter(midpoint, x[1])
    x[1], x[32] = 100, -0.0
    codes[1], codes[2] = 31, 1
    initial = torch.randint(0, 64, (3, row), device="cuda", dtype=torch.int32)
    buf = pack_fp6(initial)
    initial[1, row - width:] = codes
    desc = TensorDescriptor.from_tensor(buf, [1, width], fp6_format=fmt) if host else None
    raw = torch.empty(width, device="cuda", dtype=torch.uint8)
    kernel = _fp6_desc_store[(1, )](x, buf, raw, desc, fmt, width, row)
    torch.testing.assert_close(raw, codes.byte())
    torch.testing.assert_close(buf, pack_fp6(initial))
    assert f"cvt.rn.satfinite.{fmt}x2.f32" in kernel.asm["ptx"]
    assert "cp.async.bulk.tensor.2d.global.shared" in kernel.asm["ptx"]
    if host:
        assert kernel.metadata.global_scratch_size == 0


@pytest.mark.parametrize("host", [True, False], ids=["host", "device"])
def test_fp6_tma_store_64_rejects_sm100(host, capfd):
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    sig = {
        "src": "*fp32", "dst": "*u8", "raw": "*u8", "desc": "tensordesc<fp6e2m3[1,64]>" if host else "constexpr",
        "FORMAT": "constexpr", "WIDTH": "constexpr", "ROW": "constexpr"
    }
    constants = {"FORMAT": "e2m3", "WIDTH": 64, "ROW": 64}
    if not host:
        constants["desc"] = None
    with pytest.raises(RuntimeError):
        triton.compile(ASTSource(_fp6_desc_store, sig, constants), target=GPUTarget("cuda", 100, 32))
    assert "64-value FP6 TMA stores require SM103" in capfd.readouterr().err
