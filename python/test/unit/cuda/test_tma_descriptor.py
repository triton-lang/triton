from contextlib import ExitStack, contextmanager, nullcontext
import ctypes
from types import SimpleNamespace
from math import prod
import pytest
import torch
import triton
import triton.language as tl
from triton._internal_testing import fp6_values, pack_fp6, is_blackwell_ultra, is_cuda
from triton.backends.nvidia.compiler import get_ptx_version_from_options
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
def _fp6_desc_store(src, dst, raw, desc, FORMAT: tl.constexpr, BLOCK_M: tl.constexpr, WIDTH: tl.constexpr = 128,
                    COL: tl.constexpr = 128):
    offsets = tl.arange(0, BLOCK_M * WIDTH)
    codes = to_fp6(tl.load(src + offsets), FORMAT)
    tl.static_assert(codes.dtype == tl.uint8)
    tl.store(raw + offsets, codes)
    if desc is None:
        desc = tl.make_tensor_descriptor(dst, [BLOCK_M + 2, 256], [192, 1], [BLOCK_M, WIDTH], fp6_format=FORMAT)
    coords: tl.constexpr = (1, ) * (len(desc.block_shape) - 1) + (COL, )
    desc.store(coords, codes.reshape(desc.block_shape))


@pytest.mark.parametrize("shape", ["2,64", "1,1,1,2,64"])
def test_fp6_tma_store_box64_sm100(capfd, shape):
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    signature = {"src": "*fp32", "dst": "*u8", "raw": "*u8", "desc": f"tensordesc<fp6e2m3[{shape}]>"}
    constants = {"FORMAT": "e2m3", "BLOCK_M": 2, "WIDTH": 64, "COL": 0}
    source = ASTSource(_fp6_desc_store, {**signature, **dict.fromkeys(constants, "constexpr")}, constants)
    with pytest.raises(RuntimeError, match="PassManager::run failed"):
        triton.compile(source, target=GPUTarget("cuda", 100, 32))
    assert "64-value FP6 TMA stores require SM103" in capfd.readouterr().err


@pytest.mark.parametrize("fmt", ["e2m3", "e3m2"])
@pytest.mark.parametrize("outer_tile,columns", [
    ((1, ), 64),
    ((1, ), 128),
    ((2, ), 64),
    ((2, ), 128),
    ((256, ), 128),
    ((2, 4), 64),
    ((2, 1, 4), 128),
    ((2, 1, 2, 4), 128),
])
def test_fp6_tma_store_host_box64(fmt, outer_tile, columns):
    if not is_blackwell_ultra():
        pytest.skip("64-value FP6 stores require SM103")

    row_bytes = columns * 6 // 8
    pitch = 48 if columns == 64 else 112
    outer_shape = tuple(n + 2 for n in outer_tile)
    storage = torch.full((32 + prod(outer_shape) * pitch, ), 0xa5, device="cuda", dtype=torch.uint8)
    dst = storage[16:-16].view(*outer_shape, pitch)[..., :row_bytes]
    desc = TensorDescriptor.from_tensor(dst, [*outer_tile, 64], fp6_format=fmt)
    rows = prod(outer_tile)
    fp6_bits = (torch.arange(rows, device="cuda")[:, None] * 7 + torch.arange(64, device="cuda")) % 64
    expected = storage.clone()
    tile = (slice(1, -1), ) * len(outer_tile) + (slice(row_bytes - 48, row_bytes), )
    expected[16:-16].view(*outer_shape, pitch)[tile] = pack_fp6(fp6_bits).reshape(*outer_tile, 48)
    raw = torch.empty_like(fp6_bits, dtype=torch.uint8)
    kernel = _fp6_desc_store[(1, )](fp6_values(fmt, fp6_bits), dst, raw, desc, fmt, rows, 64, columns - 64)
    torch.testing.assert_close(storage, expected)
    assert kernel.metadata.global_scratch_size == 0
    tile_type = "x".join(map(str, (*outer_tile, 64)))
    assert f"!tt.tensordesc<{tile_type}xui8," in kernel.asm["ttgir"]
    assert 'padding = "b6p2x16"' in kernel.asm["ttgir"]
    assert f"cp.async.bulk.tensor.{dst.ndim}d.global.shared::cta" in kernel.asm["ptx"]
    assert "tensormap.replace" not in kernel.asm["ptx"]


@pytest.mark.parametrize("block_m", [1, 2, 16, 128, 256])
@pytest.mark.parametrize("fmt", ["e2m3", "e3m2"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("host", [False, True], ids=["device", "host"])
def test_fp6_tma_store_padded_bytes(block_m, fmt, dtype, host, with_allocator):
    if not is_cuda() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("FP6 stores require SM100 or SM103")
    codes = torch.randint(0, 64, (block_m * 128, ), device="cuda", dtype=torch.int32)
    codes[:128] = torch.arange(128, device="cuda") % 64
    x = fp6_values(fmt, codes).to(dtype)
    # Check the halfway tie and the next representable input above it.
    midpoint = x[1] / 2
    x[0], x[2] = midpoint, torch.nextafter(midpoint, x[1])
    x[1], x[32] = 100, -0.0
    codes[1], codes[2] = 31, 1
    initial = torch.randint(0, 64, (block_m + 2, 256), device="cuda", dtype=torch.int32)
    buf = pack_fp6(initial)
    initial[1:-1, 128:] = codes.reshape(block_m, 128)
    desc = TensorDescriptor.from_tensor(buf, [block_m, 128], fp6_format=fmt) if host else None
    raw = torch.empty_like(codes, dtype=torch.uint8)
    kernel = _fp6_desc_store[(1, )](x, buf, raw, desc, fmt, block_m)
    torch.testing.assert_close(raw, codes.byte())
    torch.testing.assert_close(buf, pack_fp6(initial))
    assert f"!tt.tensordesc<{block_m}x128xf6{fmt.upper()}FN>" in kernel.asm["ttir"]
    assert f"!tt.tensordesc<{block_m}x128xui8," in kernel.asm["ttgir"]
    src = "f32"
    if get_ptx_version_from_options(kernel.metadata, kernel.metadata.target.arch) >= 91:
        src = {torch.float16: "f16x2", torch.bfloat16: "bf16x2", torch.float32: "f32"}[dtype]
    assert f"cvt.rn.satfinite.{fmt}x2.{src}" in kernel.asm["ptx"]
    assert "cp.async.bulk.tensor.2d.global.shared" in kernel.asm["ptx"]
    if host:
        assert kernel.metadata.global_scratch_size == 0


class _VmmLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _VmmAllocationProp(ctypes.Structure):
    # CUmemAllocationProp_v1 from cuda.h. All allocation flags stay zero.
    _fields_ = [
        ("type", ctypes.c_int),
        ("requestedHandleTypes", ctypes.c_int),
        ("location", _VmmLocation),
        ("win32HandleMetaData", ctypes.c_void_p),
        ("allocFlags", ctypes.c_ubyte * 8),
    ]


class _VmmAccessDesc(ctypes.Structure):
    _fields_ = [("location", _VmmLocation), ("flags", ctypes.c_int)]


@contextmanager
def _tensor_before_unmapped_guard(source):
    """Copy a packed activation immediately before an unmapped CUDA VMM page.

    Unlike a slice of a Torch allocation, this cannot borrow accessible padding
    from the caching allocator. No CUDA extension or cuda-python is required.
    The yielded tensor interface is borrowed and must not outlive this context.
    """
    assert source.is_cuda and source.dtype == torch.uint8 and source.is_contiguous()
    assert source.numel() > 0 and source.numel() % 16 == 0
    try:
        driver = ctypes.CDLL("libcuda.so.1")
    except OSError:
        pytest.skip("CUDA VMM guard test requires the Linux CUDA driver")

    ptr = ctypes.c_uint64
    size = ctypes.c_size_t
    pointer = ctypes.POINTER
    signatures = {
        "cuCtxGetDevice": [pointer(ctypes.c_int)],
        "cuDeviceGetAttribute": [pointer(ctypes.c_int), ctypes.c_int, ctypes.c_int],
        "cuMemGetAllocationGranularity": [pointer(size), pointer(_VmmAllocationProp), ctypes.c_int],
        "cuMemAddressReserve": [pointer(ptr), size, size, ptr, ctypes.c_uint64],
        "cuMemCreate": [pointer(ptr), size, pointer(_VmmAllocationProp), ctypes.c_uint64],
        "cuMemMap": [ptr, size, size, ptr, ctypes.c_uint64],
        "cuMemSetAccess": [ptr, size, pointer(_VmmAccessDesc), size],
        "cuMemcpyDtoDAsync_v2": [ptr, ptr, size, ctypes.c_void_p],
        "cuMemUnmap": [ptr, size],
        "cuMemRelease": [ptr],
        "cuMemAddressFree": [ptr, size],
    }
    for name, arguments in signatures.items():
        getattr(driver, name).argtypes = arguments
        getattr(driver, name).restype = ctypes.c_int

    def check(name, *args):
        result = getattr(driver, name)(*args)
        if result:
            raise RuntimeError(f"{name} failed with CUDA driver error {result}")

    with torch.cuda.device(source.device), ExitStack() as resources:

        def release_on_exit(function, *args):

            def release(exc_type, exc, traceback):
                try:
                    function(*args)
                except RuntimeError:
                    # Still release every resource without hiding the first CUDA error.
                    if exc_type is None:
                        raise

            resources.push(release)

        device_id, supported = ctypes.c_int(), ctypes.c_int()
        check("cuCtxGetDevice", ctypes.byref(device_id))
        # CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED = 102.
        check("cuDeviceGetAttribute", ctypes.byref(supported), 102, device_id)
        if not supported.value:
            pytest.skip("device does not support CUDA virtual memory management")
        # PINNED allocation, DEVICE location, no shareable handle requested.
        location = _VmmLocation(1, device_id.value)
        prop = _VmmAllocationProp(type=1, location=location)
        granularity = size()
        check("cuMemGetAllocationGranularity", ctypes.byref(granularity), ctypes.byref(prop), 0)
        mapped_bytes = triton.cdiv(source.numel(), granularity.value) * granularity.value
        reserved_bytes = mapped_bytes + granularity.value
        address, handle = ptr(), ptr()
        check("cuMemAddressReserve", ctypes.byref(address), reserved_bytes, 0, 0, 0)
        release_on_exit(check, "cuMemAddressFree", address, reserved_bytes)
        check("cuMemCreate", ctypes.byref(handle), mapped_bytes, ctypes.byref(prop), 0)
        release_on_exit(check, "cuMemRelease", handle)
        check("cuMemMap", address, mapped_bytes, 0, handle, 0)
        release_on_exit(check, "cuMemUnmap", address, mapped_bytes)
        access = _VmmAccessDesc(location, 3)  # CU_MEM_ACCESS_FLAGS_PROT_READWRITE.
        check("cuMemSetAccess", address, mapped_bytes, ctypes.byref(access), 1)
        tensor_address = address.value + mapped_bytes - source.numel()
        release_on_exit(torch.cuda.synchronize, source.device)
        check("cuMemcpyDtoDAsync_v2", tensor_address, source.data_ptr(), source.numel(),
              torch.cuda.current_stream(source.device).cuda_stream)
        yield SimpleNamespace(shape=source.shape, dtype=source.dtype, stride=source.stride,
                              data_ptr=lambda: tensor_address)


@triton.jit
def _load_ragged_boundary(X, Y, slice_off, slice_size, RANK: tl.constexpr):
    if RANK == 2:
        value = load_ragged(X, slice_off, slice_size, [0, 0])
    else:
        value = load_ragged(X, slice_off, slice_size, [0, 0, 0], ragged_dim=1)
    offsets = tl.arange(0, 128)[:, None] * 128 + tl.arange(0, 128)[None, :]
    tl.store(Y + offsets, value.reshape(128, 128))


@pytest.mark.parametrize("rows, rank, slice_off, slice_size", [(32, 2, 0, 1), (32, 3, 24, 8), (32, 2, 32, 0),
                                                               (64, 2, 48, 16)])
def test_ragged_tma_small_allocation_boundary(rows, rank, slice_off, slice_size):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("Test requires Hopper or Blackwell target.")
    # Match packed activation storage, independently of matmul or FP4 arithmetic.
    source = (torch.arange(rows * 2048, device="cuda") % 251 + 1).to(torch.uint8).reshape(rows, 2048)
    output = torch.empty((128, 128), device="cuda", dtype=torch.uint8)
    expected = torch.zeros_like(output)
    expected[:slice_size] = source[slice_off:slice_off + slice_size, :128]
    if rank == 3:
        source = source.as_strided((1, rows, 2048), (0, 2048, 1))
    with _tensor_before_unmapped_guard(source) as guarded:
        block_shape = [128, 128] if rank == 2 else [1, 128, 128]
        descriptor = create_ragged_descriptor(guarded, block_shape, ragged_dim=rank - 2)
        _load_ragged_boundary[(1, )](descriptor, output, slice_off, slice_size, rank)
        torch.cuda.synchronize()
        assert torch.equal(output, expected)


@pytest.mark.parametrize("rows, row_stride, contiguous_bytes", [
    (32, 2048, 65536),
    (64, 2048, 131072),
    (64, 4096, 2048),
])
def test_ragged_tma_contiguous_bytes(rows, row_stride, contiguous_bytes):
    # A virtual rank dimension and gaps between real rows cannot establish
    # that the corresponding bytes are accessible to the descriptor.
    storage = torch.empty((rows, row_stride), dtype=torch.uint8)
    tensor = storage[:, :2048].unsqueeze(0)
    desc = create_ragged_descriptor(tensor, [1, 128, 128], ragged_dim=1)
    assert desc.contiguous_bytes == contiguous_bytes
