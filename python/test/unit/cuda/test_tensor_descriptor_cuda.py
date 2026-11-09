import ctypes
import math
import re
import struct

import pytest
import torch

import triton
import triton.language as tl
from triton._internal_testing import requires_tma
from triton.tools.tensor_descriptor import TensorDescriptor


@requires_tma
def test_specialization_after_host_tensordesc():

    @triton.jit
    def kernel(a, b):
        pass

    device = "cuda"
    A = torch.randn(1024, device=device)
    desc = TensorDescriptor.from_tensor(A, [128])
    h = kernel.warmup(desc, 16, grid=(1, ))
    assert "%a: !tt.tensordesc<128xf32>" in h.asm["ttir"]
    assert "%b: i32 {tt.divisibility = 16 : i32}" in h.asm["ttir"]


@requires_tma
@pytest.mark.parametrize("rank", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("dtype", [torch.uint8, torch.float16, torch.float32])
@pytest.mark.parametrize("large", [False, True])
@pytest.mark.parametrize("store", [False, True])
@pytest.mark.parametrize("static", [False, True])
def test_tensor_descriptor_device_load_store(rank, dtype, large, store, static, with_allocator):

    @triton.jit
    def kernel(tensor_ptr, tile_ptr, SHAPE: tl.constexpr, STRIDES: tl.constexpr, shape, strides, BLOCK: tl.constexpr,
               STORE: tl.constexpr):
        if SHAPE is not None:
            desc = tl.make_tensor_descriptor(tensor_ptr, SHAPE, STRIDES, BLOCK)
        else:
            desc = tl.make_tensor_descriptor(tensor_ptr, shape, strides, BLOCK)
        offsets = (1, ) * (len(BLOCK) - 1) + (0, )
        idx = tl.arange(0, 64)
        if STORE:
            desc.store(offsets, tl.reshape(tl.load(tile_ptr + idx), BLOCK))
        else:
            tl.store(tile_ptr + idx, tl.reshape(desc.load(offsets), (64, )))

    shape = [2] * (rank - 1) + [64]
    if large:
        shape[0] = 262144 if rank == 1 else 4096
    block = (1, ) * (rank - 1) + (64, )
    data = (torch.arange(math.prod(shape), device="cuda") % 251).to(dtype).reshape(shape)
    tile = (torch.arange(64, device="cuda") % 251).to(dtype)
    region = (1, ) * (rank - 1) + (slice(0, 64), )
    static_shape = tuple(shape) if static else None
    static_strides = data.stride() if static else None
    if store:
        actual = torch.zeros_like(data)
        kernel[(1, )](actual, tile, static_shape, static_strides, actual.shape, actual.stride(), block, True)
        expected = torch.zeros_like(data)
        expected[region] = tile
    else:
        actual = torch.empty_like(tile)
        kernel[(1, )](data, actual, static_shape, static_strides, data.shape, data.stride(), block, False)
        expected = data[region]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@requires_tma
@pytest.mark.parametrize("rank", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("dtype", [torch.uint8, torch.float16, torch.float32])
def test_tensor_descriptor_device_matches_host(rank, dtype, with_allocator):
    from triton.backends.nvidia.driver import TMA_DTYPE_DEVICE_TO_HOST

    @triton.jit
    def kernel(src, dst, shape, strides, BLOCK: tl.constexpr, STATIC: tl.constexpr, desc):
        if desc is None:
            if STATIC is None:
                desc = tl.make_tensor_descriptor(src, shape, strides, BLOCK)
            else:
                desc = tl.make_tensor_descriptor(src, STATIC[0], STATIC[1], BLOCK)
        tile = desc.load((0, ) * len(BLOCK))
        tl.store(dst + tl.arange(0, tile.numel), tile.reshape((tile.numel, )))

    capability = triton.runtime.driver.active.get_current_target().arch
    if capability not in (90, 100, 103):
        pytest.skip("Requires an architecture with packed TMA templates")

    shape = tuple(range(rank + 1, 2, -1)) + (32, )
    block = (2, ) * (rank - 1) + (16, )
    data = torch.zeros(shape, dtype=dtype, device="cuda")
    out = torch.empty(block, dtype=dtype, device="cuda")
    strides = data.stride()
    host_desc = TensorDescriptor.from_tensor(data, block)
    compiled = kernel.warmup(data, out, shape, strides, block, None, host_desc, grid=(1, ))
    metadata, = compiled.metadata.tensordesc_meta

    storage = ctypes.create_string_buffer(128 + 63)
    address = (ctypes.addressof(storage) + 63) & -64
    elem_size = data.element_size()
    encode = ctypes.CDLL("libcuda.so.1").cuTensorMapEncodeTiled
    # Request the same L2 promotion as the device path (none).
    status = encode(
        ctypes.c_void_p(address),
        TMA_DTYPE_DEVICE_TO_HOST[metadata["elem_type"]],
        rank,
        ctypes.c_void_p(data.data_ptr()),
        (ctypes.c_uint64 * rank)(*reversed(shape)),
        (ctypes.c_uint64 * (rank - 1))(*(stride * elem_size for stride in reversed(strides[:-1]))),
        (ctypes.c_uint32 * rank)(*reversed(metadata["block_size"])),
        (ctypes.c_uint32 * rank)(*([1] * rank)),
        0,
        metadata["swizzle"],
        0,
        0,
    )
    assert status == 0
    # CUDA's host encoder adds metadata after the 64-byte hardware descriptor.
    expected = ctypes.string_at(address, 64) + bytes(64)

    allocations = []

    def allocate(size, alignment, stream):
        buffer = torch.empty(size, device="cuda", dtype=torch.uint8)
        allocations.append(buffer)
        return buffer

    triton.set_allocator(allocate)
    for static in (None, (shape, strides)):
        kernel[(1, )](data, out, shape, strides, block, static, None)
        assert len(allocations) == 1
        assert bytes(allocations.pop().cpu().tolist()) == expected


@requires_tma
@pytest.mark.parametrize("stride", [16, 32])
def test_tensor_descriptor_alloc_class_span_threshold(stride, with_allocator):

    @triton.jit(do_not_specialize=["rows"])
    def kernel(src, dst, rows, STRIDE: tl.constexpr, ROWS: tl.constexpr):
        if ROWS is not None:
            rows = ROWS
        desc = tl.make_tensor_descriptor(src, [rows, 16], [STRIDE, 1], [1, 16])
        tile = desc.load([rows - 1, 0])
        tl.store(dst + tl.arange(0, 16), tile.reshape((16, )))

    allocations = []

    def allocate(size, alignment, stream):
        buffer = torch.empty(size, device="cuda", dtype=torch.uint8)
        allocations.append(buffer)
        return buffer

    triton.set_allocator(allocate)
    capability = triton.runtime.driver.active.get_current_target().arch
    hashes = set()
    threshold_rows = 1 + ((128 * 1024 - 16 + stride - 1) // stride)
    for rows in (threshold_rows - 1, threshold_rows):
        data = torch.empty_strided((rows, 16), (stride, 1), device="cuda", dtype=torch.uint8)
        data[-1] = torch.arange(16, device="cuda", dtype=torch.uint8)
        result = torch.empty_like(data[-1])
        compiled = kernel[(1, )](data, result, rows, stride, None)
        torch.testing.assert_close(result, data[-1], rtol=0, atol=0)
        assert len(allocations) == 1 and allocations[0].numel() == 128
        dynamic_desc = bytes(allocations.pop().cpu().tolist())
        if capability in (90, 100, 103):
            assert dynamic_desc[64:] == bytes(64)
        word2, = struct.unpack_from("<I", dynamic_desc, 8)
        version = re.search(r"(?m)^\.version (\d+)\.(\d+)$", compiled.asm["ptx"])
        assert version is not None
        ptx_version = tuple(map(int, version.groups()))
        expect_large = capability in (100, 103) and ptx_version >= (9, 5) and rows == threshold_rows
        assert bool(word2 & (1 << 21)) == expect_large
        hashes.add(compiled.hash)
        kernel[(1, )](data, result, rows, stride, rows)
        torch.testing.assert_close(result, data[-1], rtol=0, atol=0)
        assert len(allocations) == 1
        assert bytes(allocations.pop().cpu().tolist()) == dynamic_desc
    assert len(hashes) == 1
