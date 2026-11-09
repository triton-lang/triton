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
def test_tensor_descriptor_device_load_store(rank, dtype, large, store, with_allocator):

    @triton.jit
    def kernel(tensor_ptr, tile_ptr, shape, strides, BLOCK: tl.constexpr, STORE: tl.constexpr):
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
    if store:
        actual = torch.zeros_like(data)
        kernel[(1, )](actual, tile, actual.shape, actual.stride(), block, True)
        expected = torch.zeros_like(data)
        expected[region] = tile
    else:
        actual = torch.empty_like(tile)
        kernel[(1, )](data, actual, data.shape, data.stride(), block, False)
        expected = data[region]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@requires_tma
@pytest.mark.parametrize("stride", [16, 32])
def test_tensor_descriptor_alloc_class_span_threshold(stride, with_allocator):

    @triton.jit(do_not_specialize=["rows"])
    def kernel(src, dst, rows, STRIDE: tl.constexpr):
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
        compiled = kernel[(1, )](data, result, rows, stride)
        torch.testing.assert_close(result, data[-1], rtol=0, atol=0)
        assert len(allocations) == 1 and allocations[0].numel() == 128
        word2, = struct.unpack_from("<I", bytes(allocations.pop().cpu().tolist()), 8)
        version = re.search(r"(?m)^\.version (\d+)\.(\d+)$", compiled.asm["ptx"])
        assert version is not None
        ptx_version = tuple(map(int, version.groups()))
        expect_large = capability in (100, 103) and ptx_version >= (9, 5) and rows == threshold_rows
        assert bool(word2 & (1 << 21)) == expect_large
        hashes.add(compiled.hash)
    assert len(hashes) == 1
