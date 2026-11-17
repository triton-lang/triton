import struct
from itertools import product

import pytest
import torch

from triton._internal_testing import is_cuda
from triton.experimental.gsan import create_mem_pool
from triton.experimental.gsan._utils import uint8_cuda_tensor_from_ptr


@pytest.mark.parametrize("rank", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("high_word, expected", [
    (0, [0, 0x123456780, 0xFFFFFFFF0, 0xFFFFFFFF0]),
    (0x0F71, [0x1000000000, 0x7123456780, 0xFFFFFFFFF0, 0xFFFFFFFF0]),
])
def test_tma_descriptor_byte_strides(rank, high_word, expected):
    from triton._C.libtriton import gsan_testing

    # Native stride slots run from the innermost noncontiguous dimension out.
    # Exercise 64 GiB and the largest encodable stride, alongside small strides.
    words = [0] * 32
    words[3:8] = [0, 0x12345678, 0xFFFFFFFF, 0xFFFFFFFF, high_word]
    descriptor = struct.pack("<32I", *words)
    strides = gsan_testing.tensor_desc_byte_strides(descriptor, rank, 2)
    assert strides == list(reversed(expected[:rank - 1])) + [2]


@pytest.mark.parametrize("num_warps", [1, 4, 8])
@pytest.mark.parametrize("rank", [3, 4, 5])
def test_tma_im2col_access_ranges(rank, num_warps):
    from triton._C.libtriton import gsan_testing

    base = 0x10000
    words = [0] * 32
    words[0] = base
    words[2] = 1 | ((rank - 1) << 4) | (3 << 7)
    # N[2], spatial[3, ...], C[16], with a 16-byte gap between pixels.
    words[8:8 + rank] = [15] + [2] * (rank - 2) + [1]
    for dim in range(rank - 1):
        words[3 + dim] = 5 * 3**dim
    words[13] = (31 << 24) | sum(1 << (3 * dim) for dim in range(1, rank - 1))
    bits = {3: 16, 4: 8, 5: 5}[rank]
    lower = sum(((1 << bits) - 1) << (bits * dim) for dim in range(rank - 2))
    words[14] = lower | (lower << 16)
    words[15] = 15
    descriptor = struct.pack("<32I", *words)
    coords = [0] + [1] * (rank - 2) + [4]
    offsets = [1] * (rank - 2)

    ranges = []
    for warp in range(num_warps):
        ranges += gsan_testing.im2col_access_ranges(descriptor, coords, [16, 32], offsets, 4, warp, num_warps)

    # The first pixel is the final corner of image 0. The remaining in-bounds
    # pixels are every combination of spatial coordinates 0 and 2 in image 1.
    spatial_stride = [80 * 3**dim for dim in reversed(range(rank - 2))]
    batch_stride = 80 * 3**(rank - 2)
    first = base + 16 + sum(2 * stride for stride in spatial_stride)
    expected = [[first, 48]]
    expected += [[base + batch_stride + 16 + sum(coord * stride for coord, stride in zip(pixel, spatial_stride)), 48]
                 for pixel in product([0, 2], repeat=rank - 2)]
    assert sorted(ranges) == sorted(expected)


def test_tma_im2col_unsigned_offset():
    from triton._C.libtriton import gsan_testing

    # cuTensorMapEncodeIm2col output captured by PTXV's encoder probes, for
    # BF16 NWC[32, 32, 64]. Keep the native field encoding as a decoder fixture.
    descriptor = bytes.fromhex(
        "00100000007f0000210520000800000000010000000000000000000000000000"
        "3f0000001f0000001f00000000000000000000000000003f000000001f000000"
        "0010000000000000100000000000000000000000000000000000000000000000"
        "0000000000000000000000000000000000000000000000000000000000000000")
    words = list(struct.unpack("<32I", descriptor))
    words[9] = 65535
    words[14] = 0x80008000
    descriptor = struct.pack("<32I", *words)
    ranges = gsan_testing.im2col_access_ranges(descriptor, [1, -32768, 0], [2, 64], [-32768], 2, 0, 1)
    base = 0x7f0000001000 + 4096
    assert ranges == [[base, 128], [base + 128, 128]]


def test_tma_im2col_fp4_padded_access_ranges():
    from triton._C.libtriton import gsan_testing

    words = [0] * 32
    words[0] = 0x10000
    words[2] = 1 | (2 << 4) | (12 << 7) | (3 << 13)
    words[3:5] = [4, 8]
    words[8:11] = [127, 1, 0]
    words[13] = 127 << 24
    words[15] = 1
    descriptor = struct.pack("<32I", *words)
    ranges = gsan_testing.im2col_access_ranges(descriptor, [0, 0, 32], [2, 64], [0], 1, 0, 1)
    assert ranges == [[0x10020, 32], [0x10060, 32]]


def test_tma_im2col_captured_5d_descriptor():
    from triton._C.libtriton import gsan_testing

    # PTXV encoder capture for U8 NDHWC[1,16,16,16,64], with lower DHW
    # corners [-4,-3,-2] and upper DHW corners [-7,-6,-5].
    descriptor = bytes.fromhex(
        "0000010000000000416020000400000040000000000400000040000000000000"
        "3f0000000f0000000f0000000f000000000000000000007fbe735b677f000000"
        "0040000000000000000400000000000000000000000000000000000000000000"
        "0000000000000000000000000000000000000000000000000000000000000000")
    ranges = gsan_testing.im2col_access_ranges(descriptor, [0, 7, 9, 8, 0], [128, 128], [4, 3, 2], 1, 0, 1)
    assert len(ranges) == 128
    pixels = [(11, 12, 10), (11, 12, 11), (11, 12, 12)] + [(12, 0, w) for w in range(5)]
    assert ranges[:8] == [[0x10000 + d * 16384 + h * 1024 + w * 64, 64] for d, h, w in pixels]


def test_tma_im2col_fp6_padded_access_ranges():
    from triton._C.libtriton import gsan_testing

    words = [0] * 32
    words[0] = 0x10000
    words[2] = 1 | (2 << 4) | (13 << 7) | (3 << 13)
    words[3:5] = [12, 24]
    words[8:11] = [255, 1, 0]
    words[13] = 127 << 24
    words[15] = 1
    descriptor = struct.pack("<32I", *words)
    ranges = gsan_testing.im2col_access_ranges(descriptor, [0, 0, 128], [2, 128], [0], 1, 0, 1)
    assert ranges == [[0x10060, 96], [0x10120, 96]]


@pytest.mark.skipif(not is_cuda(), reason="requires CUDA backend")
def test_uint8_cuda_tensor_from_ptr_delete_tensor(shadow_granularity):
    device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
    pool = create_mem_pool(shadow_granularity=shadow_granularity)
    with torch.cuda.use_mem_pool(pool, device=device):
        backing = torch.arange(10, dtype=torch.uint8, device=device)
    view = uint8_cuda_tensor_from_ptr(backing.data_ptr(), backing.numel(), device.index)
    assert view.data_ptr() == backing.data_ptr()
    assert view.shape == (10, )
    assert view.dtype == torch.uint8
    assert view.device == device
    assert torch.equal(view, backing)


@pytest.mark.parametrize("nodes, entries, waits", [
    ([((), (), True), ((0, ), (), True), ((0, ), (), True),
      ((1, 2), (), True)], ((), (0, ), (0, ), (1, 2)), ((), (), (), ())),
    ([((), (), True), ((0, ), (), True), ((), (1, ), True),
      ((), (2, ), True)], ((), (0, ), (0, ), (0, )), ((), (), (1, ), (2, ))),
    ([((), (), True), ((0, ), (), False), ((1, ), (), False),
      ((2, ), (), True)], ((), (0, ), (0, ), (0, )), ((), (), (), ())),
    ([((), (), True), ((), (), True), ((0, ), (1, ), True),
      ((1, 2), (), True)], ((), (), (0, ), (1, 2)), ((), (), (1, ), ())),
    ([((), (), True), ((), (), True), ((0, 1), (), True),
      ((0, 1), (), True)], ((), (), (0, 1), (0, 1)), ((), (), (), ())),
])
def test_graph_dependency_frontiers(nodes, entries, waits):
    from triton.experimental.gsan import graph

    assert graph.dependency_frontiers([graph.GraphNode(*node) for node in nodes]) == (entries, waits)


@pytest.mark.parametrize("nodes", [
    [((0, ), (), True)],
    [((-1, ), (), True)],
    [((), (), False), ((), (0, ), True)],
    [((), (), True), ((), (0, ), False)],
])
def test_graph_dependency_frontiers_reject_invalid_edges(nodes):
    from triton.experimental.gsan import graph

    with pytest.raises(ValueError):
        graph.dependency_frontiers([graph.GraphNode(*node) for node in nodes])
