import struct

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
