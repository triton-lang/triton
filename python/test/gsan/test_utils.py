import pytest
import torch

from triton._internal_testing import is_cuda
from triton.experimental.gsan._utils import uint8_cuda_tensor_from_ptr


@pytest.mark.skipif(not is_cuda(), reason="requires CUDA backend")
def test_uint8_cuda_tensor_from_ptr_delete_tensor():
    device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
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
