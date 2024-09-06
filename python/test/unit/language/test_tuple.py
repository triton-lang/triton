import pytest
import triton
import triton.language as tl
import torch


@triton.jit
def _tuple_increment(values):
    for i in tl.static_range(len(values)):
        values[i] = values[i] + 1
    return values


@triton.jit
def _tuple_index_func(Ptrs, values):
    for i in tl.static_range(len(values)):
        tl.store(Ptrs[i], values[i])


@pytest.mark.parametrize("size", [0, 1, 2, 3, 4])
def test_index(size, device="cuda"):

    @triton.jit
    def kernel(_0, Ptrs, _1: tl.constexpr, values, _2, _3: tl.constexpr, _4):
        values = _tuple_increment(values)
        # _tuple_index_func(Ptrs, values)

    vals = tuple([i + 1 for i in range(size)])
    rets = tuple([torch.zeros((1, ), dtype=torch.float32, device=device) for _ in vals])
    kernel[(1, )](0, rets, 0, vals, 0, 0, 0)
    assert vals == tuple([x.item() + 1 for x in rets])


# function call (tuple argument)
# function call (tuple return value)
# tuple of tuples
# assignment
