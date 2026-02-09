"""Tests for TMEM double buffering API."""
import pytest
import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    tmem_double_buffer,
    get_tmem_reg_layout,
)


def is_blackwell():
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[0] >= 10


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell GPU")
def test_allocate_double_buffer():
    """Test single allocation with slice-based phase selection."""

    @gluon.jit
    def kernel(BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, num_warps: gl.constexpr):
        layout: gl.constexpr = TensorMemoryLayout([BLOCK_M, BLOCK_N], col_stride=1)
        db = tmem_double_buffer.allocate_double_buffer(gl.float32, BLOCK_M, BLOCK_N, layout)
        buf0 = db.get_buffer(0)
        buf1 = db.get_buffer(1)
        reg_layout: gl.constexpr = get_tmem_reg_layout(gl.float32, (BLOCK_M, BLOCK_N), layout, num_warps)
        _ = buf0.load(reg_layout)
        _ = buf1.load(reg_layout)

    kernel[(1, )](128, 128, num_warps=4)
    torch.cuda.synchronize()


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell GPU")
def test_allocate_double_buffer_pair():
    """Test two-allocation approach with index-based selection."""

    @gluon.jit
    def kernel(BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, num_warps: gl.constexpr):
        layout: gl.constexpr = TensorMemoryLayout([BLOCK_M, BLOCK_N], col_stride=1)
        pair = tmem_double_buffer.allocate_double_buffer_pair(gl.float32, BLOCK_M, BLOCK_N, layout)
        buf0 = pair.index(0)
        buf1 = pair.index(1)
        reg_layout: gl.constexpr = get_tmem_reg_layout(gl.float32, (BLOCK_M, BLOCK_N), layout, num_warps)
        _ = buf0.load(reg_layout)
        _ = buf1.load(reg_layout)

    kernel[(1, )](128, 128, num_warps=4)
    torch.cuda.synchronize()
