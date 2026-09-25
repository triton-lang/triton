from __future__ import annotations

import pytest
import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from triton._internal_testing import is_blackwell, is_cuda, is_hopper_or_newer, is_sm12x
from triton.experimental.gsan import create_mem_pool
from triton.experimental.gsan._testing_utils import shadow_cell_from_address, shadow_tensor_for
from triton.tools.tensor_descriptor import TensorDescriptor
from test_gsan import (
    _device_tma_masked_load_kernel,
    _device_tma_masked_store_kernel,
    _gather_reference,
    _gluon_async_copy_masked_kernel,
    _host_tma_gather_kernel,
    _host_tma_scatter_kernel,
)

# These tests exercise write-once cell coverage, not the general ordering matrix.
pytestmark = [
    pytest.mark.skipif(not is_cuda(), reason="requires CUDA backend"),
    pytest.mark.parametrize("shadow_granularity", [1, 2, 4, 8, 16], indirect=True),
]


@pytest.fixture
def with_write_once_gsan(fresh_knobs, shadow_granularity):
    triton.knobs.compilation.instrumentation_mode = "gsan"
    pool = create_mem_pool(write_once=True, shadow_granularity=shadow_granularity)
    normal_pool = create_mem_pool()
    # Both pools must outlive the use_mem_pool context. PyTorch cannot destroy
    # an inner pool while the outer pool is still active.
    with torch.cuda.use_mem_pool(normal_pool):
        yield pool


@gluon.jit
def _write_once_mask_kernel(ptr, out, PHASE: gl.constexpr, STORE: gl.constexpr, BLOCK: gl.constexpr,
                            GROUP: gl.constexpr):
    offsets = gl.arange(0, BLOCK, layout=gl.BlockedLayout([GROUP], [32], [4], [0]))
    mask = offsets // GROUP % 4 == PHASE
    if STORE:
        gl.store(ptr + offsets, offsets + 1, mask)
    else:
        values = gl.load(ptr + offsets, mask, other=0)
        gl.store(out + offsets, values, mask)


@pytest.mark.parametrize("dtype", [torch.uint8, torch.int16, torch.int32, torch.int64])
def test_write_once_masked_bytes_and_repeated_reads(with_write_once_gsan, dtype, shadow_granularity):
    pool = with_write_once_gsan
    with torch.cuda.use_mem_pool(pool):
        target = torch.empty(128, device="cuda", dtype=dtype)
    out = torch.zeros_like(target)
    shadow = shadow_tensor_for(target)
    assert shadow.numel() == target.numel() * target.element_size() // shadow_granularity * 4
    group = max(1, shadow_granularity // target.element_size())
    written = torch.zeros(target.numel(), dtype=torch.bool)
    for phase in range(4):
        _write_once_mask_kernel[(1, )](target, out, phase, True, 128, group, num_warps=4)
        written.reshape(-1, group)[phase::4] = True
        clocks = shadow.cpu().view(torch.int32)
        expected = written.repeat_interleave(target.element_size()).reshape(-1, shadow_granularity).any(dim=1)
        assert torch.equal(clocks != 0, expected)
        for _ in range(2):
            _write_once_mask_kernel[(1, )](target, out, phase, False, 128, group, num_warps=4)
        assert torch.equal(shadow.cpu().view(torch.int32).reshape_as(clocks), clocks)
    torch.testing.assert_close(out, torch.arange(1, 129, device="cuda").to(dtype))


@gluon.jit
def _write_once_mixed_pointers_kernel(compact, normal, out, GRANULARITY: gl.constexpr):
    offsets = gl.arange(0, 128, layout=gl.BlockedLayout([GRANULARITY], [32], [4], [0]))
    ptrs = gl.where(offsets // GRANULARITY % 2 == 0, compact + offsets, normal + offsets)
    gl.store(ptrs, offsets)
    value = gl.load(ptrs)
    gl.store(out + offsets, value)


def test_write_once_mixed_pointer_tensor(with_write_once_gsan, shadow_granularity):
    pool = with_write_once_gsan
    with torch.cuda.use_mem_pool(pool):
        target = torch.empty(128, device="cuda", dtype=torch.uint8)
    normal = torch.empty_like(target)
    out = torch.empty_like(target)
    _write_once_mixed_pointers_kernel[(1, )](target, normal, out, shadow_granularity)
    torch.testing.assert_close(out, torch.arange(128, device="cuda", dtype=torch.uint8))
    clocks = shadow_tensor_for(target).cpu().view(torch.int32)
    assert torch.all(clocks[::2] != 0)
    assert torch.all(clocks[1::2] == 0)


def test_write_once_async_copy_reads_preserve_shadow(with_write_once_gsan, shadow_granularity):
    pool = with_write_once_gsan
    with torch.cuda.use_mem_pool(pool):
        target = torch.empty(128, dtype=torch.int32, device="cuda")
    out = torch.zeros_like(target)
    for phase in range(4):
        _write_once_mask_kernel[(1, )](target, out, phase, True, 128, max(1, shadow_granularity // 4))
    before = shadow_tensor_for(target).cpu()
    start, end = (4, 72) if shadow_granularity == 16 else (3, 73)
    _gluon_async_copy_masked_kernel[(1, )](out, target, end, start, BLOCK=128, num_warps=2)
    torch.testing.assert_close(out[start:end], target[start:end])
    assert torch.equal(shadow_tensor_for(target).cpu(), before)


@pytest.mark.skipif(not is_hopper_or_newer() or is_sm12x(), reason="Requires TMA")
@pytest.mark.parametrize("num_ctas", [1, 2])
def test_write_once_tma_partial_store_and_load(with_write_once_gsan, with_allocator, num_ctas, shadow_granularity):
    pool = with_write_once_gsan
    with torch.cuda.use_mem_pool(pool):
        storage = torch.zeros((40, 40), dtype=torch.int32, device="cuda")
    columns = 36 if shadow_granularity == 16 else 37
    target = storage[:35, :columns]
    out = torch.empty((32, 32), dtype=torch.int32, device="cuda")
    _device_tma_masked_store_kernel[(1, )](target, 35, columns, 30, 8, 40, BLOCK=32, num_ctas=num_ctas)
    shadow = shadow_tensor_for(storage).cpu()
    expected_written = torch.zeros((40, 40, 4), dtype=torch.bool)
    expected_written[30:35, 8:columns] = True
    expected_cells = expected_written.reshape(-1, shadow_granularity).any(dim=1)
    assert torch.equal(shadow.view(torch.int32) != 0, expected_cells)
    _device_tma_masked_load_kernel[(1, )](out, target, 35, columns, 30, 8, 40, BLOCK=32, num_ctas=num_ctas)
    expected = torch.zeros_like(out)
    expected[:5, :columns - 8] = 1
    torch.testing.assert_close(out, expected)
    assert torch.equal(shadow_tensor_for(storage).cpu(), shadow)


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell gather/scatter")
def test_write_once_indexed_descriptor_accesses(with_write_once_gsan, shadow_granularity):
    pool = with_write_once_gsan
    with torch.cuda.use_mem_pool(pool):
        storage = torch.zeros((16, 16), dtype=torch.int32, device="cuda")
    columns = 12 if shadow_granularity == 16 else 13
    target = storage[:11, :columns]
    desc = TensorDescriptor.from_tensor(target, [1, 8])
    indices = torch.tensor([1, 3, 5, 7, 9, 10, 11, 13], dtype=torch.int32, device="cuda")
    src = torch.arange(1, 65, dtype=torch.int32, device="cuda").reshape(8, 8)
    out = torch.empty_like(src)
    _host_tma_scatter_kernel[(1, )](desc, indices, 8, src, 8, 1, BLOCK_X=8)
    before = shadow_tensor_for(storage).cpu()
    expected_written = torch.zeros((16, 16, 4), dtype=torch.bool)
    expected_written[[1, 3, 5, 7, 9, 10], 8:columns] = True
    expected_cells = expected_written.reshape(-1, shadow_granularity).any(dim=1)
    assert torch.equal(before.view(torch.int32) != 0, expected_cells)
    _host_tma_gather_kernel[(1, )](out, 8, 1, desc, indices, 8, BLOCK_X=8)
    torch.testing.assert_close(out, _gather_reference(target, indices, 8, 8))
    assert torch.equal(shadow_tensor_for(storage).cpu(), before)


@triton.jit
def _write_once_masked_atomics_kernel(ptr, active):
    tl.atomic_store(ptr, 1, sem="relaxed", mask=active)
    tl.atomic_load(ptr, sem="relaxed", mask=active)
    tl.atomic_add(ptr, 1, sem="relaxed", mask=active)


def test_write_once_masked_off_atomics(with_write_once_gsan):
    pool = with_write_once_gsan
    with torch.cuda.use_mem_pool(pool):
        target = torch.empty(1, dtype=torch.int32, device="cuda")
    _write_once_masked_atomics_kernel[(1, )](target, False)
    assert torch.count_nonzero(shadow_tensor_for(target)).item() == 0


@gluon.jit
def _write_once_adjacent_cells_kernel(ptr, GRANULARITY: gl.constexpr):
    pid = gl.program_id(0)
    offsets = gl.arange(0, GRANULARITY, layout=gl.BlockedLayout([GRANULARITY], [32], [1], [0]))
    gl.store(ptr + pid * GRANULARITY + offsets, pid + 1)


def test_write_once_adjacent_cta_writes(with_write_once_gsan, shadow_granularity):
    pool = with_write_once_gsan
    with torch.cuda.use_mem_pool(pool):
        target = torch.empty(4 * shadow_granularity, dtype=torch.uint8, device="cuda")
    _write_once_adjacent_cells_kernel[(4, )](target, shadow_granularity, num_warps=1)
    expected = torch.arange(1, 5, dtype=torch.uint8, device="cuda").repeat_interleave(shadow_granularity)
    torch.testing.assert_close(target, expected)
    for cell in range(4):
        assert shadow_cell_from_address(target.data_ptr() + cell * shadow_granularity).epoch > 0
