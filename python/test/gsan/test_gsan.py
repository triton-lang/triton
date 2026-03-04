from __future__ import annotations

import pytest
import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import fence_async_shared, mbarrier, tma
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor as GluonTensorDescriptor

from triton._internal_testing import is_blackwell, is_cuda
from triton.experimental.gsan import create_mem_pool
from triton._C.libtriton.gsan_testing import ScalarClock, AtomicScope
from triton.experimental.gsan._testing_utils import (load_one_i32, shadow_cell_from_address, store_one_i32,
                                                     thread_state_from_smid)


@pytest.fixture()
def with_gsan(fresh_knobs):
    triton.knobs.compilation.instrumentation_mode = "gsan"
    pool = create_mem_pool()
    with torch.cuda.use_mem_pool(pool):
        yield


@pytest.mark.skipif(not is_cuda(), reason="GSan requires CUDA")
def test_load_store_updates_shadow(with_gsan):
    target = torch.zeros(1, dtype=torch.int32, device="cuda")
    scratch = torch.zeros(1, dtype=torch.int32, device="cuda")

    store_one_i32[(1, )](target, num_warps=1)
    cell0 = shadow_cell_from_address(target.data_ptr())

    tid = cell0.write_clock.thread_id
    epoch0 = thread_state_from_smid(tid).vector_clock[tid]

    assert cell0.write_clock.thread_id == tid
    assert cell0.write_clock.epoch == epoch0
    assert cell0.read_clocks[0].thread_id == 0
    assert cell0.read_clocks[0].epoch == 0
    assert cell0.num_reads == 0

    load_one_i32[(1, )](target, scratch, num_warps=1)
    cell1 = shadow_cell_from_address(target.data_ptr())
    epoch1 = thread_state_from_smid(tid).vector_clock[tid]

    assert epoch1 == epoch0 + 1
    assert cell1.write_clock == cell0.write_clock
    assert cell1.read_clocks[0] == ScalarClock(epoch1, tid, AtomicScope.NON_ATOMIC)
    # Scalar accesses are instrumented once via the redundant-thread predicate.
    assert cell1.num_reads == 1


@triton.jit
def _write_blocks_kernel(ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.store(ptr + offsets, 1, mask=mask)


@triton.jit
def _read_reversed_blocks_kernel(ptr, scratch_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    src_pid = tl.num_programs(0) - 1 - pid
    src_offsets = src_pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    dst_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = dst_offsets < n_elements
    value = tl.load(ptr + src_offsets, mask=mask)
    tl.store(scratch_ptr + dst_offsets, value, mask=mask)


@pytest.mark.skipif(not is_cuda(), reason="GSan requires CUDA")
def test_implicit_stream_ordering(with_gsan, capfd):
    block_size = 128
    size = block_size * 1024
    target = torch.zeros(size, dtype=torch.int32, device="cuda")
    scratch = torch.zeros(size, dtype=torch.int32, device="cuda")

    grid = (triton.cdiv(size, block_size), )
    _write_blocks_kernel[grid](target, size, BLOCK_SIZE=block_size)
    _read_reversed_blocks_kernel[grid](target, scratch, size, BLOCK_SIZE=block_size)
    torch.cuda.synchronize()

    assert scratch.sum().item() == size
    assert "GSanLibrary.cu" not in capfd.readouterr()


@triton.jit
def _device_tma_masked_store_kernel(ptr, m_size, n_size, row_idx, col_idx, stride_0, BLOCK: tl.constexpr):
    desc = tl.make_tensor_descriptor(ptr, [m_size, n_size], [stride_0, 1], [BLOCK, BLOCK])
    values = tl.full((BLOCK, BLOCK), 1, dtype=tl.int32)
    desc.store([row_idx, col_idx], values)


@gluon.jit
def _host_tma_gather_kernel(out_ptr, out_stride_0, out_stride_1, desc, x_offsets_ptr, y_offset, BLOCK_X: gl.constexpr):
    BLOCK_Y: gl.constexpr = desc.block_type.shape[1]
    coalesced_1d_layout: gl.constexpr = gl.BlockedLayout([1], [32], [gl.num_warps()], [0])
    x_offsets = gl.load(x_offsets_ptr + gl.arange(0, BLOCK_X, coalesced_1d_layout))
    offsets_layout: gl.constexpr = gl.SliceLayout(0, gl.BlockedLayout([1, 4], [32, 1], [1, gl.num_warps()], [1, 0]))
    x_offsets = gl.convert_layout(x_offsets, offsets_layout)
    smem_dest = gl.allocate_shared_memory(desc.dtype, [BLOCK_X, BLOCK_Y], desc.layout)
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    mbarrier.expect(bar, BLOCK_X * desc.block_type.nbytes)
    tma.async_gather(desc, x_offsets, y_offset, barrier=bar, result=smem_dest)
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)

    coalesced_2d_layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, gl.num_warps()], [1, 0])
    out = smem_dest.load(coalesced_2d_layout)
    indices_x = gl.arange(0, BLOCK_X, gl.SliceLayout(1, coalesced_2d_layout))[:, None] * out_stride_0
    indices_y = gl.arange(0, BLOCK_Y, gl.SliceLayout(0, coalesced_2d_layout))[None, :] * out_stride_1
    gl.store(out_ptr + indices_x + indices_y, out)


@gluon.jit
def _host_tma_scatter_kernel(desc, x_offsets_ptr, y_offset, src_ptr, src_stride_0, src_stride_1, BLOCK_X: gl.constexpr):
    BLOCK_Y: gl.constexpr = desc.block_type.shape[1]
    coalesced_2d_layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, gl.num_warps()], [1, 0])
    indices_x = gl.arange(0, BLOCK_X, gl.SliceLayout(1, coalesced_2d_layout))[:, None] * src_stride_0
    indices_y = gl.arange(0, BLOCK_Y, gl.SliceLayout(0, coalesced_2d_layout))[None, :] * src_stride_1
    src = gl.load(src_ptr + indices_x + indices_y)

    coalesced_1d_layout: gl.constexpr = gl.BlockedLayout([1], [32], [gl.num_warps()], [0])
    x_offsets = gl.load(x_offsets_ptr + gl.arange(0, BLOCK_X, coalesced_1d_layout))
    offsets_layout: gl.constexpr = gl.SliceLayout(0, gl.BlockedLayout([1, 4], [32, 1], [1, gl.num_warps()], [1, 0]))
    x_offsets = gl.convert_layout(x_offsets, offsets_layout)

    smem_src = gl.allocate_shared_memory(desc.dtype, [BLOCK_X, BLOCK_Y], desc.layout)
    smem_src.store(src)
    fence_async_shared()
    tma.async_scatter(desc, x_offsets, y_offset, smem_src)
    tma.store_wait(0)


def _assert_ttgir_contains(kernel, *ops):
    ttgir = kernel.asm["ttgir"]
    for op in ops:
        assert op in ttgir, ttgir


def _tensor_storage_address(tensor: torch.Tensor, row_idx: int, storage_col_idx: int) -> int:
    offset = row_idx * tensor.stride(0) + storage_col_idx * tensor.stride(1)
    return tensor.data_ptr() + offset * tensor.element_size()


def _shadow_cell_state(cell) -> tuple[int, object, tuple[object, ...]]:
    return (cell.num_reads, cell.write_clock, tuple(cell.read_clocks))


def _gather_reference(target: torch.Tensor, x_offsets: torch.Tensor, y_offset: int, block_y: int) -> torch.Tensor:
    result = torch.zeros((x_offsets.numel(), block_y), dtype=target.dtype, device=target.device)
    valid_rows = (x_offsets >= 0) & (x_offsets < target.shape[0])
    valid_cols = max(min(target.shape[1] - y_offset, block_y), 0)
    if valid_cols == 0:
        return result

    safe_rows = torch.where(valid_rows, x_offsets, 0)
    gathered = target[safe_rows.long(), y_offset:y_offset + valid_cols]
    result[:, :valid_cols] = gathered * valid_rows[:, None]
    return result


def _scatter_reference(dst: torch.Tensor, src: torch.Tensor, x_offsets: torch.Tensor, y_offset: int) -> torch.Tensor:
    result = torch.zeros_like(dst)
    valid_cols = max(min(dst.shape[1] - y_offset, src.shape[1]), 0)
    if valid_cols == 0:
        return result

    for src_row, dst_row in enumerate(x_offsets.tolist()):
        if 0 <= dst_row < dst.shape[0]:
            result[dst_row, y_offset:y_offset + valid_cols] = src[src_row, :valid_cols]
    return result


def _make_gather_scatter_descriptor(tensor: torch.Tensor, block_x: int, block_y: int) -> GluonTensorDescriptor:
    gl_dtype = getattr(gl, str(tensor.dtype).split(".")[1])
    layout = gl.NVMMASharedLayout.get_default_for([block_x, block_y], gl_dtype)
    return GluonTensorDescriptor.from_tensor(tensor, [1, block_y], layout)


@pytest.mark.skipif(not is_cuda(), reason="GSan requires CUDA")
def test_tma_masked_store_updates_shadow(with_gsan, with_allocator, capfd):
    block = 32
    m_size = 35
    n_size = 37
    padded_m = 40
    padded_n = 40
    row_idx = 5
    col_idx = 8
    valid_rows = m_size - row_idx
    valid_cols = n_size - col_idx
    target_storage = torch.zeros((padded_m, padded_n), dtype=torch.int32, device="cuda")
    target = target_storage[:m_size, :n_size]

    valid_addr = _tensor_storage_address(target, row_idx, col_idx)
    masked_addr = _tensor_storage_address(target_storage, m_size, col_idx)

    valid0 = shadow_cell_from_address(valid_addr)
    masked0 = shadow_cell_from_address(masked_addr)

    compiled = _device_tma_masked_store_kernel.warmup(target, m_size, n_size, row_idx, col_idx, target.stride(0),
                                                      BLOCK=block, grid=(1, ), num_warps=4)
    _assert_ttgir_contains(compiled, "ttng.async_tma_copy_local_to_global")

    _device_tma_masked_store_kernel[(1, )](target, m_size, n_size, row_idx, col_idx, target.stride(0), BLOCK=block,
                                           num_warps=4)
    torch.cuda.synchronize()

    expected = torch.zeros_like(target)
    expected[row_idx:, col_idx:] = 1
    torch.testing.assert_close(target, expected)

    valid1 = shadow_cell_from_address(valid_addr)
    masked1 = shadow_cell_from_address(masked_addr)
    assert valid1.write_clock != valid0.write_clock
    assert valid1.write_clock.epoch != 0
    assert _shadow_cell_state(masked1) == _shadow_cell_state(masked0)
    assert target_storage[m_size, col_idx].item() == 0
    assert valid_rows < block
    assert valid_cols < block
    assert "GSanLibrary.cu" not in capfd.readouterr().err


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_host_tma_gather_updates_shadow(with_gsan, capfd):
    block_x = 8
    block_y = 8
    m_size = 11
    n_size = 13
    padded_m = 16
    padded_n = 16
    y_offset = 8
    x_offsets = torch.tensor([1, 3, 5, 7, 9, 10, 11, 13], dtype=torch.int32, device="cuda")
    target_storage = torch.arange(padded_m * padded_n, dtype=torch.int32, device="cuda").reshape(padded_m, padded_n)
    target = target_storage[:m_size, :n_size]
    target_desc = _make_gather_scatter_descriptor(target, block_x, block_y)
    out = torch.empty((block_x, block_y), dtype=torch.int32, device="cuda")

    valid_addr = _tensor_storage_address(target_storage, x_offsets[0].item(), y_offset)
    masked_row_addr = _tensor_storage_address(target_storage, m_size, y_offset)
    masked_col_addr = _tensor_storage_address(target_storage, x_offsets[0].item(), n_size)

    valid0 = shadow_cell_from_address(valid_addr)
    masked_row0 = shadow_cell_from_address(masked_row_addr)
    masked_col0 = shadow_cell_from_address(masked_col_addr)

    compiled = _host_tma_gather_kernel.warmup(out, out.stride(0), out.stride(1), target_desc, x_offsets, y_offset,
                                              BLOCK_X=block_x, grid=(1, ), num_warps=4)
    _assert_ttgir_contains(compiled, "ttng.async_tma_gather")

    _host_tma_gather_kernel[(1, )](out, out.stride(0), out.stride(1), target_desc, x_offsets, y_offset, BLOCK_X=block_x,
                                   num_warps=4)
    torch.cuda.synchronize()

    torch.testing.assert_close(out, _gather_reference(target, x_offsets, y_offset, block_y))

    valid1 = shadow_cell_from_address(valid_addr)
    masked_row1 = shadow_cell_from_address(masked_row_addr)
    masked_col1 = shadow_cell_from_address(masked_col_addr)
    assert _shadow_cell_state(valid1) != _shadow_cell_state(valid0)
    assert _shadow_cell_state(masked_row1) == _shadow_cell_state(masked_row0)
    assert _shadow_cell_state(masked_col1) == _shadow_cell_state(masked_col0)
    assert "GSanLibrary.cu" not in capfd.readouterr().err


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_host_tma_scatter_updates_shadow(with_gsan, capfd):
    block_x = 8
    block_y = 8
    m_size = 11
    n_size = 13
    padded_m = 16
    padded_n = 16
    y_offset = 8
    x_offsets = torch.tensor([1, 3, 5, 7, 9, 10, 11, 13], dtype=torch.int32, device="cuda")
    target_storage = torch.zeros((padded_m, padded_n), dtype=torch.int32, device="cuda")
    target = target_storage[:m_size, :n_size]
    target_desc = _make_gather_scatter_descriptor(target, block_x, block_y)
    src = torch.arange(1, block_x * block_y + 1, dtype=torch.int32, device="cuda").reshape(block_x, block_y)

    valid_addr = _tensor_storage_address(target_storage, x_offsets[0].item(), y_offset)
    masked_row_addr = _tensor_storage_address(target_storage, m_size, y_offset)
    masked_col_addr = _tensor_storage_address(target_storage, x_offsets[0].item(), n_size)

    valid0 = shadow_cell_from_address(valid_addr)
    masked_row0 = shadow_cell_from_address(masked_row_addr)
    masked_col0 = shadow_cell_from_address(masked_col_addr)

    compiled = _host_tma_scatter_kernel.warmup(target_desc, x_offsets, y_offset, src, src.stride(0), src.stride(1),
                                               BLOCK_X=block_x, grid=(1, ), num_warps=4)
    _assert_ttgir_contains(compiled, "ttng.async_tma_scatter")

    _host_tma_scatter_kernel[(1, )](target_desc, x_offsets, y_offset, src, src.stride(0), src.stride(1),
                                    BLOCK_X=block_x, num_warps=4)
    torch.cuda.synchronize()

    torch.testing.assert_close(target, _scatter_reference(target, src, x_offsets, y_offset))

    valid1 = shadow_cell_from_address(valid_addr)
    masked_row1 = shadow_cell_from_address(masked_row_addr)
    masked_col1 = shadow_cell_from_address(masked_col_addr)
    assert valid1.write_clock != valid0.write_clock
    assert valid1.write_clock.epoch != 0
    assert _shadow_cell_state(masked_row1) == _shadow_cell_state(masked_row0)
    assert _shadow_cell_state(masked_col1) == _shadow_cell_state(masked_col0)
    assert target_storage[m_size, y_offset].item() == 0
    assert "GSanLibrary.cu" not in capfd.readouterr().err
