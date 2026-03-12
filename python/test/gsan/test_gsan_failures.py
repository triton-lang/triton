from __future__ import annotations

import functools
import inspect
from pathlib import Path

import pytest
import torch
import triton
import triton.language as tl

from triton._internal_testing import is_blackwell, is_cuda, run_in_process
from triton.experimental.gsan import create_mem_pool
from triton.experimental.gsan._testing_utils import nanosleep
from triton.tools.tensor_descriptor import TensorDescriptor

pytestmark = pytest.mark.skipif(not is_cuda(), reason="requires CUDA backend")


@triton.jit
def atomic_poll(counter_ptr, expected):
    while tl.atomic_add(counter_ptr, 0, sem="relaxed") < expected:
        nanosleep(100)


@triton.jit
def _raw_kernel(ptr, scratch_ptr, counter_ptr):
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(ptr, 1)
        tl.atomic_add(counter_ptr, 1, sem="relaxed")
    else:
        atomic_poll(counter_ptr, 1)
        value = tl.load(ptr)
        tl.store(scratch_ptr, value)


@triton.jit
def _war_kernel(ptr, scratch_ptr, counter_ptr):
    pid = tl.program_id(0)
    if pid == 0:
        value = tl.load(ptr)
        tl.store(scratch_ptr, value)
        tl.atomic_add(counter_ptr, 1, sem="relaxed")
    else:
        atomic_poll(counter_ptr, 1)
        tl.store(ptr, 1)


@triton.jit
def _waw_kernel(ptr, scratch_ptr, counter_ptr):
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(ptr, 1)
        tl.atomic_add(counter_ptr, 1, sem="relaxed")
    else:
        atomic_poll(counter_ptr, 1)
        tl.store(ptr, 2)


@triton.jit
def _cross_sm_relaxed_flag_kernel(payload_ptr, flag_ptr, scratch_ptr):
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(payload_ptr, 1000)
        tl.atomic_xchg(flag_ptr, 1, sem="relaxed", scope="gpu")
    elif pid == 1:
        ready = 0
        while ready != 1:
            nanosleep(500_000)
            ready = tl.atomic_add(flag_ptr, 0, sem="acquire", scope="gpu")
        result = tl.load(payload_ptr)
        tl.store(scratch_ptr, result)


@triton.jit
def _cross_sm_cta_scope_kernel(payload_ptr, flag_ptr, scratch_ptr):
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(payload_ptr, 1000)
        tl.atomic_xchg(flag_ptr, 1, sem="release", scope="cta")
    elif pid == 1:
        ready = 0
        while ready != 1:
            nanosleep(500_000)
            ready = tl.atomic_add(flag_ptr, 0, sem="acquire", scope="cta")
        result = tl.load(payload_ptr)
        tl.store(scratch_ptr, result)


@triton.jit
def _transitive_relaxed_relay_kernel(payload_ptr, flag0_ptr, flag1_ptr, scratch_ptr):
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(payload_ptr, 1000)
        tl.atomic_xchg(flag0_ptr, 1, sem="release", scope="gpu")
    elif pid == 1:
        ready = 0
        while ready != 1:
            nanosleep(500_000)
            ready = tl.atomic_add(flag0_ptr, 0, sem="relaxed", scope="gpu")
        tl.atomic_xchg(flag1_ptr, 1, sem="release", scope="gpu")
    elif pid == 2:
        ready = 0
        while ready != 1:
            nanosleep(500_000)
            ready = tl.atomic_add(flag1_ptr, 0, sem="acquire", scope="gpu")
        result = tl.load(payload_ptr)
        tl.store(scratch_ptr, result)


@triton.jit
def _tma_raw_kernel(ptr, scratch_ptr, counter_ptr, m_size, n_size, row_idx, col_idx, stride_0, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    if pid == 0:
        desc = tl.make_tensor_descriptor(ptr, [m_size, n_size], [stride_0, 1], [BLOCK, BLOCK])
        values = tl.full((BLOCK, BLOCK), 1, dtype=tl.int32)
        desc.store([row_idx, col_idx], values)
        tl.atomic_add(counter_ptr, 1, sem="relaxed")
    else:
        atomic_poll(counter_ptr, 1)
        value = tl.load(ptr + row_idx * stride_0 + col_idx)
        tl.store(scratch_ptr, value)


@triton.jit
def _host_tma_war_kernel(target_ptr, target_desc, scratch_desc, counter_ptr, row_idx, col_idx, stride_0):
    pid = tl.program_id(0)
    if pid == 0:
        block = target_desc.load([row_idx, col_idx])
        scratch_desc.store([row_idx, col_idx], block)
        tl.atomic_add(counter_ptr, 1, sem="relaxed")
    else:
        atomic_poll(counter_ptr, 1)
        tl.store(target_ptr + row_idx * stride_0 + col_idx, 1)


@triton.jit
def _host_tma_gather_war_kernel(target_ptr, target_desc, x_offsets_ptr, scratch_ptr, counter_ptr, row_idx, y_offset,
                                stride_0, scratch_stride_0, scratch_stride_1, BLOCK_X: tl.constexpr):
    BLOCK_Y: tl.constexpr = target_desc.block_shape[1]
    pid = tl.program_id(0)
    if pid == 0:
        x_offsets = tl.load(x_offsets_ptr + tl.arange(0, BLOCK_X))
        values = target_desc.gather(x_offsets, y_offset)
        indices_x = tl.arange(0, BLOCK_X)[:, None] * scratch_stride_0
        indices_y = tl.arange(0, BLOCK_Y)[None, :] * scratch_stride_1
        tl.store(scratch_ptr + indices_x + indices_y, values)
        tl.atomic_add(counter_ptr, 1, sem="relaxed")
    else:
        atomic_poll(counter_ptr, 1)
        tl.store(target_ptr + row_idx * stride_0 + y_offset, 1)


@triton.jit
def _host_tma_scatter_war_kernel(target_ptr, target_desc, x_offsets_ptr, src_ptr, src_stride_0, src_stride_1,
                                 scratch_ptr, counter_ptr, row_idx, y_offset, stride_0, BLOCK_X: tl.constexpr):
    BLOCK_Y: tl.constexpr = target_desc.block_shape[1]
    pid = tl.program_id(0)
    if pid == 0:
        value = tl.load(target_ptr + row_idx * stride_0 + y_offset)
        tl.store(scratch_ptr, value)
        tl.atomic_add(counter_ptr, 1, sem="relaxed")
    else:
        atomic_poll(counter_ptr, 1)
        x_offsets = tl.load(x_offsets_ptr + tl.arange(0, BLOCK_X))
        indices_x = tl.arange(0, BLOCK_X)[:, None] * src_stride_0
        indices_y = tl.arange(0, BLOCK_Y)[None, :] * src_stride_1
        values = tl.load(src_ptr + indices_x + indices_y)
        target_desc.scatter(values, x_offsets, y_offset)


def _cuda_byte_allocator(size: int, _align: int, _stream):
    return torch.empty(size, dtype=torch.int8, device="cuda")


def run_with_gsan(fn):

    @functools.wraps(fn)
    def wrapped() -> None:
        triton.knobs.compilation.instrumentation_mode = "gsan"
        pool = create_mem_pool()
        with torch.cuda.use_mem_pool(pool):
            fn()

    return wrapped


@run_with_gsan
def _run_raw_case() -> None:
    target = torch.zeros(1, dtype=torch.int32, device="cuda")
    scratch = torch.zeros(1, dtype=torch.int32, device="cuda")
    counter = torch.zeros(1, dtype=torch.int32, device="cuda")
    _raw_kernel[(2, )](target, scratch, counter, num_warps=1)


@run_with_gsan
def _run_war_case() -> None:
    target = torch.zeros(1, dtype=torch.int32, device="cuda")
    scratch = torch.zeros(1, dtype=torch.int32, device="cuda")
    counter = torch.zeros(1, dtype=torch.int32, device="cuda")
    _war_kernel[(2, )](target, scratch, counter, num_warps=1)


@run_with_gsan
def _run_waw_case() -> None:
    target = torch.zeros(1, dtype=torch.int32, device="cuda")
    scratch = torch.zeros(1, dtype=torch.int32, device="cuda")
    counter = torch.zeros(1, dtype=torch.int32, device="cuda")
    _waw_kernel[(2, )](target, scratch, counter, num_warps=1)


@run_with_gsan
def _run_tma_raw_case() -> None:
    block = 32
    m_size = 35
    n_size = 37
    padded_n = 40
    row_idx = 5
    col_idx = 8

    counter = torch.zeros(1, dtype=torch.int32, device="cuda")
    target_storage = torch.zeros((m_size, padded_n), dtype=torch.int32, device="cuda")
    target = target_storage[:, :n_size]
    scratch = torch.zeros(1, dtype=torch.int32, device="cuda")
    triton.set_allocator(_cuda_byte_allocator)
    _tma_raw_kernel[(2, )](target, scratch, counter, m_size, n_size, row_idx, col_idx, target.stride(0), BLOCK=block,
                           num_warps=4)


@run_with_gsan
def _run_host_tma_war_case() -> None:
    block = 32
    m_size = 35
    n_size = 37
    padded_n = 40
    row_idx = 5
    col_idx = 8

    target_storage = torch.zeros((m_size, padded_n), dtype=torch.int32, device="cuda")
    scratch_storage = torch.zeros_like(target_storage)
    target = target_storage[:, :n_size]
    scratch = scratch_storage[:, :n_size]
    target_desc = TensorDescriptor.from_tensor(target, [block, block])
    scratch_desc = TensorDescriptor.from_tensor(scratch, [block, block])
    counter = torch.zeros(1, dtype=torch.int32, device="cuda")
    _host_tma_war_kernel[(2, )](target, target_desc, scratch_desc, counter, row_idx, col_idx, target.stride(0), num_warps=4)


@run_with_gsan
def _run_host_tma_gather_war_case() -> None:
    block_x = 8
    block_y = 8
    m_size = 11
    n_size = 13
    padded_m = 16
    padded_n = 16
    row_idx = 5
    y_offset = 8
    x_offsets_values = [5, 7, 9, 10, 1, 3, 11, 13]

    target_storage = torch.zeros((padded_m, padded_n), dtype=torch.int32, device="cuda")
    target = target_storage[:m_size, :n_size]
    x_offsets = torch.tensor(x_offsets_values, dtype=torch.int32, device="cuda")
    target_desc = TensorDescriptor.from_tensor(target, [1, block_y])
    scratch = torch.zeros((block_x, block_y), dtype=torch.int32, device="cuda")
    counter = torch.zeros(1, dtype=torch.int32, device="cuda")
    _host_tma_gather_war_kernel[(2, )](target, target_desc, x_offsets, scratch, counter, row_idx, y_offset, target.stride(0),
                                       scratch.stride(0), scratch.stride(1), BLOCK_X=block_x, num_warps=4)


@run_with_gsan
def _run_host_tma_scatter_war_case() -> None:
    block_x = 8
    block_y = 8
    m_size = 11
    n_size = 13
    padded_m = 16
    padded_n = 16
    row_idx = 5
    y_offset = 8
    x_offsets_values = [5, 7, 9, 10, 1, 3, 11, 13]

    target_storage = torch.zeros((padded_m, padded_n), dtype=torch.int32, device="cuda")
    target = target_storage[:m_size, :n_size]
    x_offsets = torch.tensor(x_offsets_values, dtype=torch.int32, device="cuda")
    target_desc = TensorDescriptor.from_tensor(target, [1, block_y])
    src = torch.arange(1, block_x * block_y + 1, dtype=torch.int32, device="cuda").reshape(block_x, block_y)
    scratch = torch.zeros(1, dtype=torch.int32, device="cuda")
    counter = torch.zeros(1, dtype=torch.int32, device="cuda")
    _host_tma_scatter_war_kernel[(2, )](target, target_desc, x_offsets, src, src.stride(0), src.stride(1), scratch, counter,
                                        row_idx, y_offset, target.stride(0), BLOCK_X=block_x, num_warps=4)


@run_with_gsan
def _run_cross_sm_relaxed_flag_case() -> None:
    payload = torch.zeros(1, dtype=torch.int32, device="cuda")
    flags = torch.zeros(1, dtype=torch.int32, device="cuda")
    scratch = torch.full((1, ), -1, dtype=torch.int32, device="cuda")
    _cross_sm_relaxed_flag_kernel[(2, )](payload, flags, scratch, num_warps=1)


@run_with_gsan
def _run_cross_sm_cta_scope_case() -> None:
    payload = torch.zeros(1, dtype=torch.int32, device="cuda")
    flags = torch.zeros(1, dtype=torch.int32, device="cuda")
    scratch = torch.full((1, ), -1, dtype=torch.int32, device="cuda")
    _cross_sm_cta_scope_kernel[(2, )](payload, flags, scratch, num_warps=1)


@run_with_gsan
def _run_transitive_relaxed_relay_case() -> None:
    payload = torch.zeros(1, dtype=torch.int32, device="cuda")
    flag0 = torch.zeros(1, dtype=torch.int32, device="cuda")
    flag1 = torch.zeros(1, dtype=torch.int32, device="cuda")
    scratch = torch.full((1, ), -1, dtype=torch.int32, device="cuda")
    _transitive_relaxed_relay_kernel[(3, )](payload, flag0, flag1, scratch, num_warps=1)


def _expected_file_line(source_function, marker: str) -> str:
    source_lines, starting_line = inspect.getsourcelines(source_function)
    for line_offset, line in enumerate(source_lines):
        if marker in line:
            return f"{Path(__file__).name}:{starting_line + line_offset}"
    raise AssertionError(f"Could not find marker {marker!r} for function {source_function!r}")


def _run_failure_case(case: str, *, runner, source_function, marker: str, error: str) -> None:
    if torch.cuda.device_count() < 1:
        pytest.skip("requires at least 1 CUDA device")

    result = run_in_process(runner, ())
    print(result.driver_stderr_output)
    assert isinstance(result.exc, RuntimeError), (f"case={case} completed without the expected GSan failure\n"
                                                  f"exc={result.exc!r}\n"
                                                  f"driver stderr:\n{result.driver_stderr_output}")
    assert "GSanLibrary.cu" not in result.driver_stderr_output
    assert Path(__file__).name in result.driver_stderr_output
    assert _expected_file_line(source_function, marker) in result.driver_stderr_output
    assert error in result.driver_stderr_output


def test_read_after_write():
    _run_failure_case("raw", runner=_run_raw_case, source_function=_raw_kernel.fn, marker="value = tl.load(ptr)",
                      error="Read after write race detected")


def test_write_after_read():
    _run_failure_case("war", runner=_run_war_case, source_function=_war_kernel.fn, marker="tl.store(ptr, 1)",
                      error="Write after read race detected")


def test_write_after_write():
    _run_failure_case("waw", runner=_run_waw_case, source_function=_waw_kernel.fn, marker="tl.store(ptr, 2)",
                      error="Write after write race detected")


def test_tma_read_after_write():
    _run_failure_case("tma_raw", runner=_run_tma_raw_case, source_function=_tma_raw_kernel.fn,
                      marker="value = tl.load(ptr + row_idx * stride_0 + col_idx)",
                      error="Read after write race detected")


def test_host_tma_write_after_read():
    _run_failure_case("host_tma_war", runner=_run_host_tma_war_case, source_function=_host_tma_war_kernel.fn,
                      marker="tl.store(target_ptr + row_idx * stride_0 + col_idx, 1)",
                      error="Write after read race detected")


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_host_tma_gather_write_after_read():
    _run_failure_case("host_tma_gather_war", runner=_run_host_tma_gather_war_case,
                      source_function=_host_tma_gather_war_kernel.fn,
                      marker="tl.store(target_ptr + row_idx * stride_0 + y_offset, 1)",
                      error="Write after read race detected")


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_host_tma_scatter_write_after_read():
    _run_failure_case("host_tma_scatter_war", runner=_run_host_tma_scatter_war_case,
                      source_function=_host_tma_scatter_war_kernel.fn,
                      marker="target_desc.scatter(values, x_offsets, y_offset)", error="Write after read race detected")


def test_cross_sm_relaxed_flag_read_after_write():
    _run_failure_case("cross_sm_relaxed_flag", runner=_run_cross_sm_relaxed_flag_case,
                      source_function=_cross_sm_relaxed_flag_kernel.fn, marker="result = tl.load(payload_ptr)",
                      error="Read after write race detected")


def test_cross_sm_cta_scope_read_after_write():
    _run_failure_case("cross_sm_cta_scope", runner=_run_cross_sm_cta_scope_case,
                      source_function=_cross_sm_cta_scope_kernel.fn,
                      marker='ready = tl.atomic_add(flag_ptr, 0, sem="acquire", scope="cta"',
                      error="Read after write race detected")


def test_transitive_release_acquire_requires_middle_acquire():
    _run_failure_case("transitive_relaxed_relay", runner=_run_transitive_relaxed_relay_case,
                      source_function=_transitive_relaxed_relay_kernel.fn, marker="result = tl.load(payload_ptr)",
                      error="Read after write race detected")
