from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch
import triton
import triton.language as tl

from triton._internal_testing import is_blackwell, is_cuda, run_in_process
from triton.experimental.gsan import create_mem_pool
from triton.tools.tensor_descriptor import TensorDescriptor

pytestmark = pytest.mark.skipif(not is_cuda(), reason="requires CUDA backend")


@triton.jit
def nanosleep(duration):
    duration = tl.to_tensor(duration)
    tl.inline_asm_elementwise("nanosleep.u32 $1; mov.b32 $0, 0;", "=r, r", [duration], tl.int32, is_pure=False, pack=1)


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
        indices_x = tl.arange(0, BLOCK_X)[:, None] * src_stride_0
        indices_y = tl.arange(0, BLOCK_Y)[None, :] * src_stride_1
        values = tl.load(src_ptr + indices_x + indices_y)
        x_offsets = tl.load(x_offsets_ptr + tl.arange(0, BLOCK_X))
        target_desc.scatter(values, x_offsets, y_offset)


def _run_case(case: str) -> None:
    block = 32
    m_size = 35
    n_size = 37
    padded_n = 40
    row_idx = 5
    col_idx = 8
    gather_block_x = 8
    gather_block_y = 8
    gather_m_size = 11
    gather_n_size = 13
    gather_padded_m = 16
    gather_padded_n = 16
    gather_row_idx = 5
    gather_y_offset = 8
    gather_x_offsets = [5, 7, 9, 10, 1, 3, 11, 13]
    pool = create_mem_pool()
    with torch.cuda.use_mem_pool(pool):
        if case == "tma_raw":
            target_storage = torch.zeros((m_size, padded_n), dtype=torch.int32, device="cuda")
            target = target_storage[:, :n_size]
            scratch = torch.zeros(1, dtype=torch.int32, device="cuda")

            def alloc_fn(size: int, _align: int, _stream):
                return torch.empty(size, dtype=torch.int8, device="cuda")

            triton.set_allocator(alloc_fn)
        elif case == "host_tma_war":
            target_storage = torch.zeros((m_size, padded_n), dtype=torch.int32, device="cuda")
            scratch_storage = torch.zeros_like(target_storage)
            target = target_storage[:, :n_size]
            scratch = scratch_storage[:, :n_size]
            target_desc = TensorDescriptor.from_tensor(target, [block, block])
            scratch_desc = TensorDescriptor.from_tensor(scratch, [block, block])
        elif case in {"host_tma_gather_war", "host_tma_scatter_war"}:
            target_storage = torch.zeros((gather_padded_m, gather_padded_n), dtype=torch.int32, device="cuda")
            target = target_storage[:gather_m_size, :gather_n_size]
            target_desc = TensorDescriptor.from_tensor(target, [1, gather_block_y])
            x_offsets = torch.tensor(gather_x_offsets, dtype=torch.int32, device="cuda")
            if case == "host_tma_gather_war":
                scratch = torch.zeros((gather_block_x, gather_block_y), dtype=torch.int32, device="cuda")
            else:
                src = torch.arange(1, gather_block_x * gather_block_y + 1, dtype=torch.int32,
                                   device="cuda").reshape(gather_block_x, gather_block_y)
                scratch = torch.zeros(1, dtype=torch.int32, device="cuda")
        else:
            target = torch.zeros(1, dtype=torch.int32, device="cuda")
            scratch = torch.zeros(1, dtype=torch.int32, device="cuda")
        counter = torch.zeros(1, dtype=torch.int32, device="cuda")

    triton.knobs.compilation.instrumentation_mode = "gsan"
    kernel = globals()[f"_{case}_kernel"]
    if case == "tma_raw":
        kernel[(2, )](target, scratch, counter, m_size, n_size, row_idx, col_idx, target.stride(0), BLOCK=block,
                      num_warps=4)
    elif case == "host_tma_war":
        kernel[(2, )](target, target_desc, scratch_desc, counter, row_idx, col_idx, target.stride(0), num_warps=4)
    elif case == "host_tma_gather_war":
        kernel[(2, )](target, target_desc, x_offsets, scratch, counter, gather_row_idx, gather_y_offset,
                      target.stride(0), scratch.stride(0), scratch.stride(1), BLOCK_X=gather_block_x, num_warps=4)
    elif case == "host_tma_scatter_war":
        kernel[(2, )](target, target_desc, x_offsets, src, src.stride(0), src.stride(1), scratch, counter,
                      gather_row_idx, gather_y_offset, target.stride(0), BLOCK_X=gather_block_x, num_warps=4)
    else:
        kernel[(2, )](target, scratch, counter, num_warps=1)


CASE_INFO = {
    "raw": {
        "error": "Read after write race detected",
        "function": _raw_kernel.fn,
        "marker": "value = tl.load(ptr)",
    },
    "war": {
        "error": "Write after read race detected",
        "function": _war_kernel.fn,
        "marker": "tl.store(ptr, 1)",
    },
    "waw": {
        "error": "Write after write race detected",
        "function": _waw_kernel.fn,
        "marker": "tl.store(ptr, 2)",
    },
    "tma_raw": {
        "error": "Read after write race detected",
        "function": _tma_raw_kernel.fn,
        "marker": "value = tl.load(ptr + row_idx * stride_0 + col_idx)",
    },
    "host_tma_war": {
        "error": "Write after read race detected",
        "function": _host_tma_war_kernel.fn,
        "marker": "tl.store(target_ptr + row_idx * stride_0 + col_idx, 1)",
    },
    "host_tma_gather_war": {
        "error": "Write after read race detected",
        "function": _host_tma_gather_war_kernel.fn,
        "marker": "tl.store(target_ptr + row_idx * stride_0 + y_offset, 1)",
    },
    "host_tma_scatter_war": {
        "error": "Write after read race detected",
        "function": _host_tma_scatter_war_kernel.fn,
        "marker": "target_desc.scatter(values, x_offsets, y_offset)",
    },
}


def _expected_file_line(case: str) -> str:
    source_lines, starting_line = inspect.getsourcelines(CASE_INFO[case]["function"])
    markers = CASE_INFO[case]["marker"]
    if isinstance(markers, str):
        markers = (markers, )

    matches = []
    for marker in markers:
        for line_offset, line in enumerate(source_lines):
            if marker in line:
                matches.append(f"{Path(__file__).name}:{starting_line + line_offset}")
                break
        else:
            raise AssertionError(f"Could not find marker {marker!r} for case {case!r}")
    return matches[0] if len(matches) == 1 else tuple(matches)


def _run_failure_case(case: str) -> None:
    if torch.cuda.device_count() < 1:
        pytest.skip("requires at least 1 CUDA device")

    result = run_in_process(_run_case, (case, ))
    print(result.driver_stderr_output)
    assert isinstance(result.exc, RuntimeError), (f"case={case} completed without the expected GSan failure\n"
                                                  f"exc={result.exc!r}\n"
                                                  f"driver stderr:\n{result.driver_stderr_output}")
    assert "GSanLibrary.cu" not in result.driver_stderr_output
    assert Path(__file__).name in result.driver_stderr_output
    assert _expected_file_line(case) in result.driver_stderr_output
    assert CASE_INFO[case]["error"] in result.driver_stderr_output


def test_read_after_write():
    _run_failure_case("raw")


def test_write_after_read():
    _run_failure_case("war")


def test_write_after_write():
    _run_failure_case("waw")


def test_tma_read_after_write():
    _run_failure_case("tma_raw")


def test_host_tma_write_after_read():
    _run_failure_case("host_tma_war")


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_host_tma_gather_write_after_read():
    _run_failure_case("host_tma_gather_war")


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_host_tma_scatter_write_after_read():
    _run_failure_case("host_tma_scatter_war")
