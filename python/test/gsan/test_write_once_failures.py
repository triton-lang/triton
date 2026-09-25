from __future__ import annotations

import contextlib

import pytest
import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from triton._internal_testing import is_cuda, is_hopper_or_newer
from triton.experimental.gsan import create_mem_pool
from triton.experimental.gsan._allocator import get_reserve_pointer, get_reserve_size
from triton.tools.tensor_descriptor import TensorDescriptor
from test_gsan_failures import _raw_kernel, _run_failure_case

pytestmark = pytest.mark.skipif(not is_cuda(), reason="requires CUDA backend")


@contextlib.contextmanager
def _write_once_pools(shadow_granularity=1):
    triton.knobs.compilation.instrumentation_mode = "gsan"
    pool = create_mem_pool(write_once=True, shadow_granularity=shadow_granularity)
    normal_pool = create_mem_pool()
    with torch.cuda.use_mem_pool(normal_pool):
        yield pool


@gluon.jit
def _write_once_failure_kernel(ptr, out, CASE: gl.constexpr, WIDTH: gl.constexpr):
    offsets = gl.arange(0, WIDTH, layout=gl.BlockedLayout([WIDTH], [32], [1], [0]))
    if CASE == "unwritten":
        value = gl.load(ptr + offsets)
        gl.store(out + offsets, value)
    elif CASE == "second":
        gl.store(ptr + offsets, 1)
        gl.store(ptr + offsets, 2)
    elif CASE == "store":
        gl.store(ptr + offsets, 3)
    elif CASE == "overlap":
        gl.store(ptr + offsets, 1)
        if WIDTH == 4:
            overlap_ptr = ptr + offsets
            gl.store(overlap_ptr, 2)
        else:
            gl.store(ptr.to(gl.pointer_type(gl.uint8)) + 3, 2)
    elif CASE == "adjacent":
        gl.store(ptr.to(gl.pointer_type(gl.uint8)), 1)
        gl.store(ptr.to(gl.pointer_type(gl.uint8)) + 1, 2)
    elif CASE == "partial":
        gl.store(ptr, 1)
    elif CASE == "atomic_load":
        value = gl.atomic_load(ptr, sem="relaxed")
        gl.store(out + offsets, value)
    elif CASE == "atomic_store":
        gl.atomic_store(ptr, 1, sem="relaxed")
    elif CASE == "atomic_rmw":
        gl.atomic_add(ptr, 1, sem="relaxed")
    elif CASE == "atomic_cas":
        gl.atomic_cas(ptr, 0, 1, sem="relaxed")
    elif CASE == "poll_timeout":
        matched = gl.atomic_poll(ptr, 1, timeout_ns=0, sem="relaxed")
        gl.store(out, matched)
    elif CASE == "poll_matched":
        matched = gl.atomic_poll(ptr, 0, sem="relaxed")
        gl.store(out, matched)


def _run_write_once_failure(case, shadow_granularity):
    width = max(1, shadow_granularity // 4)
    with _write_once_pools(shadow_granularity) as pool:
        with torch.cuda.use_mem_pool(pool):
            target = torch.zeros(width, dtype=torch.int32, device="cuda")
        out = torch.empty(width, dtype=torch.int32, device="cuda")
        if case in ("cross_kernel", "cache_reuse", "alias"):
            _write_once_failure_kernel[(1, )](target, out, "store", width, num_warps=1)
            if case == "cache_reuse":
                ptr = target.data_ptr()
                del target
                with torch.cuda.use_mem_pool(pool):
                    target = torch.empty(width, dtype=torch.int32, device="cuda")
                assert target.data_ptr() == ptr
            elif case == "alias":
                import os
                from triton.experimental.gsan._allocator import (
                    ShareableHandleType,
                    export_allocation_handles,
                    export_allocation_memhandle_regions,
                    import_allocation_handles,
                )
                from triton.experimental.gsan._utils import uint8_cuda_tensor_from_ptr
                base, _, _, _ = export_allocation_memhandle_regions(target.data_ptr())
                handles = export_allocation_handles(target.data_ptr(), ShareableHandleType.POSIX_FILE_DESCRIPTOR,
                                                    write_once=True, include_granularity=True)
                ptr = import_allocation_handles(*handles[:3], torch.cuda.current_device(),
                                                ShareableHandleType.POSIX_FILE_DESCRIPTOR, write_once=True,
                                                shadow_granularity=handles[3])
                os.close(handles[0])
                os.close(handles[1])
                target = uint8_cuda_tensor_from_ptr(ptr + target.data_ptr() - base, width * 4,
                                                    torch.cuda.current_device()).view(torch.int32)
            case = "store"
        _write_once_failure_kernel[(1, )](target, out, case, width, num_warps=1)
        torch.cuda.synchronize()


@pytest.mark.parametrize("case,marker,error", [
    ("unwritten", "value = gl.load(ptr + offsets)", "Read of write-once memory before its first write"),
    ("second", "gl.store(ptr + offsets, 2)", "Write-once memory written more than once"),
    ("overlap", "gl.store(ptr.to(gl.pointer_type(gl.uint8)) + 3, 2)", "Write-once memory written more than once"),
])
@pytest.mark.parametrize("shadow_granularity", [1, 2, 4, 8, 16], indirect=True)
def test_write_once_rejects_invalid_cell_access(case, marker, error, shadow_granularity):
    if case == "overlap" and shadow_granularity == 16:
        marker = "gl.store(overlap_ptr, 2)"
    _run_failure_case(case, runner=_run_write_once_failure, runner_args=(case, shadow_granularity),
                      source_function=_write_once_failure_kernel.fn, marker=marker, error=error)


@pytest.mark.parametrize("case,marker,error", [
    ("cross_kernel", "gl.store(ptr + offsets, 3)", "Write-once memory written more than once"),
    ("cache_reuse", "gl.store(ptr + offsets, 3)", "Write-once memory written more than once"),
    ("alias", "gl.store(ptr + offsets, 3)", "Write-once memory written more than once"),
    ("atomic_load", "value = gl.atomic_load", "Atomic operations on write-once memory are not supported"),
    ("atomic_store", "gl.atomic_store", "Atomic operations on write-once memory are not supported"),
    ("atomic_rmw", "gl.atomic_add", "Atomic operations on write-once memory are not supported"),
    ("atomic_cas", "gl.atomic_cas", "Atomic operations on write-once memory are not supported"),
    pytest.param(
        "poll_timeout",
        "matched = gl.atomic_poll(ptr, 1, timeout_ns=0",
        "Atomic operations on write-once memory are not supported",
        marks=pytest.mark.xfail(reason="GSan instruments only matched atomic_poll reads", strict=True),
    ),
    ("poll_matched", "matched = gl.atomic_poll(ptr, 0, sem=",
     "Atomic operations on write-once memory are not supported"),
])
def test_write_once_rejects_invalid_access(case, marker, error, shadow_granularity):
    _run_failure_case(case, runner=_run_write_once_failure, runner_args=(case, shadow_granularity),
                      source_function=_write_once_failure_kernel.fn, marker=marker, error=error)


def _run_write_once_unordered_read(shadow_granularity):
    width = max(1, shadow_granularity // 4)
    with _write_once_pools(shadow_granularity) as pool:
        with torch.cuda.use_mem_pool(pool):
            target = torch.empty(width, dtype=torch.int32, device="cuda")
        scratch = torch.empty_like(target)
        counter = torch.zeros_like(target)
        _raw_kernel[(2, )](target, scratch, counter, WIDTH=width, num_warps=1)
        torch.cuda.synchronize()


@pytest.mark.parametrize("shadow_granularity", [1, 2, 4, 8, 16], indirect=True)
def test_write_once_unordered_read(shadow_granularity):
    _run_failure_case("write_once_raw", runner=_run_write_once_unordered_read, runner_args=(shadow_granularity, ),
                      source_function=_raw_kernel.fn, marker="value = gl.load(ptr + offsets)",
                      error="Read after write race detected")


@triton.jit
def _write_once_reduce_kernel(desc):
    desc.atomic_add([0, 0], tl.full((1, 16), 1, tl.int32))


def _run_write_once_reduce(shadow_granularity):
    with _write_once_pools(shadow_granularity) as pool:
        with torch.cuda.use_mem_pool(pool):
            target = torch.zeros((1, 16), dtype=torch.int32, device="cuda")
        desc = TensorDescriptor.from_tensor(target, [1, 16])
        _write_once_reduce_kernel[(1, )](desc)
        torch.cuda.synchronize()


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Requires TMA")
def test_write_once_rejects_descriptor_reduce(shadow_granularity):
    _run_failure_case("write_once_reduce", runner=_run_write_once_reduce, runner_args=(shadow_granularity, ),
                      source_function=_write_once_reduce_kernel.fn, marker="desc.atomic_add",
                      error="Atomic operations on write-once memory are not supported")


@triton.jit
def _memory_category_boundary_kernel(address, out, STORE: tl.constexpr):
    ptr = address.to(tl.pointer_type(tl.int16))
    if STORE:
        tl.store(ptr, 1)
    else:
        value = tl.load(ptr)
        tl.store(out, value)


def _run_memory_category_boundary(pool_index, boundary, is_store):
    with _write_once_pools():
        out = torch.empty(1, dtype=torch.int16, device="cuda")
        pool_size = get_reserve_size() // 16
        offset = pool_index * pool_size + (pool_size // 2 if boundary == "start" else pool_size)
        # A two-byte access straddles a real/shadow or real/unused-slot boundary.
        address = get_reserve_pointer() + offset - 1
        _memory_category_boundary_kernel[(1, )](address, out, is_store)
        torch.cuda.synchronize()


# Cover both modes at the first and last granularity, including the transition
# to unused slots. The allocator tests check address mapping for all ten pools.
@pytest.mark.parametrize("pool_index", [0, 1, 8, 9])
@pytest.mark.parametrize("boundary", ["start", "end"])
@pytest.mark.parametrize("is_store", [False, True])
def test_access_rejects_crossing_memory_category_boundary(pool_index, boundary, is_store):
    _run_failure_case(f"memory_category_boundary_{pool_index}_{boundary}_{is_store}",
                      runner=_run_memory_category_boundary, runner_args=(pool_index, boundary, is_store),
                      source_function=_memory_category_boundary_kernel.fn,
                      marker="tl.store(ptr, 1)" if is_store else "value = tl.load(ptr)",
                      error="Access crosses a GSan memory category boundary")


@pytest.mark.parametrize("granularity", [2, 4, 8])
def test_write_once_rejects_disjoint_writes_in_one_cell(granularity):
    _run_failure_case("adjacent", runner=_run_write_once_failure, runner_args=("adjacent", granularity),
                      source_function=_write_once_failure_kernel.fn,
                      marker="gl.store(ptr.to(gl.pointer_type(gl.uint8)) + 1, 2)",
                      error="Write-once memory written more than once")


def test_write_once_rejects_partial_16_byte_cell():
    _run_failure_case("partial", runner=_run_write_once_failure, runner_args=("partial", 16),
                      source_function=_write_once_failure_kernel.fn, marker="gl.store(ptr, 1)",
                      error="GSan 16-byte pools require complete aligned 16-byte accesses")
