from __future__ import annotations

import contextlib

import pytest
import torch
import triton
import triton.language as tl

from triton._internal_testing import is_cuda, is_hopper_or_newer
from triton.experimental.gsan import create_mem_pool
from triton.experimental.gsan._allocator import get_reserve_pointer, get_reserve_size
from triton.tools.tensor_descriptor import TensorDescriptor
from test_gsan_failures import _raw_kernel, _run_failure_case

pytestmark = pytest.mark.skipif(not is_cuda(), reason="requires CUDA backend")


@contextlib.contextmanager
def _write_once_pools():
    triton.knobs.compilation.instrumentation_mode = "gsan"
    pool = create_mem_pool(write_once=True)
    normal_pool = create_mem_pool()
    with torch.cuda.use_mem_pool(normal_pool):
        yield pool


@triton.jit
def _write_once_failure_kernel(ptr, out, CASE: tl.constexpr):
    if CASE == "unwritten":
        value = tl.load(ptr)
        tl.store(out, value)
    elif CASE == "second":
        tl.store(ptr, 1)
        tl.store(ptr, 2)
    elif CASE == "store":
        tl.store(ptr, 3)
    elif CASE == "overlap":
        tl.store(ptr.to(tl.pointer_type(tl.int32)), 1)
        tl.store(ptr.to(tl.pointer_type(tl.uint8)) + 3, 2)
    elif CASE == "atomic_load":
        value = tl.atomic_load(ptr, sem="relaxed")
        tl.store(out, value)
    elif CASE == "atomic_store":
        tl.atomic_store(ptr, 1, sem="relaxed")
    elif CASE == "atomic_rmw":
        tl.atomic_add(ptr, 1, sem="relaxed")
    elif CASE == "atomic_cas":
        tl.atomic_cas(ptr, 0, 1, sem="relaxed")
    elif CASE == "poll_timeout":
        matched = tl.atomic_poll(ptr, 1, timeout_ns=0, sem="relaxed")
        tl.store(out, matched)
    elif CASE == "poll_matched":
        matched = tl.atomic_poll(ptr, 0, sem="relaxed")
        tl.store(out, matched)


def _run_write_once_failure(case):
    with _write_once_pools() as pool:
        with torch.cuda.use_mem_pool(pool):
            target = torch.zeros(1, dtype=torch.int32, device="cuda")
        out = torch.empty(1, dtype=torch.int32, device="cuda")
        if case in ("cross_kernel", "cache_reuse", "alias"):
            _write_once_failure_kernel[(1, )](target, out, "store")
            if case == "cache_reuse":
                ptr = target.data_ptr()
                del target
                with torch.cuda.use_mem_pool(pool):
                    target = torch.empty(1, dtype=torch.int32, device="cuda")
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
                                                    write_once=True)
                ptr = import_allocation_handles(*handles, torch.cuda.current_device(),
                                                ShareableHandleType.POSIX_FILE_DESCRIPTOR, write_once=True)
                os.close(handles[0])
                os.close(handles[1])
                target = uint8_cuda_tensor_from_ptr(ptr + target.data_ptr() - base, 4,
                                                    torch.cuda.current_device()).view(torch.int32)
            case = "store"
        _write_once_failure_kernel[(1, )](target, out, case)
        torch.cuda.synchronize()


@pytest.mark.parametrize("case,marker,error", [
    ("unwritten", "value = tl.load(ptr)", "Read of write-once memory before its first write"),
    ("second", "tl.store(ptr, 2)", "Write-once memory written more than once"),
    ("cross_kernel", "tl.store(ptr, 3)", "Write-once memory written more than once"),
    ("cache_reuse", "tl.store(ptr, 3)", "Write-once memory written more than once"),
    ("alias", "tl.store(ptr, 3)", "Write-once memory written more than once"),
    ("overlap", "tl.store(ptr.to(tl.pointer_type(tl.uint8)) + 3, 2)", "Write-once memory written more than once"),
    ("atomic_load", "value = tl.atomic_load", "Atomic operations on write-once memory are not supported"),
    ("atomic_store", "tl.atomic_store", "Atomic operations on write-once memory are not supported"),
    ("atomic_rmw", "tl.atomic_add", "Atomic operations on write-once memory are not supported"),
    ("atomic_cas", "tl.atomic_cas", "Atomic operations on write-once memory are not supported"),
    pytest.param(
        "poll_timeout",
        "matched = tl.atomic_poll(ptr, 1, timeout_ns=0",
        "Atomic operations on write-once memory are not supported",
        marks=pytest.mark.xfail(reason="GSan instruments only matched atomic_poll reads", strict=True),
    ),
    ("poll_matched", "matched = tl.atomic_poll(ptr, 0, sem=",
     "Atomic operations on write-once memory are not supported"),
])
def test_write_once_rejects_invalid_access(case, marker, error):
    _run_failure_case(case, runner=_run_write_once_failure, runner_args=(case, ),
                      source_function=_write_once_failure_kernel.fn, marker=marker, error=error)


def _run_write_once_unordered_read():
    with _write_once_pools() as pool:
        with torch.cuda.use_mem_pool(pool):
            target = torch.empty(1, dtype=torch.int32, device="cuda")
        scratch = torch.empty_like(target)
        counter = torch.zeros_like(target)
        _raw_kernel[(2, )](target, scratch, counter, num_warps=1)
        torch.cuda.synchronize()


def test_write_once_unordered_read():
    _run_failure_case("write_once_raw", runner=_run_write_once_unordered_read, source_function=_raw_kernel.fn,
                      marker="value = tl.load(ptr)", error="Read after write race detected")


@triton.jit
def _write_once_reduce_kernel(desc):
    desc.atomic_add([0, 0], tl.full((1, 16), 1, tl.int32))


def _run_write_once_reduce():
    with _write_once_pools() as pool:
        with torch.cuda.use_mem_pool(pool):
            target = torch.zeros((1, 16), dtype=torch.int32, device="cuda")
        desc = TensorDescriptor.from_tensor(target, [1, 16])
        _write_once_reduce_kernel[(1, )](desc)
        torch.cuda.synchronize()


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Requires TMA")
def test_write_once_rejects_descriptor_reduce():
    _run_failure_case("write_once_reduce", runner=_run_write_once_reduce, source_function=_write_once_reduce_kernel.fn,
                      marker="desc.atomic_add", error="Atomic operations on write-once memory are not supported")


@triton.jit
def _memory_category_boundary_kernel(address, out, STORE: tl.constexpr):
    ptr = address.to(tl.pointer_type(tl.int16))
    if STORE:
        tl.store(ptr, 1)
    else:
        value = tl.load(ptr)
        tl.store(out, value)


def _run_memory_category_boundary(boundary, is_store):
    with _write_once_pools():
        out = torch.empty(1, dtype=torch.int16, device="cuda")
        offset = {
            "reserve_start": 0,
            "write_once_start": get_reserve_size() // 2,
            "reserve_end": get_reserve_size(),
        }[boundary]
        # A two-byte access straddles the boundary without needing a huge allocation.
        address = get_reserve_pointer() + offset - 1
        _memory_category_boundary_kernel[(1, )](address, out, is_store)
        torch.cuda.synchronize()


@pytest.mark.parametrize("boundary", ["reserve_start", "write_once_start", "reserve_end"])
@pytest.mark.parametrize("is_store", [False, True])
def test_access_rejects_crossing_memory_category_boundary(boundary, is_store):
    _run_failure_case(f"memory_category_boundary_{boundary}_{is_store}", runner=_run_memory_category_boundary,
                      runner_args=(boundary, is_store), source_function=_memory_category_boundary_kernel.fn,
                      marker="tl.store(ptr, 1)" if is_store else "value = tl.load(ptr)",
                      error="Access crosses a GSan memory category boundary")
