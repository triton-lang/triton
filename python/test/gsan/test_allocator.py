from __future__ import annotations

import os

import pytest
import torch

from triton._internal_testing import is_cuda
from triton.experimental.gsan import create_mem_pool
from triton.experimental.gsan._allocator import (export_allocation_handles, free_allocation, get_reserve_pointer,
                                                 get_reserve_size, gsan_free, gsan_malloc,
                                                 import_allocation_handles)
from triton.experimental.gsan._testing_utils import shadow_region, uint8_cuda_tensor_from_ptr


def _shadow_tensor_for(real: torch.Tensor) -> torch.Tensor:
    reserve_ptr = get_reserve_pointer()
    reserve_size = get_reserve_size()
    shadow_ptr, shadow_size = shadow_region(real.data_ptr(), real.untyped_storage().nbytes(), reserve_ptr, reserve_size)
    return uint8_cuda_tensor_from_ptr(shadow_ptr, shadow_size, torch.cuda.current_device())


@pytest.fixture
def _direct_allocator():
    device = torch.cuda.current_device()
    stream = 0
    reserve_ptr = get_reserve_pointer()
    reserve_size = get_reserve_size()
    allocated = set()

    def malloc(size: int) -> int:
        ptr_int = gsan_malloc(size, device, stream)
        if ptr_int != 0:
            allocated.add(ptr_int)
        return ptr_int

    def free(ptr: int, size: int = 0) -> None:
        gsan_free(ptr, device, size, stream)
        if ptr in allocated:
            allocated.remove(ptr)

    try:
        yield malloc, free, reserve_ptr, reserve_size
    finally:
        # Cleanup any allocated pointers
        for ptr in list(allocated):
            gsan_free(ptr, device, 0, stream)


@pytest.mark.skipif(not is_cuda(), reason="requires CUDA backend")
def test_malloc_edge_cases(_direct_allocator):
    malloc, free, reserve_ptr, reserve_size = _direct_allocator

    # Invalid sizes are rejected.
    assert malloc(0) == 0
    assert malloc(-1) == 0
    assert malloc(reserve_size) == 0  # larger than the full real region

    # Null free is a no-op.
    free(0)


def test_malloc_free(_direct_allocator):
    malloc, free, reserve_ptr, reserve_size = _direct_allocator
    real_base = reserve_ptr + reserve_size // 2

    # First valid allocation should come from the real base and be reusable.
    p0 = malloc(1)
    assert p0 == real_base
    free(p0)
    assert malloc(1) == p0

    p1 = malloc(1)
    _ = malloc(1)

    free(p1)
    p3 = malloc(1)
    assert p3 == p1


@pytest.mark.skipif(not is_cuda(), reason="requires CUDA backend")
def test_malloc_fragmentation_reuse_and_coalesce(_direct_allocator):
    malloc, free, _, _ = _direct_allocator

    p0 = malloc(1)
    p1 = malloc(1)
    assert p0 != 0 and p1 != 0
    assert p0 < p1

    block = p1 - p0
    assert block > 0

    # Reuse exact freed block under fragmentation.
    free(p1)
    p1_reuse = malloc(1)
    assert p1_reuse == p1

    # Free two siblings and request a slightly larger block; should coalesce.
    free(p0)
    free(p1_reuse)
    parent = malloc(block + 1)
    assert parent == p0

    free(parent)


@pytest.mark.skipif(not is_cuda(), reason="requires CUDA backend")
def test_free_invalid_pointer_and_double_free(_direct_allocator):
    malloc, free, _, _ = _direct_allocator

    p0 = malloc(1)
    assert p0 != 0

    # Invalid interior-pointer free should not free p0 and must not crash.
    free(p0 + 1)

    free(p0)
    free(p0)  # double free must be a no-op

    # p0 should become reusable after the valid free above.
    p0_reuse = malloc(1)
    assert p0_reuse == p0

    free(p0_reuse)


@pytest.mark.skipif(not is_cuda(), reason="requires CUDA backend")
def test_mem_pool():
    pool = create_mem_pool()
    with torch.cuda.use_mem_pool(pool):
        real = torch.empty(4096, dtype=torch.uint8, device="cuda")

    reserve_ptr = get_reserve_pointer()
    reserve_size = get_reserve_size()
    assert reserve_ptr != 0
    assert reserve_size > 0

    # Check real allocation is in higher half of reserve
    real_base = reserve_ptr + reserve_size // 2
    assert real_base <= real.data_ptr() < reserve_ptr + reserve_size

    shadow = _shadow_tensor_for(real)
    assert reserve_ptr <= shadow.data_ptr() < reserve_ptr + reserve_size // 2

    # Test that real and shadow allocation can be used
    real.zero_()
    real.add_(7)
    # Note: shadow memory is zero-initialized by the allocator
    shadow.add_(3)

    assert torch.all(real == 7).item()
    assert torch.all(shadow == 3).item()
    del pool
    del real
    del shadow
    torch.cuda.synchronize()


@pytest.mark.skipif(not is_cuda(), reason="requires CUDA backend")
def test_export_import_allocation_handles_maps_real_and_shadow(_direct_allocator):
    malloc, free, reserve_ptr, reserve_size = _direct_allocator
    device = torch.cuda.current_device()

    real_ptr = malloc(4096)
    assert real_ptr != 0

    imported_ptr = 0
    real_fd = -1
    shadow_fd = -1
    try:
        real_fd, shadow_fd, alloc_size = export_allocation_handles(real_ptr)
        assert alloc_size > 0

        imported_ptr = import_allocation_handles(real_fd, shadow_fd, alloc_size, device)
        assert imported_ptr != 0
        assert imported_ptr != real_ptr

        local_real = uint8_cuda_tensor_from_ptr(real_ptr, alloc_size, device)
        imported_real = uint8_cuda_tensor_from_ptr(imported_ptr, alloc_size, device)

        local_shadow_ptr, local_shadow_size = shadow_region(real_ptr, alloc_size, reserve_ptr, reserve_size)
        imported_shadow_ptr, imported_shadow_size = shadow_region(imported_ptr, alloc_size, reserve_ptr, reserve_size)
        assert local_shadow_size == imported_shadow_size

        local_shadow = uint8_cuda_tensor_from_ptr(local_shadow_ptr, local_shadow_size, device)
        imported_shadow = uint8_cuda_tensor_from_ptr(imported_shadow_ptr, imported_shadow_size, device)

        imported_real.fill_(11)
        assert torch.all(local_real == 11).item()

        imported_shadow.fill_(5)
        assert torch.all(local_shadow == 5).item()
    finally:
        if real_fd >= 0:
            os.close(real_fd)
        if shadow_fd >= 0:
            os.close(shadow_fd)
        if imported_ptr != 0:
            free_allocation(imported_ptr, device)
        free(real_ptr)
