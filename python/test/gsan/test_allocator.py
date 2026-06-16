from __future__ import annotations

import os

import pytest
import torch

from triton._internal_testing import is_cuda, run_in_process
from triton.experimental.gsan import ShareableHandleType, configure, create_mem_pool, freeze_config
from triton.experimental.gsan._allocator import (
    export_allocation_handles,
    export_allocation_memhandle_regions,
    export_runtime_state_handle,
    free_allocation,
    get_device_rank,
    get_reserve_pointer,
    get_reserve_size,
    gsan_free,
    gsan_malloc,
    import_allocation_handles,
    import_runtime_state_handle,
)
from triton.experimental.gsan._testing_utils import global_state, shadow_tensor_for
from triton.experimental.gsan._utils import uint8_cuda_tensor_from_ptr

# With 2 MiB pages, this rounds to a 6 MiB allocation inside an 8 MiB tree node.
# This tests cases where AllocNode.size != AllocNode.allocSize
_ODD_LARGE_ALLOCATION_SIZE = 4 * 1024 * 1024 + 1


def _run_configure_check(device_ranks: dict[int, int], num_devices: int) -> None:
    configure(device_ranks=device_ranks, num_devices=num_devices)
    assert get_device_rank(0) == device_ranks[0]
    assert get_device_rank(1) == device_ranks[1]


def _run_configure_runtime_fields_check() -> None:
    device = torch.cuda.current_device()
    configure(rng_seed=12345, clock_buffer_size=17)

    ptr = gsan_malloc(1, device, 0)
    try:
        state = global_state(device_index=device)
        assert state.rng_seed == 12345
        assert state.clock_buffer_size == 17
    finally:
        gsan_free(ptr, device, 0, 0)


def _run_freeze_config_check() -> None:
    configure(rng_seed=12345)
    freeze_config()
    try:
        configure(rng_seed=12345)
    except RuntimeError as exc:
        assert "configuration is already frozen" in str(exc)
    else:
        raise AssertionError("expected freeze_config() to reject later config changes")


def _run_allocator_freezes_config_check() -> None:
    device = torch.cuda.current_device()
    configure(rng_seed=12345)
    ptr = gsan_malloc(1, device, 0)
    try:
        try:
            configure(rng_seed=12345)
        except RuntimeError as exc:
            assert "configuration is already frozen" in str(exc)
        else:
            raise AssertionError("expected allocator initialization to freeze config")
    finally:
        gsan_free(ptr, device, 0, 0)


def _run_export_import_fabric_handles_check(explicit_config: bool) -> None:
    device = torch.cuda.current_device()
    configure(
        device_ranks={device: 0},
        num_devices=2,
        handle_type=ShareableHandleType.FABRIC if explicit_config else None,
    )
    real_ptr = gsan_malloc(4096, device)
    imported_ptr = 0
    try:
        runtime_handle, runtime_alloc_size = export_runtime_state_handle(
            device,
            ShareableHandleType.FABRIC,
        )
        assert isinstance(runtime_handle, bytes)
        assert len(runtime_handle) == 64
        assert runtime_alloc_size > 0
        import_runtime_state_handle(
            runtime_handle,
            runtime_alloc_size,
            1,
            device,
            ShareableHandleType.FABRIC,
        )

        real_handle, shadow_handle, alloc_size = export_allocation_handles(
            real_ptr,
            ShareableHandleType.FABRIC,
        )
        assert isinstance(real_handle, bytes)
        assert isinstance(shadow_handle, bytes)
        assert len(real_handle) == 64
        assert len(shadow_handle) == 64

        imported_ptr = import_allocation_handles(
            real_handle,
            shadow_handle,
            alloc_size,
            device,
            ShareableHandleType.FABRIC,
        )
        local_real = uint8_cuda_tensor_from_ptr(real_ptr, alloc_size, device)
        imported_real = uint8_cuda_tensor_from_ptr(imported_ptr, alloc_size, device)
        local_shadow = shadow_tensor_for(local_real)
        imported_shadow = shadow_tensor_for(imported_real)

        imported_real.fill_(11)
        assert torch.all(local_real == 11).item()
        imported_shadow.fill_(5)
        assert torch.all(local_shadow == 5).item()
    finally:
        if imported_ptr != 0:
            free_allocation(imported_ptr, device)
        gsan_free(real_ptr, device)


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


@pytest.mark.skipif(not is_cuda() or torch.cuda.device_count() < 2, reason="requires at least two CUDA devices")
def test_configure_supports_swapped_cuda_device_ids():
    device_ranks = {0: 1, 1: 0}
    result = run_in_process(_run_configure_check, args=(device_ranks, 2))
    assert result.exc is None


@pytest.mark.skipif(not is_cuda() or torch.cuda.device_count() < 2, reason="requires at least two CUDA devices")
def test_configure_supports_sparse_global_device_ids():
    device_ranks = {0: 2, 1: 3}
    result = run_in_process(_run_configure_check, args=(device_ranks, 4))
    assert result.exc is None


@pytest.mark.skipif(not is_cuda(), reason="requires CUDA backend")
def test_configure_exposes_runtime_fields():
    result = run_in_process(_run_configure_runtime_fields_check)
    assert result.exc is None


@pytest.mark.skipif(not is_cuda(), reason="requires CUDA backend")
def test_freeze_config_rejects_later_changes():
    result = run_in_process(_run_freeze_config_check)
    assert result.exc is None


@pytest.mark.skipif(not is_cuda(), reason="requires CUDA backend")
def test_allocator_initialization_rejects_later_config():
    result = run_in_process(_run_allocator_freezes_config_check)
    assert result.exc is None


@pytest.mark.skipif(not is_cuda() or torch.cuda.device_count() < 2, reason="requires at least two CUDA devices")
def test_default_topology_uses_cuda_device_indices():
    assert get_device_rank(0) == 0
    assert get_device_rank(1) == 1


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
    torch.cuda.synchronize()


@pytest.mark.skipif(not is_cuda(), reason="requires CUDA backend")
def test_malloc_free_large_odd_size(_direct_allocator):
    malloc, free, _, _ = _direct_allocator

    ptr = malloc(_ODD_LARGE_ALLOCATION_SIZE)
    assert ptr != 0

    free(ptr)
    torch.cuda.synchronize()


@pytest.mark.skipif(not is_cuda(), reason="requires CUDA backend")
def test_free_invalid_pointer_and_double_free(_direct_allocator):
    malloc, free, _, _ = _direct_allocator

    p0 = malloc(1)
    assert p0 != 0

    free(p0 + 1)  # freeing an invalid pointer should not crash.

    free(p0)
    free(p0)  # double free must be a no-op

    # p0 should become reusable after the valid free above.
    p0_reuse = malloc(1)
    assert p0_reuse == p0

    free(p0_reuse)
    torch.cuda.synchronize()


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

    shadow = shadow_tensor_for(real)
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
def test_export_allocation_memhandle_regions_identifies_real_and_shadow(_direct_allocator):
    malloc, free, _, _ = _direct_allocator
    device = torch.cuda.current_device()

    real_ptr = malloc(4096)
    assert real_ptr != 0

    try:
        exported_real_ptr, exported_real_size, shadow_ptr, shadow_size = export_allocation_memhandle_regions(real_ptr)
        assert exported_real_ptr == real_ptr
        assert exported_real_size > 0
        assert shadow_ptr != 0
        assert shadow_size > 0

        real = uint8_cuda_tensor_from_ptr(exported_real_ptr, exported_real_size, device)
        shadow = shadow_tensor_for(real)
        assert shadow.data_ptr() == shadow_ptr
        assert shadow.numel() * shadow.element_size() == shadow_size
    finally:
        free(real_ptr)


@pytest.mark.skipif(not is_cuda(), reason="requires CUDA backend")
def test_export_allocation_memhandle_regions_accepts_interior_pointer(_direct_allocator):
    malloc, free, _, _ = _direct_allocator
    device = torch.cuda.current_device()

    real_ptr = malloc(4096)
    assert real_ptr != 0

    try:
        exported_real_ptr, exported_real_size, shadow_ptr, shadow_size = export_allocation_memhandle_regions(real_ptr +
                                                                                                             128)
        assert exported_real_ptr == real_ptr
        assert exported_real_size > 128
        assert shadow_ptr != 0
        assert shadow_size > 0

        real = uint8_cuda_tensor_from_ptr(exported_real_ptr, exported_real_size, device)
        shadow = shadow_tensor_for(real)
        assert shadow.data_ptr() == shadow_ptr
        assert shadow.numel() * shadow.element_size() == shadow_size
    finally:
        free(real_ptr)


@pytest.mark.skipif(not is_cuda(), reason="requires CUDA backend")
@pytest.mark.parametrize("size", [4096, _ODD_LARGE_ALLOCATION_SIZE])
def test_export_import_allocation_handles_maps_real_and_shadow(_direct_allocator, size):
    malloc, free, reserve_ptr, reserve_size = _direct_allocator
    device = torch.cuda.current_device()

    real_ptr = malloc(size)
    assert real_ptr != 0

    imported_ptr = 0
    real_fd = -1
    shadow_fd = -1
    try:
        real_fd, shadow_fd, alloc_size = export_allocation_handles(
            real_ptr,
            ShareableHandleType.POSIX_FILE_DESCRIPTOR,
        )
        assert isinstance(real_fd, int)
        assert isinstance(shadow_fd, int)
        assert alloc_size > 0

        imported_ptr = import_allocation_handles(
            real_fd,
            shadow_fd,
            alloc_size,
            device,
            ShareableHandleType.POSIX_FILE_DESCRIPTOR,
        )
        assert imported_ptr != 0
        assert imported_ptr != real_ptr

        local_real = uint8_cuda_tensor_from_ptr(real_ptr, alloc_size, device)
        imported_real = uint8_cuda_tensor_from_ptr(imported_ptr, alloc_size, device)

        local_shadow = shadow_tensor_for(local_real)
        imported_shadow = shadow_tensor_for(imported_real)
        assert local_shadow.numel() == imported_shadow.numel()

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


@pytest.mark.skipif(not is_cuda(), reason="requires CUDA backend")
@pytest.mark.parametrize(
    ("explicit_config", "allocator_config"),
    [
        pytest.param(True, "fabric_handles:False", id="explicit-config"),
        pytest.param(False, "fabric_handles:True", id="pytorch-config-default"),
    ],
)
def test_export_import_fabric_handles(explicit_config, allocator_config):
    result = run_in_process(
        _run_export_import_fabric_handles_check,
        args=(explicit_config, ),
        env={"PYTORCH_CUDA_ALLOC_CONF": allocator_config},
    )
    if (isinstance(result.exc, RuntimeError) and str(result.exc) == "gsanExportRuntimeStateHandle failed."
            and "operation not permitted" in result.driver_stderr_output.lower()):
        pytest.skip("CUDA fabric handles require an accessible NVIDIA IMEX channel")
    assert result.exc is None
