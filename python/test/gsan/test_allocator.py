from __future__ import annotations

import os

import pytest
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from triton._internal_testing import is_cuda, run_in_process
from triton.experimental.gsan import ShareableHandleType, configure, create_mem_pool, freeze_config, get_allocator, has_live_allocations, reset
from triton.experimental.gsan import _stream_sync
from triton._C.libtriton.gsan_testing import shadow_granularity as get_shadow_granularity, shadow_cell_address
from triton.experimental.gsan._allocator import (
    export_allocation_handles,
    export_allocation_memhandle_regions,
    export_runtime_state_handle,
    free_allocation,
    get_device_rank,
    get_global_state_pointer,
    get_reserve_pointer,
    get_reserve_size,
    gsan_free,
    gsan_malloc,
    import_allocation_handles,
    is_write_once_allocation,
    import_runtime_state_handle,
    supports_fabric_handles,
)
from triton.experimental.gsan._testing_utils import (global_state, shadow_cell_from_address, shadow_tensor_for,
                                                     store_one_i32, thread_state_from_smid)
from triton.experimental.gsan._utils import uint8_cuda_tensor_from_ptr

# With 2 MiB pages, this rounds to a 6 MiB allocation inside an 8 MiB tree node.
# This tests cases where AllocNode.size != AllocNode.allocSize
_ODD_LARGE_ALLOCATION_SIZE = 4 * 1024 * 1024 + 1


def _run_configure_check(device_ranks: dict[int, int], num_devices: int) -> None:
    configure(device_ranks=device_ranks, num_devices=num_devices)
    assert get_device_rank(0) == device_ranks[0]
    assert get_device_rank(1) == device_ranks[1]


def _run_configure_runtime_fields_check(shadow_granularity: int) -> None:
    device = torch.cuda.current_device()
    configure(rng_seed=12345, clock_buffer_size=17)

    ptr = gsan_malloc(1, device, 0, shadow_granularity=shadow_granularity)
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


def _run_allocator_freezes_config_check(shadow_granularity: int) -> None:
    device = torch.cuda.current_device()
    configure(rng_seed=12345)
    ptr = gsan_malloc(1, device, 0, shadow_granularity=shadow_granularity)
    try:
        try:
            configure(rng_seed=12345)
        except RuntimeError as exc:
            assert "configuration is already frozen" in str(exc)
        else:
            raise AssertionError("expected allocator initialization to freeze config")
    finally:
        gsan_free(ptr, device, 0, 0)


@gluon.jit
def _store_reset_target(ptr, WIDTH: gl.constexpr):
    offsets = gl.arange(0, WIDTH, layout=gl.BlockedLayout([4], [32], [1], [0]))
    gl.store(ptr + offsets, 1)


def _run_has_live_allocations_check(shadow_granularity) -> None:
    assert has_live_allocations() is False
    configure(rng_seed=12345)
    device = torch.cuda.current_device()
    ptr = gsan_malloc(1, device, shadow_granularity=shadow_granularity)
    try:
        assert has_live_allocations() is True
    finally:
        gsan_free(ptr, device)
    assert has_live_allocations() is False


def _run_reset_rejects_live_allocations_check(shadow_granularity) -> None:
    device = torch.cuda.current_device()
    stream = torch.cuda.current_stream().cuda_stream
    ptr = gsan_malloc(max(4, shadow_granularity), device, shadow_granularity=shadow_granularity)
    target = uint8_cuda_tensor_from_ptr(ptr, max(4, shadow_granularity), device).view(torch.int32)
    try:
        with triton.knobs.compilation.scope():
            triton.knobs.compilation.instrumentation_mode = "gsan"
            _store_reset_target[(1, )](target, WIDTH=target.numel(), num_warps=1)
            torch.cuda.synchronize()

            stream_state = _stream_sync._launch_stream_state(device, stream)
            runtime_layout = _stream_sync._runtime_state_layout(get_device_rank(device), device)
            clocks_before = stream_state.clocks.cpu()
            shadow_before = shadow_tensor_for(target).cpu()
            thread_id = shadow_cell_from_address(ptr).write_clock.thread_id
            thread_state_before = thread_state_from_smid(thread_id)
            assert stream_state.next_kernel_id == 1

            with pytest.raises(AssertionError, match="GSan allocations are still live"):
                reset()

            assert has_live_allocations() is True
            assert _stream_sync._launch_stream_state(device, stream) is stream_state
            assert _stream_sync._runtime_state_layout(get_device_rank(device), device) is runtime_layout
            assert stream_state.next_kernel_id == 1
            assert torch.equal(stream_state.clocks.cpu(), clocks_before)
            assert torch.equal(shadow_tensor_for(target).cpu(), shadow_before)
            assert thread_state_from_smid(thread_id).vector_clock == thread_state_before.vector_clock

            _store_reset_target[(1, )](target, WIDTH=target.numel(), num_warps=1)
            torch.cuda.synchronize()
            assert stream_state.next_kernel_id == 2
    finally:
        del target
        gsan_free(ptr, device)

    reset()


def _run_reset_collects_unreachable_allocations_check(shadow_granularity) -> None:
    device = torch.cuda.current_device()

    class CyclicAllocation:

        def __init__(self):
            self.ptr = gsan_malloc(1, device, shadow_granularity=shadow_granularity)
            self.cycle = self

        def __del__(self):
            gsan_free(self.ptr, device)

    allocation = CyclicAllocation()
    del allocation
    reset()


def _run_reset_reinitializes_runtime_state_check(shadow_granularity) -> None:
    device = torch.cuda.current_device()
    configure(rng_seed=12345, clock_buffer_size=17)
    original_state_pointer = get_global_state_pointer()

    ptr = gsan_malloc(max(4, shadow_granularity), device, shadow_granularity=shadow_granularity)
    target = uint8_cuda_tensor_from_ptr(ptr, max(4, shadow_granularity), device).view(torch.int32)
    with triton.knobs.compilation.scope():
        triton.knobs.compilation.instrumentation_mode = "gsan"
        _store_reset_target[(1, )](target, WIDTH=target.numel(), num_warps=1)
    torch.cuda.synchronize()

    thread_id = shadow_cell_from_address(ptr).write_clock.thread_id
    old_thread_state = thread_state_from_smid(thread_id)
    assert old_thread_state.globals_ptr != 0
    assert old_thread_state.vector_clock[thread_id] != 0
    assert _stream_sync._launch_stream_state.cache_info().currsize == 1
    assert _stream_sync._runtime_state_layout.cache_info().currsize == 1

    del target
    gsan_free(ptr, device)
    reset()

    assert get_global_state_pointer() == original_state_pointer
    assert _stream_sync._launch_stream_state.cache_info().currsize == 0
    assert _stream_sync._runtime_state_layout.cache_info().currsize == 0

    state = global_state(device_index=device)
    assert state.rng_seed == 12345
    assert state.clock_buffer_size == 17
    with pytest.raises(RuntimeError, match="configuration is already frozen"):
        configure(rng_seed=12345)

    new_thread_state = thread_state_from_smid(thread_id)
    assert new_thread_state.globals_ptr == 0
    assert all(epoch == 0 for epoch in new_thread_state.vector_clock)

    ptr = gsan_malloc(max(4, shadow_granularity), device, shadow_granularity=shadow_granularity)
    target = uint8_cuda_tensor_from_ptr(ptr, max(4, shadow_granularity), device).view(torch.int32)
    try:
        with triton.knobs.compilation.scope():
            triton.knobs.compilation.instrumentation_mode = "gsan"
            _store_reset_target[(1, )](target, WIDTH=target.numel(), num_warps=1)
        torch.cuda.synchronize()
        stream = torch.cuda.current_stream().cuda_stream
        assert _stream_sync._launch_stream_state(device, stream).next_kernel_id == 1
    finally:
        del target
        gsan_free(ptr, device)


def _run_reset_releases_cached_pool_clocks_check(num_pools: int, shadow_granularity) -> None:
    pools = [create_mem_pool(shadow_granularity=shadow_granularity) for _ in range(num_pools)]
    streams = [torch.cuda.Stream() for _ in pools]
    for pool, stream in zip(pools, streams):
        with torch.cuda.stream(stream), torch.cuda.use_mem_pool(pool), triton.knobs.compilation.scope():
            triton.knobs.compilation.instrumentation_mode = "gsan"
            target = torch.zeros(max(1, shadow_granularity // 4), dtype=torch.int32, device="cuda")
            _store_reset_target[(1, )](target, WIDTH=target.numel(), num_warps=1)
        del target
    torch.cuda.synchronize()

    del pool
    del pools
    reset()


def _run_launch_stream_clocks_use_private_pool_check(shadow_granularity) -> None:
    device = torch.cuda.current_device()
    pool = create_mem_pool(shadow_granularity=shadow_granularity)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream), torch.cuda.use_mem_pool(pool):
        before = torch.empty(1, device=device)
        clocks, kernel_id = _stream_sync.get_launch_stream_clock(device, stream.cuda_stream)
        after = torch.empty(1, device=device)
    stream.synchronize()

    reserve_begin = get_reserve_pointer()
    reserve_end = reserve_begin + get_reserve_size()
    assert reserve_begin <= before.data_ptr() < reserve_end
    assert not reserve_begin <= clocks.data_ptr() < reserve_end
    assert reserve_begin <= after.data_ptr() < reserve_end
    assert kernel_id == 0
    assert torch.count_nonzero(clocks).item() == 0

    del before, after, clocks, pool
    reset()


def _run_export_import_fabric_handles_check(explicit_config: bool, shadow_granularity: int, write_once: bool) -> None:
    device = torch.cuda.current_device()
    configure(
        device_ranks={device: 0},
        num_devices=2,
        handle_type=ShareableHandleType.FABRIC if explicit_config else None,
    )
    real_ptr = gsan_malloc(4096, device, shadow_granularity=shadow_granularity, write_once=write_once)
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

        # Keep coverage of the legacy three-field export/default import API.
        if shadow_granularity == 4 or write_once:
            real_handle, shadow_handle, alloc_size = export_allocation_handles(real_ptr, ShareableHandleType.FABRIC,
                                                                               write_once=write_once)
            import_kwargs = {"write_once": write_once}
        else:
            real_handle, shadow_handle, alloc_size, exported_granularity = export_allocation_handles(
                real_ptr, ShareableHandleType.FABRIC, include_granularity=True)
            assert exported_granularity == shadow_granularity
            import_kwargs = {"shadow_granularity": exported_granularity}
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
            **import_kwargs,
        )
        assert get_shadow_granularity(imported_ptr) == shadow_granularity
        assert is_write_once_allocation(imported_ptr) == write_once
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
def _direct_allocator(shadow_granularity):
    if not is_cuda():
        pytest.skip("requires CUDA backend")
    device = torch.cuda.current_device()
    stream = 0
    reserve_ptr = get_reserve_pointer()
    reserve_size = get_reserve_size()
    allocated = set()

    def malloc(size: int) -> int:
        ptr_int = gsan_malloc(size, device, stream, shadow_granularity=shadow_granularity)
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


@pytest.mark.xdist_group("gsan-multi-gpu")
@pytest.mark.skipif(not is_cuda() or torch.cuda.device_count() < 2, reason="requires at least two CUDA devices")
def test_configure_supports_swapped_cuda_device_ids():
    device_ranks = {0: 1, 1: 0}
    result = run_in_process(_run_configure_check, args=(device_ranks, 2))
    assert result.exc is None


@pytest.mark.xdist_group("gsan-multi-gpu")
@pytest.mark.skipif(not is_cuda() or torch.cuda.device_count() < 2, reason="requires at least two CUDA devices")
def test_configure_supports_sparse_global_device_ids():
    device_ranks = {0: 2, 1: 3}
    result = run_in_process(_run_configure_check, args=(device_ranks, 4))
    assert result.exc is None


@pytest.mark.skipif(not is_cuda(), reason="requires CUDA backend")
def test_configure_exposes_runtime_fields(shadow_granularity):
    result = run_in_process(_run_configure_runtime_fields_check, args=(shadow_granularity, ))
    assert result.exc is None


@pytest.mark.skipif(not is_cuda(), reason="requires CUDA backend")
def test_freeze_config_rejects_later_changes():
    result = run_in_process(_run_freeze_config_check)
    assert result.exc is None


@pytest.mark.skipif(not is_cuda(), reason="requires CUDA backend")
def test_allocator_initialization_rejects_later_config(shadow_granularity):
    result = run_in_process(_run_allocator_freezes_config_check, args=(shadow_granularity, ))
    assert result.exc is None


@pytest.mark.skipif(not is_cuda(), reason="requires CUDA backend")
def test_has_live_allocations(shadow_granularity):
    result = run_in_process(_run_has_live_allocations_check, args=(shadow_granularity, ))
    assert result.exc is None


@pytest.mark.skipif(not is_cuda(), reason="requires CUDA backend")
def test_reset_rejects_live_allocations(shadow_granularity):
    result = run_in_process(_run_reset_rejects_live_allocations_check, args=(shadow_granularity, ))
    assert result.exc is None


@pytest.mark.skipif(not is_cuda(), reason="requires CUDA backend")
def test_reset_collects_unreachable_allocations(shadow_granularity):
    result = run_in_process(_run_reset_collects_unreachable_allocations_check, args=(shadow_granularity, ))
    assert result.exc is None


@pytest.mark.skipif(not is_cuda(), reason="requires CUDA backend")
def test_reset_reinitializes_runtime_state(shadow_granularity):
    result = run_in_process(_run_reset_reinitializes_runtime_state_check, args=(shadow_granularity, ))
    assert result.exc is None


@pytest.mark.skipif(not is_cuda(), reason="requires CUDA backend")
@pytest.mark.parametrize("num_pools", [1, 2])
def test_reset_releases_cached_pool_clocks(num_pools, shadow_granularity):
    result = run_in_process(_run_reset_releases_cached_pool_clocks_check, args=(num_pools, shadow_granularity))
    assert result.exc is None


@pytest.mark.skipif(not is_cuda(), reason="requires CUDA backend")
def test_launch_stream_clocks_use_private_pool(shadow_granularity):
    result = run_in_process(_run_launch_stream_clocks_use_private_pool_check, args=(shadow_granularity, ))
    assert result.exc is None


@pytest.mark.xdist_group("gsan-multi-gpu")
@pytest.mark.skipif(not is_cuda() or torch.cuda.device_count() < 2, reason="requires at least two CUDA devices")
def test_default_topology_uses_cuda_device_indices():
    assert get_device_rank(0) == 0
    assert get_device_rank(1) == 1


def test_malloc_free(_direct_allocator, shadow_granularity):
    malloc, free, reserve_ptr, reserve_size = _direct_allocator
    pool_index = (1, 4, 16).index(shadow_granularity)
    real_base = reserve_ptr + (2 * pool_index + 1) * reserve_size // 8

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
def test_mem_pool(shadow_granularity):
    pool = create_mem_pool(shadow_granularity=shadow_granularity)
    with torch.cuda.use_mem_pool(pool):
        real = torch.empty(4096, dtype=torch.uint8, device="cuda")

    reserve_ptr = get_reserve_pointer()
    reserve_size = get_reserve_size()
    assert reserve_ptr != 0
    assert reserve_size > 0

    assert reserve_ptr <= real.data_ptr() < reserve_ptr + reserve_size
    assert get_shadow_granularity(real.data_ptr()) == shadow_granularity
    assert shadow_cell_address(real.data_ptr() + shadow_granularity - 1) == shadow_cell_address(real.data_ptr())
    assert shadow_cell_address(real.data_ptr() + shadow_granularity) == shadow_cell_address(real.data_ptr()) + 24

    shadow = shadow_tensor_for(real)
    assert reserve_ptr <= shadow.data_ptr() < real.data_ptr()
    assert shadow.numel() == real.numel() // shadow_granularity * 24

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
def test_export_import_allocation_handles_maps_real_and_shadow(_direct_allocator, size, shadow_granularity):
    malloc, free, reserve_ptr, reserve_size = _direct_allocator
    device = torch.cuda.current_device()

    real_ptr = malloc(size)
    assert real_ptr != 0

    imported_ptr = 0
    real_fd = -1
    shadow_fd = -1
    try:
        real_fd, shadow_fd, alloc_size, exported_granularity = export_allocation_handles(
            real_ptr,
            ShareableHandleType.POSIX_FILE_DESCRIPTOR,
            include_granularity=True,
        )
        assert isinstance(real_fd, int)
        assert isinstance(shadow_fd, int)
        assert alloc_size > 0
        assert exported_granularity == shadow_granularity
        if shadow_granularity != 4:
            with pytest.raises(ValueError, match="include_granularity=True"):
                export_allocation_handles(real_ptr, ShareableHandleType.POSIX_FILE_DESCRIPTOR)

        imported_ptr = import_allocation_handles(
            real_fd,
            shadow_fd,
            alloc_size,
            device,
            ShareableHandleType.POSIX_FILE_DESCRIPTOR,
            shadow_granularity=exported_granularity,
        )
        assert imported_ptr != 0
        assert imported_ptr != real_ptr
        assert get_shadow_granularity(imported_ptr) == shadow_granularity

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


@pytest.mark.parametrize("granularity", [0, 2, 8, 32, -1, True, 1.0, "1"])
@pytest.mark.parametrize("factory", [get_allocator, create_mem_pool])
def test_mem_pool_rejects_invalid_granularity(factory, granularity):
    with pytest.raises(ValueError, match="shadow_granularity must be 1, 4, or 16"):
        factory(shadow_granularity=granularity)


@pytest.mark.parametrize("granularity", [4, 16])
@pytest.mark.parametrize("factory", [get_allocator, create_mem_pool])
def test_write_once_rejects_non_byte_granularity(factory, granularity):
    with pytest.raises(ValueError, match="write_once allocations require shadow_granularity=1"):
        factory(write_once=True, shadow_granularity=granularity)


@pytest.mark.skipif(not is_cuda(), reason="requires CUDA backend")
def test_mem_pools_have_distinct_regions():
    tensors = []
    pools = [create_mem_pool(shadow_granularity=g) for g in (1, 4, 16)]
    pools.append(create_mem_pool(write_once=True))
    for pool in pools:
        with torch.cuda.use_mem_pool(pool):
            tensors.append(torch.empty(4096, dtype=torch.uint8, device="cuda"))
    assert [get_shadow_granularity(t.data_ptr()) for t in tensors] == [1, 4, 16, 1]
    assert len({shadow_cell_address(t.data_ptr()) for t in tensors}) == 4
    assert [is_write_once_allocation(t.data_ptr()) for t in tensors] == [False, False, False, True]
    assert get_allocator() is get_allocator(shadow_granularity=4)
    assert get_allocator(write_once=True) is get_allocator(write_once=True, shadow_granularity=1)


@pytest.mark.skipif(not is_cuda(), reason="requires CUDA backend")
@pytest.mark.parametrize(
    ("explicit_config", "allocator_config"),
    [
        pytest.param(True, "fabric_handles:False", id="explicit-config"),
        pytest.param(False, "fabric_handles:True", id="pytorch-config-default"),
    ],
)
@pytest.mark.parametrize("write_once", [False, True])
def test_export_import_fabric_handles(explicit_config, allocator_config, write_once, shadow_granularity):
    if write_once and shadow_granularity != 1:
        pytest.skip("write_once allocations always use byte granularity")
    if not supports_fabric_handles(torch.cuda.current_device()):
        pytest.skip("CUDA device does not support fabric handles")
    result = run_in_process(
        _run_export_import_fabric_handles_check,
        args=(explicit_config, shadow_granularity, write_once),
        env={"PYTORCH_CUDA_ALLOC_CONF": allocator_config},
    )
    if (isinstance(result.exc, RuntimeError) and str(result.exc) == "gsanExportRuntimeStateHandle failed."
            and "operation not permitted" in result.driver_stderr_output.lower()):
        pytest.skip("CUDA fabric handles require an accessible NVIDIA IMEX channel")
    assert result.exc is None


@pytest.mark.parametrize("write_once", [False, True])
def test_allocation_mode_and_shadow_size(write_once):
    device = torch.cuda.current_device()
    ptr = gsan_malloc(513, device, write_once=write_once)
    assert ptr
    try:
        assert is_write_once_allocation(ptr) == write_once
        assert is_write_once_allocation(ptr + 512) == write_once
        real_ptr, real_size, shadow_ptr, shadow_size = export_allocation_memhandle_regions(ptr + 1)
        assert real_ptr == ptr
        assert shadow_size == real_size * (4 if write_once else 6)
        shadow = uint8_cuda_tensor_from_ptr(shadow_ptr, shadow_size, device)
        assert torch.count_nonzero(shadow).item() == 0
        shadow.fill_(5)
        torch.cuda.synchronize()
    finally:
        gsan_free(ptr, device)
    with pytest.raises(ValueError, match="live GSan allocation"):
        is_write_once_allocation(ptr)
    reused = gsan_malloc(513, device, write_once=write_once)
    try:
        assert reused == ptr
        _, _, shadow_ptr, shadow_size = export_allocation_memhandle_regions(reused)
        shadow = uint8_cuda_tensor_from_ptr(shadow_ptr, shadow_size, device)
        assert torch.count_nonzero(shadow).item() == 0
    finally:
        gsan_free(reused, device)


def test_write_once_export_import_preserves_shadow(fresh_knobs):
    triton.knobs.compilation.instrumentation_mode = "gsan"
    device = torch.cuda.current_device()
    ptr = gsan_malloc(513, device, write_once=True)
    imported = 0
    fds = []
    try:
        with pytest.raises(ValueError, match="allocation mode"):
            export_allocation_handles(ptr, ShareableHandleType.POSIX_FILE_DESCRIPTOR)
        real_fd, shadow_fd, size = export_allocation_handles(ptr, ShareableHandleType.POSIX_FILE_DESCRIPTOR,
                                                             write_once=True)
        fds = [real_fd, shadow_fd]
        imported = import_allocation_handles(real_fd, shadow_fd, size, device,
                                             ShareableHandleType.POSIX_FILE_DESCRIPTOR, write_once=True)
        assert is_write_once_allocation(imported + 1)
        original = uint8_cuda_tensor_from_ptr(ptr, 513, device)
        alias = uint8_cuda_tensor_from_ptr(imported, 513, device)
        store_one_i32[(1, )](original)
        assert alias[0].item() == 1
        assert shadow_cell_from_address(imported) == shadow_cell_from_address(ptr)
        assert shadow_cell_from_address(imported).epoch > 0
        assert shadow_cell_from_address(imported + 1).epoch == 0
    finally:
        for fd in fds:
            os.close(fd)
        if imported:
            free_allocation(imported, device)
        gsan_free(ptr, device)


def _run_write_once_live_allocation_check():
    device = torch.cuda.current_device()
    ptr = gsan_malloc(1, device, write_once=True)
    assert has_live_allocations()
    with pytest.raises(AssertionError, match="allocations are still live"):
        reset()
    gsan_free(ptr, device)
    assert not has_live_allocations()
    reset()


def test_write_once_live_allocation_blocks_reset():
    result = run_in_process(_run_write_once_live_allocation_check)
    assert result.exc is None, result.exc
