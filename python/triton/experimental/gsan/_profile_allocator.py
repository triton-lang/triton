"""Independent allocator for profiling data and its compact observation cells."""
from __future__ import annotations

import functools
from contextlib import contextmanager

from ._allocator_common import ShareableHandleType, load_allocator

_installed = None


@functools.lru_cache()
def _load_module():
    return load_allocator("ProfileCUDAAllocator.cc", "gsan_profile_allocator", torch_integration=True,
                          extra_sources=("ProfileAllocator.cc", ))


def install_allocator() -> None:
    """Install once at process startup, before PyTorch initializes CUDA.

    Importing gsan.profile does not load or install this allocator.
    Ordinary allocations retain their original allocator. AllocationProfile
    enables profiling for tensor storage on its creating thread/devices.
    Raw-pointer allocations continue to use the original allocator.
    """
    global _installed
    if _installed is not None:
        return
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError("install the profiling allocator before PyTorch initializes CUDA")
    module = _load_module()
    module._register_cuda_allocator()
    native = module.CUDAAllocator(torch._C._cuda_getAllocator())
    torch._C._cuda_changeCurrentAllocator(native)
    _installed = native


def is_installed() -> bool:
    return _installed is not None


def get_allocator():
    """Return the installed native CUDAAllocator."""
    if _installed is None:
        raise RuntimeError("call profile.install_allocator() at startup, before PyTorch initializes CUDA")
    return _installed


@contextmanager
def _suspend():
    """Keep profiler bookkeeping outside the profiling allocator."""
    native = get_allocator()
    enabled = native.set_enabled(False, [])
    try:
        yield
    finally:
        native.set_enabled(enabled, [])


def configure(*, handle_type: ShareableHandleType) -> None:
    """Select CUDA IPC or FABRIC handles before the first profiling allocation.

    GSan's topology, clock settings, and allocator configuration are independent.
    Without an explicit setting, follow ``PYTORCH_CUDA_ALLOC_CONF``.
    """
    _load_module().configure(handle_type)


def malloc(size: int, device: int, stream: int = 0) -> int:
    """Allocate profiling storage; the caller must release it with free_allocation."""
    return _load_module().malloc(size, device, stream)


def free_allocation(ptr: int, device: int, stream: int = 0) -> None:
    """Release a complete owned or imported mapping after its last use."""
    _load_module().free(ptr, device, stream)


def get_reserve_pointer() -> int:
    return _load_module().get_reserve_pointer()


def get_allocation_profile_layout(ptr: int) -> tuple[int, int, int] | None:
    """Return backing base/size and metadata address; ignore other allocators."""
    return _load_module().get_allocation_profile_layout(ptr)


def export_allocation_memhandle_regions(ptr: int) -> tuple[int, int, int, int]:
    return _load_module().export_allocation_memhandle_regions(ptr)


def export_allocation_handles(ptr: int, handle_type: ShareableHandleType) -> tuple[int | bytes, int | bytes, int]:
    """Export data/metadata handles and the backing size for a profiling allocation.

    Exchange all three fields with peers and import through this module. CUDA IPC
    file descriptors must be transferred with SCM_RIGHTS; FABRIC bytes can be
    exchanged directly. Exported file descriptors belong to the caller.
    """
    return _load_module().export_allocation_handles(ptr, handle_type)


def import_allocation_handles(real_handle: int | bytes, shadow_handle: int | bytes, alloc_size: int, device: int,
                              handle_type: ShareableHandleType) -> int:
    """Map shared data and observations on a launch device, without race clocks.

    Handles must come from the profiling allocator. Keep the mapping alive until
    all kernels and reports using it finish, then call :func:`free_allocation`.
    """
    return _load_module().import_allocation_handles(real_handle, shadow_handle, alloc_size, device, handle_type)


def has_live_allocations() -> bool:
    return _load_module().has_live_allocations()


def reset() -> None:
    """Release cached profiling mappings once all storages and imports are freed.

    The native allocator remains installed; later contexts can allocate again.
    This explicit cleanup may synchronize, unlike storage deallocation.
    """
    from .profile import _active
    if _active.get() is not None:
        raise RuntimeError("cannot reset the profiling allocator in an active profile")
    if _installed is not None:
        if _installed.has_live_storage():
            raise RuntimeError("cannot reset profiling allocator with live allocations")
        import torch
        for device in range(torch.cuda.device_count()):
            torch.cuda.synchronize(device)
        _installed.empty_cache()
    _load_module().reset()
