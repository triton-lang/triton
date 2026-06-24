from __future__ import annotations

import functools
from enum import IntEnum
from pathlib import Path
from types import ModuleType
from typing import Literal, overload

from triton.runtime import driver as runtime_driver
from triton.runtime.build import compile_module_from_file

_THIS_DIR = Path(__file__).resolve().parent
_GSAN_SOURCE_PATH = _THIS_DIR / "src" / "GSanAllocator.cc"


class ShareableHandleType(IntEnum):
    POSIX_FILE_DESCRIPTOR = 0x1
    FABRIC = 0x8


@functools.lru_cache()
def _load_gsan_module() -> ModuleType:
    if runtime_driver.active.get_current_target().backend != "cuda":
        raise RuntimeError("GSan allocator requires the CUDA backend.")

    from triton.backends.nvidia.driver import library_dirs, include_dirs

    return compile_module_from_file(
        src_path=str(_GSAN_SOURCE_PATH),
        name="gsan_allocator",
        library_dirs=library_dirs(),
        include_dirs=include_dirs,
        libraries=["libcuda.so.1"],
    )


@functools.lru_cache()
def _compile_gsan_allocator() -> str:
    # __file__ for a compiled module is the so file
    return _load_gsan_module().__file__


@functools.lru_cache()
def get_allocator():
    from torch.cuda.memory import CUDAPluggableAllocator
    so_name = _compile_gsan_allocator()
    return CUDAPluggableAllocator(so_name, "gsanMalloc", "gsanFree")


def configure(
    *,
    device_ranks: dict[int, int] | None = None,
    num_devices: int | None = None,
    rng_seed: int | None = None,
    clock_buffer_size: int | None = None,
    handle_type: ShareableHandleType | None = None,
) -> None:
    """Configures the process-local GSan state.

    GSan keeps one allocator configuration per process. Call this before the
    allocator initializes runtime state, or before calling :func:`freeze_config`.
    Once frozen, later calls to :func:`configure` raise ``RuntimeError``.

    Args:
        device_ranks (dict[int, int], optional): Mapping from local CUDA device index to the
            logical GSan device id. This enables gsan to be used with multi-node nvlink domains,
            or simply processes with different CUDA_VISIBLE_DEVICES settings. Each value must be
            unique and in ``[0, num_devices)``. If None, defaults to a 1:1 mapping from device
            index to device id.
        num_devices (int, optional): Total number of logical GSan devices in the topology. If
            None, defaults to the number of visible CUDA devices.
        rng_seed (int, optional): Optional seed for GSan's stochastic read-clock sampling. Use this
            to make sampling decisions reproducible across runs when debugging sanitizer behavior.
            If omitted, GSan first checks ``TRITON_GSAN_SEED`` and otherwise generates a random
            seed when the allocator runtime state is initialized.
        clock_buffer_size (int, optional): When doing an atomic release operation, GSan uses a
            circular buffer to record what memory accesses have been released. If the writing CTA
            has done more release writes than there are circular buffer entries, then the atomic
            flag cannot be read and you will need to increase the buffer size. If omitted, GSan
            first checks ``TRITON_GSAN_CLOCK_BUFFER_SIZE`` and otherwise defaults to 1024.
        handle_type (ShareableHandleType, optional): Type of shareable handle requested for GSan
            allocations. If omitted, GSan uses fabric handles when ``PYTORCH_CUDA_ALLOC_CONF``
            contains ``fabric_handles:True`` and otherwise uses POSIX file descriptors.
    """
    _load_gsan_module().configure(device_ranks, num_devices, rng_seed, clock_buffer_size, handle_type)


def freeze_config() -> None:
    """Prevents later `configure(...)` calls from changing allocator configuration."""
    _load_gsan_module().freeze_config()


def create_mem_pool():
    from torch.cuda.memory import MemPool
    return MemPool(get_allocator().allocator())


def gsan_malloc(size: int, device: int, stream: int = 0) -> int:
    module = _load_gsan_module()
    return module.malloc(size, device, stream)


def gsan_free(ptr: int, device: int, size: int = 0, stream: int = 0) -> None:
    module = _load_gsan_module()
    module.free(ptr, device, size, stream)


def get_reserve_pointer() -> int:
    return _load_gsan_module().get_reserve_pointer()


def get_reserve_size() -> int:
    return _load_gsan_module().get_reserve_size()


def get_global_state_pointer() -> int:
    return _load_gsan_module().get_global_state_pointer()


def get_device_rank(device: int) -> int:
    return _load_gsan_module().get_device_rank(device)


def get_runtime_state_layout(device: int) -> dict[str, int]:
    module = _load_gsan_module()
    return module.get_runtime_state_layout(device)


@overload
def export_allocation_handles(
    ptr: int,
    handle_type: Literal[ShareableHandleType.POSIX_FILE_DESCRIPTOR],
) -> tuple[int, int, int]:
    ...


@overload
def export_allocation_handles(
    ptr: int,
    handle_type: Literal[ShareableHandleType.FABRIC],
) -> tuple[bytes, bytes, int]:
    ...


def export_allocation_handles(ptr, handle_type):
    module = _load_gsan_module()
    return module.export_allocation_handles(ptr, handle_type)


def export_allocation_memhandle_regions(ptr: int) -> tuple[int, int, int, int]:
    module = _load_gsan_module()
    return module.export_allocation_memhandle_regions(ptr)


@overload
def import_allocation_handles(
    real_handle: int,
    shadow_handle: int,
    alloc_size: int,
    device: int,
    handle_type: Literal[ShareableHandleType.POSIX_FILE_DESCRIPTOR],
) -> int:
    ...


@overload
def import_allocation_handles(
    real_handle: bytes,
    shadow_handle: bytes,
    alloc_size: int,
    device: int,
    handle_type: Literal[ShareableHandleType.FABRIC],
) -> int:
    ...


def import_allocation_handles(
    real_handle,
    shadow_handle,
    alloc_size,
    device,
    handle_type,
):
    module = _load_gsan_module()
    return module.import_allocation_handles(
        real_handle,
        shadow_handle,
        alloc_size,
        device,
        handle_type,
    )


@overload
def export_runtime_state_handle(
    device: int,
    handle_type: Literal[ShareableHandleType.POSIX_FILE_DESCRIPTOR],
) -> tuple[int, int]:
    ...


@overload
def export_runtime_state_handle(
    device: int,
    handle_type: Literal[ShareableHandleType.FABRIC],
) -> tuple[bytes, int]:
    ...


def export_runtime_state_handle(device, handle_type):
    module = _load_gsan_module()
    return module.export_runtime_state_handle(device, handle_type)


@overload
def import_runtime_state_handle(
    handle: int,
    alloc_size: int,
    peer_device: int,
    device: int,
    handle_type: Literal[ShareableHandleType.POSIX_FILE_DESCRIPTOR],
) -> None:
    ...


@overload
def import_runtime_state_handle(
    handle: bytes,
    alloc_size: int,
    peer_device: int,
    device: int,
    handle_type: Literal[ShareableHandleType.FABRIC],
) -> None:
    ...


def import_runtime_state_handle(
    handle,
    alloc_size,
    peer_device,
    device,
    handle_type,
):
    module = _load_gsan_module()
    module.import_runtime_state_handle(
        handle,
        alloc_size,
        peer_device,
        device,
        handle_type,
    )


def free_allocation(ptr: int, device: int) -> None:
    gsan_free(ptr, device, size=0, stream=0)
