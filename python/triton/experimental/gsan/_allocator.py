from __future__ import annotations

import functools
from pathlib import Path
from types import ModuleType

from triton.runtime import driver as runtime_driver
from triton.runtime.build import compile_module_from_file

_THIS_DIR = Path(__file__).resolve().parent
_GSAN_SOURCE_PATH = _THIS_DIR / "src" / "GSanAllocator.cc"


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
    return str(_load_gsan_module().__file__)


@functools.lru_cache()
def get_allocator():
    from torch.cuda.memory import CUDAPluggableAllocator
    so_name = _compile_gsan_allocator()
    return CUDAPluggableAllocator(so_name, "gsanMalloc", "gsanFree")


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


def export_allocation_handles(ptr: int) -> tuple[int, int, int]:
    module = _load_gsan_module()
    return module.export_allocation_handles(ptr)


def import_allocation_handles(real_fd: int, shadow_fd: int, alloc_size: int, device: int) -> int:
    module = _load_gsan_module()
    return module.import_allocation_handles(real_fd, shadow_fd, alloc_size,
                                            device)


def free_allocation(ptr: int, device: int) -> None:
    if ptr == 0:
        return
    gsan_free(ptr, device, size=0, stream=0)
