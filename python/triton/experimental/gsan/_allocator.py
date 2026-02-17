from __future__ import annotations

import ctypes
import functools
from pathlib import Path

from triton.runtime import driver as runtime_driver
from triton.runtime.build import compile_so_from_file

_THIS_DIR = Path(__file__).resolve().parent
_GSAN_SOURCE_PATH = _THIS_DIR / "src" / "GSanAllocator.cc"


@functools.lru_cache()
def _compile_gsan_allocator() -> str:
    if runtime_driver.active.get_current_target().backend != "cuda":
        raise RuntimeError("GSan allocator requires the CUDA backend.")

    from triton.backends.nvidia.driver import library_dirs, include_dirs

    return compile_so_from_file(
        src_path=str(_GSAN_SOURCE_PATH),
        name="gsan_allocator",
        library_dirs=library_dirs(),
        include_dirs=include_dirs,
        libraries=["libcuda.so.1"],
    )


@functools.lru_cache()
def _load_gsan_library() -> ctypes.CDLL:
    so_path = _compile_gsan_allocator()
    lib = ctypes.CDLL(so_path)
    lib.gsanMalloc.argtypes = [ctypes.c_ssize_t, ctypes.c_int, ctypes.c_void_p]
    lib.gsanMalloc.restype = ctypes.c_void_p
    lib.gsanFree.argtypes = [ctypes.c_void_p, ctypes.c_ssize_t, ctypes.c_int, ctypes.c_void_p]
    lib.gsanFree.restype = None
    lib.gsanGetReservePointer.argtypes = []
    lib.gsanGetReservePointer.restype = ctypes.c_void_p
    lib.gsanGetReserveSize.argtypes = []
    lib.gsanGetReserveSize.restype = ctypes.c_size_t
    lib.gsanExportAllocationHandles.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.gsanExportAllocationHandles.restype = ctypes.c_int
    lib.gsanImportAllocationHandles.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_size_t, ctypes.c_int]
    lib.gsanImportAllocationHandles.restype = ctypes.c_void_p
    return lib


@functools.lru_cache()
def get_allocator():
    from torch.cuda.memory import CUDAPluggableAllocator
    so_name = _compile_gsan_allocator()
    return CUDAPluggableAllocator(so_name, "gsanMalloc", "gsanFree")


def create_mem_pool():
    from torch.cuda.memory import MemPool
    return MemPool(get_allocator().allocator())


def get_reserve_pointer() -> int:
    return int(_load_gsan_library().gsanGetReservePointer())


def get_reserve_size() -> int:
    return int(_load_gsan_library().gsanGetReserveSize())


def export_allocation_handles(ptr: int) -> tuple[int, int, int]:
    lib = _load_gsan_library()
    real_fd = ctypes.c_int(-1)
    shadow_fd = ctypes.c_int(-1)
    alloc_size = ctypes.c_size_t(0)
    rc = lib.gsanExportAllocationHandles(
        ctypes.c_void_p(int(ptr)),
        ctypes.byref(real_fd),
        ctypes.byref(shadow_fd),
        ctypes.byref(alloc_size),
    )
    if rc != 0:
        raise RuntimeError("gsanExportAllocationHandles failed.")
    return int(real_fd.value), int(shadow_fd.value), int(alloc_size.value)


def import_allocation_handles(real_fd: int, shadow_fd: int, alloc_size: int, device: int) -> int:
    lib = _load_gsan_library()
    ptr = lib.gsanImportAllocationHandles(int(real_fd), int(shadow_fd), int(alloc_size), int(device))
    ptr_int = 0 if ptr is None else int(ptr)
    if ptr_int == 0:
        raise RuntimeError("gsanImportAllocationHandles failed.")
    return ptr_int


def free_allocation(ptr: int, device: int) -> None:
    if ptr == 0:
        return
    lib = _load_gsan_library()
    lib.gsanFree(ctypes.c_void_p(int(ptr)), 0, int(device), ctypes.c_void_p(0))
