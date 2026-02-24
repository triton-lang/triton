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
