from __future__ import annotations

import ctypes

import pytest
import torch

import triton
from triton._internal_testing import is_cuda
from triton.experimental.gsan import create_mem_pool
from triton.experimental.gsan._allocator import _load_gsan_library, get_reserve_pointer, get_reserve_size

_PY_CAPSULE_NEW = ctypes.pythonapi.PyCapsule_New
_PY_CAPSULE_NEW.restype = ctypes.py_object
_PY_CAPSULE_NEW.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]


class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int), ("device_id", ctypes.c_int)]


class _DLDataType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", ctypes.c_int),
        ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class _DLManagedTensor(ctypes.Structure):
    pass


_DLTensorDeleter = ctypes.CFUNCTYPE(None, ctypes.POINTER(_DLManagedTensor))


@_DLTensorDeleter
def _dlpack_noop_deleter(_: ctypes.POINTER(_DLManagedTensor)) -> None:
    # Intentionally empty; shadow memory lifetime is not managed by us.
    return


_DLManagedTensor._fields_ = [
    ("dl_tensor", _DLTensor),
    ("manager_ctx", ctypes.c_void_p),
    ("deleter", _DLTensorDeleter),
]


def _uint8_cuda_tensor_from_ptr(data_ptr: int, numel: int, device_index: int):
    shape = (ctypes.c_int64 * 1)(numel)
    managed = _DLManagedTensor()
    managed.dl_tensor.data = ctypes.c_void_p(data_ptr)
    managed.dl_tensor.device = _DLDevice(2, device_index)  # 2 = kDLCUDA
    managed.dl_tensor.ndim = 1
    managed.dl_tensor.dtype = _DLDataType(1, 8, 1)  # uint8
    managed.dl_tensor.shape = ctypes.cast(shape, ctypes.POINTER(ctypes.c_int64))
    managed.dl_tensor.strides = None
    managed.dl_tensor.byte_offset = 0
    managed.manager_ctx = None
    managed.deleter = _dlpack_noop_deleter
    capsule = _PY_CAPSULE_NEW(ctypes.addressof(managed), b"dltensor", None)
    return torch.utils.dlpack.from_dlpack(capsule)


SHADOW_SIZE_BYTES = 8
SHADOW_GRANULARITY_BYTES = 4


def _shadow_region(real_ptr: int, real_size_bytes: int, reserve_ptr: int, reserve_size: int) -> tuple[int, int]:
    real_base = reserve_ptr + reserve_size // 2
    word_offset = (real_ptr - real_base) // SHADOW_GRANULARITY_BYTES
    shadow_ptr = reserve_ptr + word_offset * SHADOW_SIZE_BYTES
    shadow_size = triton.cdiv(real_size_bytes, SHADOW_GRANULARITY_BYTES) * SHADOW_SIZE_BYTES
    return shadow_ptr, shadow_size


def _shadow_tensor_for(real: torch.Tensor) -> torch.Tensor:
    reserve_ptr = get_reserve_pointer()
    reserve_size = get_reserve_size()
    shadow_ptr, shadow_size = _shadow_region(real.data_ptr(),
                                             real.untyped_storage().nbytes(), reserve_ptr, reserve_size)
    return _uint8_cuda_tensor_from_ptr(shadow_ptr, shadow_size, torch.cuda.current_device())


@pytest.fixture
def _direct_allocator():
    lib = _load_gsan_library()
    device = torch.cuda.current_device()
    stream = ctypes.c_void_p(0)
    reserve_ptr = get_reserve_pointer()
    reserve_size = get_reserve_size()
    allocated = set()

    def malloc(size: int) -> int:
        ptr = lib.gsanMalloc(int(size), device, stream)
        ptr_int = 0 if ptr is None else int(ptr)
        if ptr_int != 0:
            allocated.add(ptr_int)
        return ptr_int

    def free(ptr: int, size: int = 0) -> None:
        lib.gsanFree(ctypes.c_void_p(ptr), int(size), device, stream)
        if ptr in allocated:
            allocated.remove(ptr)

    try:
        yield malloc, free, reserve_ptr, reserve_size
    finally:
        # Cleanup any allocated pointers
        for ptr in list(allocated):
            lib.gsanFree(ctypes.c_void_p(ptr), 0, device, stream)


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
