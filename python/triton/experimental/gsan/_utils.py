from __future__ import annotations

import ctypes
import torch

_DLPACK_CAPSULE_NAME = b"dltensor"
_DL_UINT = 1
_DL_BITS_UINT8 = 8
_DL_LANES = 1
_DL_CUDA = 2


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


_DLManagedTensorHandle = ctypes.POINTER(_DLManagedTensor)
_DLManagedTensorDeleter = ctypes.CFUNCTYPE(None, _DLManagedTensorHandle)

_DLManagedTensor._fields_ = [
    ("dl_tensor", _DLTensor),
    ("manager_ctx", ctypes.c_void_p),
    ("deleter", _DLManagedTensorDeleter),
]

_PyCapsule_New = ctypes.pythonapi.PyCapsule_New
_PyCapsule_New.restype = ctypes.py_object
_PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]

# Hold ctypes-backed DLPack payloads until the tensor deleter runs.
_DLPACK_STATE: dict[int, tuple[object, object, object]] = {}


@_DLManagedTensorDeleter
def _dl_managed_tensor_deleter(dl_managed_tensor: _DLManagedTensorHandle) -> None:
    if not dl_managed_tensor:
        return
    _DLPACK_STATE.pop(ctypes.addressof(dl_managed_tensor.contents), None)


def uint8_cuda_tensor_from_ptr(data_ptr: int, numel: int, device_index: int) -> torch.Tensor:
    numel = int(numel)
    if numel < 0:
        raise ValueError(f"numel must be >= 0, got {numel}")

    shape = (ctypes.c_int64 * 1)(numel)
    strides = (ctypes.c_int64 * 1)(1)
    dl_managed_tensor = _DLManagedTensor()
    dl_managed_tensor.dl_tensor.data = ctypes.c_void_p(int(data_ptr))
    dl_managed_tensor.dl_tensor.device = _DLDevice(_DL_CUDA, device_index)
    dl_managed_tensor.dl_tensor.ndim = 1
    dl_managed_tensor.dl_tensor.dtype = _DLDataType(_DL_UINT, _DL_BITS_UINT8, _DL_LANES)
    dl_managed_tensor.dl_tensor.shape = ctypes.cast(shape, ctypes.POINTER(ctypes.c_int64))
    dl_managed_tensor.dl_tensor.strides = ctypes.cast(strides, ctypes.POINTER(ctypes.c_int64))
    dl_managed_tensor.dl_tensor.byte_offset = 0
    dl_managed_tensor.manager_ctx = None
    dl_managed_tensor.deleter = _dl_managed_tensor_deleter

    dl_managed_tensor_ptr = ctypes.addressof(dl_managed_tensor)
    _DLPACK_STATE[dl_managed_tensor_ptr] = (dl_managed_tensor, shape, strides)

    try:
        dlpack_capsule = _PyCapsule_New(
            ctypes.c_void_p(dl_managed_tensor_ptr),
            _DLPACK_CAPSULE_NAME,
            None,
        )
        return torch.from_dlpack(dlpack_capsule)
    except Exception:
        _DLPACK_STATE.pop(dl_managed_tensor_ptr, None)
        raise


SHADOW_SIZE_BYTES = 24
SHADOW_GRANULARITY_BYTES = 4


def shadow_region(real_ptr: int, real_size_bytes: int, reserve_ptr: int, reserve_size: int) -> tuple[int, int]:
    real_base = reserve_ptr + reserve_size // 2
    word_offset = (real_ptr - real_base) // SHADOW_GRANULARITY_BYTES
    shadow_ptr = reserve_ptr + word_offset * SHADOW_SIZE_BYTES
    shadow_size = ((real_size_bytes + SHADOW_GRANULARITY_BYTES - 1) // SHADOW_GRANULARITY_BYTES) * SHADOW_SIZE_BYTES
    return shadow_ptr, shadow_size
