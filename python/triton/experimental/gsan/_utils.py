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

# Keep the ctypes-owned metadata alive until PyTorch drops the imported tensor.
_DLPACK_STATE: dict[int, object] = {}


@_DLManagedTensorDeleter
def _dl_managed_tensor_deleter(dl_managed_tensor: _DLManagedTensorHandle) -> None:
    if not dl_managed_tensor:
        return
    _DLPACK_STATE.pop(ctypes.addressof(dl_managed_tensor.contents), None)


class _DLPackCudaPtrView:

    def __init__(self, data_ptr: int, numel: int, device_index: int):
        self._managed_tensor = _DLManagedTensor()
        self._shape = (ctypes.c_int64 * 1)(numel)
        self._strides = (ctypes.c_int64 * 1)(1)

        self._managed_tensor.dl_tensor.data = ctypes.c_void_p(data_ptr)
        self._managed_tensor.dl_tensor.device = _DLDevice(_DL_CUDA, device_index)
        self._managed_tensor.dl_tensor.ndim = 1
        self._managed_tensor.dl_tensor.dtype = _DLDataType(_DL_UINT, _DL_BITS_UINT8, _DL_LANES)
        self._managed_tensor.dl_tensor.shape = ctypes.cast(self._shape, ctypes.POINTER(ctypes.c_int64))
        self._managed_tensor.dl_tensor.strides = ctypes.cast(self._strides, ctypes.POINTER(ctypes.c_int64))
        self._managed_tensor.dl_tensor.byte_offset = 0
        self._managed_tensor.manager_ctx = None
        self._managed_tensor.deleter = _dl_managed_tensor_deleter

    def __dlpack_device__(self) -> tuple[int, int]:
        device = self._managed_tensor.dl_tensor.device
        return int(device.device_type), int(device.device_id)

    def __dlpack__(self, stream: int | None = None):
        # These pointer views do not carry producer-stream semantics. Callers are
        # responsible for any synchronization before exposing the pointer.
        _ = stream
        dl_managed_tensor_ptr = self.managed_tensor_ptr
        _DLPACK_STATE[dl_managed_tensor_ptr] = self
        return _PyCapsule_New(
            ctypes.c_void_p(dl_managed_tensor_ptr),
            _DLPACK_CAPSULE_NAME,
            None,
        )

    @property
    def managed_tensor_ptr(self) -> int:
        return ctypes.addressof(self._managed_tensor)


def uint8_cuda_tensor_from_ptr(data_ptr: int, numel: int, device_index: int) -> torch.Tensor:
    numel = int(numel)
    if numel < 0:
        raise ValueError(f"numel must be >= 0, got {numel}")
    if numel == 0:
        return torch.empty((0, ), dtype=torch.uint8, device=f"cuda:{device_index}")
    view = _DLPackCudaPtrView(int(data_ptr), numel, int(device_index))
    try:
        return torch.from_dlpack(view)
    except Exception:
        _DLPACK_STATE.pop(view.managed_tensor_ptr, None)
        raise
