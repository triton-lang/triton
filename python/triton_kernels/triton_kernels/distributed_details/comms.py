from typing import Callable

from .shmem import Shmem

_COMMS: dict[str, Callable[[], Shmem]] = {}

try:
    from .nvshmem import init_comms as _nv_init_comms

    _COMMS["nccl"] = _nv_init_comms
except Exception:
    pass

_SHMEM_BACKEND: str | None = None
_SHMEM: Shmem | None = None


def init_comms(backend: str) -> Shmem:
    global _SHMEM_BACKEND
    global _SHMEM

    if _SHMEM_BACKEND is not None:
        if _SHMEM_BACKEND != backend:
            raise ValueError(f"init_comms() previously called with different backend {_SHMEM_BACKEND}, now {backend}")
        assert _SHMEM is not None
        return _SHMEM

    init_fn = _COMMS.get(backend)
    if init_fn is None:
        raise ValueError(f"Comms backend {backend} not registered")
    _SHMEM = init_fn()
    _SHMEM_BACKEND = backend
    return _SHMEM


def shmem() -> Shmem:
    if _SHMEM is None:
        raise ValueError("init_comms() has not been called")
    return _SHMEM
