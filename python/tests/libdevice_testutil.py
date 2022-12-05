import os
from typing import Optional

_SYSTEM_LIBDEVICE_SEARCH_PATHS = [
    '/usr/lib/cuda/nvvm/libdevice/libdevice.10.bc',
    '/usr/local/cuda/nvvm/libdevice/libdevice.10.bc',
]

SYSTEM_LIBDEVICE_PATH: Optional[str] = None
for _p in _SYSTEM_LIBDEVICE_SEARCH_PATHS:
    if os.path.exists(_p):
        SYSTEM_LIBDEVICE_PATH = _p

def system_libdevice_path() -> str:
    assert SYSTEM_LIBDEVICE_PATH is not None, \
        "Could not find libdevice.10.bc path"
    return SYSTEM_LIBDEVICE_PATH

