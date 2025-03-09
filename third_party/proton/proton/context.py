from typing import Optional
from triton._C.libproton import proton as libproton
from .flags import get_profiling_on


def depth(session: Optional[int] = 0):
    if not get_profiling_on():
        return 0
    return libproton.get_context_depth(session)
