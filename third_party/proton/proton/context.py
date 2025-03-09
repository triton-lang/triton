from typing import Optional
from triton._C.libproton import proton as libproton


def depth(session: Optional[int] = 0):
    return libproton.get_context_depth(session)
