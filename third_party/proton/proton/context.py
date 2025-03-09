from typing import Optional
from triton._C.libproton import proton as libproton
from .flags import get_profiling_on


def depth(session: Optional[int] = 0):
    """
    Get the depth of the context.

    Args:
        session (int): The session ID of the profiling session. Defaults to 0.

    Returns:
        int: The depth of the context of the session.
    """
    if not get_profiling_on():
        return 0
    return libproton.get_context_depth(session)
