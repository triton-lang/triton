from typing import Optional
from triton._C.libproton import proton as libproton
from .flags import flags


def depth(session: Optional[int] = 0) -> Optional[int]:
    """
    Get the depth of the context.

    Args:
        session (int): The session ID of the profiling session. Defaults to 0.

    Returns:
        depth (int or None): The depth of the context. If profiling is off, returns None.
    """
    if not flags.profiling_on:
        return None
    return libproton.get_context_depth(session)
