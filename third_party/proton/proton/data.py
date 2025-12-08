from triton._C.libproton import proton as libproton  # type: ignore

def get_data(session: int) -> str:
    """
    Retrieves profiling data for a given session.

    Args:
        session (int): The session ID of the profiling session.

    Returns:
        str: The profiling data in JSON string format.
    """
    return libproton.get_data(session)
 