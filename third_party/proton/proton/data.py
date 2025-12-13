from triton._C.libproton import proton as libproton  # type: ignore


def get_data(session: int):
    """
    Retrieves profiling data for a given session.

    Args:
        session (int): The session ID of the profiling session.

    Returns:
        dict: The profiling data.
    """
    return libproton.get_data(session)


def get_data_msgpack(session: int, decode: bool = False, zero_copy: bool = False):
    """
    Retrieves profiling data for a given session encoded with MessagePack.

    Args:
        session (int): The session ID of the profiling session.
        decode (bool, optional): If True, decode using msgpack.loads.
        zero_copy (bool, optional): If True, return a memoryview backed by a
        C++ buffer (avoids an extra copy into Python bytes).

    Returns:
        bytes | memoryview | dict: Raw MessagePack bytes (or a memoryview when
        zero_copy is True) when decode is False, otherwise the decoded object.
    """
    if zero_copy:
        payload = memoryview(libproton.get_data_msgpack_buffer(session))
    else:
        payload = libproton.get_data_msgpack(session)
    if not decode:
        return payload
    import msgpack  # type: ignore
    return msgpack.loads(payload)


def clear_data(session: int):
    """
    Clears profiling data for a given session.

    Args:
        session (int): The session ID of the profiling session.
    """
    libproton.clear_data(session)
