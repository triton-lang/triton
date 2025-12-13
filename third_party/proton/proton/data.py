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


def get_data_msgpack(session: int, decode: bool = False):
    """
    Retrieves profiling data for a given session encoded with MessagePack.

    Args:
        session (int): The session ID of the profiling session.
        decode (bool, optional): If True, decode using msgpack.loads.

    Returns:
        bytes | dict: Raw MessagePack bytes when decode is False, otherwise the
        decoded object.
    """
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
