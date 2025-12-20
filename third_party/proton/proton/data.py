from triton._C.libproton import proton as libproton  # type: ignore
import json as json


def get_data(session: int):
    """
    Retrieves profiling data for a given session.

    Args:
        session (int): The session ID of the profiling session.

    Returns:
        str: The profiling data in JSON format.
    """
    return json.loads(libproton.get_data(session))


def get_data_msgpack(session: int):
    """
    Retrieves profiling data for a given session encoded with MessagePack.

    Args:
        session (int): The session ID of the profiling session.

    Returns:
        bytes: The profiling data encoded with MessagePack.
    """
    return libproton.get_data_msgpack(session)


def clear_data(session: int):
    """
    Clears profiling data for a given session.

    Args:
        session (int): The session ID of the profiling session.
    """
    libproton.clear_data(session)
