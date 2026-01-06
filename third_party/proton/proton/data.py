from typing import Optional
from triton._C.libproton import proton as libproton  # type: ignore
import json as json
from .flags import flags


def get(session: int = 0):
    """
    Retrieves profiling data for a given session.

    Args:
        session (int): The session ID of the profiling session.
    Returns:
        str: The profiling data in JSON format.
    """
    if flags.command_line and session != 0:
        raise ValueError("Only one session can be retrieved when running from the command line.")
    return json.loads(libproton.get_data(session))


def get_msgpack(session: int = 0):
    """
    Retrieves profiling data for a given session encoded with MessagePack.

    Args:
        session (int): The session ID of the profiling session.

    Returns:
        bytes: The profiling data encoded with MessagePack.
    """
    if flags.command_line and session != 0:
        raise ValueError("Only one session can be retrieved when running from the command line.")
    return libproton.get_data_msgpack(session)


def advance_phase(session: int = 0) -> int:
    """
    Advances the profiling phase for a given session.

    Args:
        session (int): The session ID of the profiling session.
    
    Returns:
        int: The next phase number after advancing.
    """
    if flags.command_line and session != 0:
        raise ValueError("Only one session can advance phase when running from the command line.")
    return libproton.advance_phase(session)


def clear(session: int = 0, phase : int = 0):
    """
    Clears profiling data for a given session.

    Args:
        session (int): The session ID of the profiling session.
        phase (int): The phase number to clear. Defaults to 0.
    """
    if flags.command_line and session != 0:
        raise ValueError("Only one session can be cleared when running from the command line.")
    libproton.clear_data(session, phase)
