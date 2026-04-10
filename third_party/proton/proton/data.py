from typing import Optional
from triton._C.libproton import proton as libproton  # type: ignore
import json as json
from .flags import flags


def get(session: Optional[int] = 0, phase: int = 0):
    """
    Retrieves profiling data for a given session.

    Args:
        session (Optional[int]): The session ID of the profiling session, or None if profiling is inactive.
    Returns:
        str: The profiling data in JSON format.
    """
    if session is None:
        return None
    if flags.command_line and session != 0:
        raise ValueError("Only one session can be retrieved when running from the command line.")
    return json.loads(libproton.get_data(session, phase))


def get_msgpack(session: Optional[int] = 0, phase: int = 0):
    """
    Retrieves profiling data for a given session encoded with MessagePack.

    Args:
        session (Optional[int]): The session ID of the profiling session, or None if profiling is inactive.

    Returns:
        bytes: The profiling data encoded with MessagePack.
    """
    if session is None:
        return None
    if flags.command_line and session != 0:
        raise ValueError("Only one session can be retrieved when running from the command line.")
    return libproton.get_data_msgpack(session, phase)


def get_buffered_profiles(
    session: Optional[int] = 0,
    clear: bool = False,
    max_profiles: int = 0,
    min_phase: int = 0,
):
    """
    Retrieves phase-tagged serialized profiles buffered in Proton's native
    in-memory ring buffer.

    Args:
        session (Optional[int]): The session ID of the profiling session, or None if profiling is inactive.
        clear (bool): Whether to clear buffered profiles after retrieval.
        max_profiles (int): Maximum number of profiles to return. A value of 0 means unlimited.
        min_phase (int): Minimum phase to return. Older profiles are removed when ``clear`` is true.

    Returns:
        list[tuple[int, bytes]]: A list of ``(phase, payload)`` tuples in FIFO order.
    """
    if session is None:
        return []
    if flags.command_line and session != 0:
        raise ValueError("Only one session can retrieve buffered profiles when running from the command line.")
    return list(libproton.get_buffered_profiles(session, clear, max_profiles, min_phase))


def get_buffered_profile_serialized_up_to_phase(session: Optional[int] = 0):
    """
    Returns the newest completed phase serialized into Proton's native buffer.
    """
    if session is None:
        return None
    if flags.command_line and session != 0:
        raise ValueError("Only one session can retrieve buffered profiles when running from the command line.")
    return libproton.get_buffered_profile_serialized_up_to_phase(session)


def get_buffered_profile_stats(session: Optional[int] = 0):
    """
    Returns ``(profile_count, byte_count)`` for Proton's native profile buffer.
    """
    if session is None:
        return (0, 0)
    if flags.command_line and session != 0:
        raise ValueError("Only one session can retrieve buffered profile stats when running from the command line.")
    return libproton.get_buffered_profile_stats(session)


def advance_phase(session: Optional[int] = 0) -> Optional[int]:
    """
    Advances the profiling phase for a given session.

    Args:
        session (Optional[int]): The session ID of the profiling session, or None if profiling is inactive.

    Returns:
        Optional[int]: The next phase number after advancing.
    """
    if session is None:
        return None
    if flags.command_line and session != 0:
        raise ValueError("Only one session can advance phase when running from the command line.")
    return libproton.advance_data_phase(session)


def is_phase_complete(session: Optional[int] = 0, phase: int = 0) -> bool:
    """
    Checks if the profiling data for a given session and phase is complete.

    A "complete" phase is safe to read/clear because all device-side records for
    the phase have been flushed to the host and the phase will no longer receive
    new records.

    Args:
        session (Optional[int]): The session ID of the profiling session, or None if profiling is inactive.
        phase (int): The phase number to check. Defaults to 0.

    Returns:
        bool: True if the phase data is complete, False otherwise.
    """
    if session is None:
        return False
    if flags.command_line and session != 0:
        raise ValueError("Only one session can check phase completion status when running from the command line.")
    return libproton.is_data_phase_complete(session, phase)


def clear(
    session: Optional[int] = 0,
    phase: int = 0,
    clear_up_to_phase: bool = False,
) -> None:
    """
    Clears profiling data for a given session.

    Args:
        session (Optional[int]): The session ID of the profiling session, or None if profiling is inactive.
        phase (int): The phase number to clear. Defaults to 0.
        clear_up_to_phase (bool): If True, clear all phases up to and including `phase`.
    """
    if session is None:
        return
    if flags.command_line and session != 0:
        raise ValueError("Only one session can be cleared when running from the command line.")
    libproton.clear_data(session, phase, clear_up_to_phase)
