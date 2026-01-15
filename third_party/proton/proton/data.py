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


def is_phase_flushed(session: Optional[int] = 0, phase: int = 0) -> bool:
    """
    Checks if the profiling data for a given session and phase has been flushed.

    Args:
        session (Optional[int]): The session ID of the profiling session, or None if profiling is inactive.
        phase (int): The phase number to check. Defaults to 0.

    Returns:
        bool: True if the phase data has been flushed, False otherwise.
    """
    if session is None:
        return False
    if flags.command_line and session != 0:
        raise ValueError("Only one session can check phase flush status when running from the command line.")
    return libproton.is_data_phase_flushed(session, phase)


def clear(session: Optional[int] = 0, phase: int = 0) -> None:
    """
    Clears profiling data for a given session.

    Args:
        session (Optional[int]): The session ID of the profiling session, or None if profiling is inactive.
        phase (int): The phase number to clear. Defaults to 0.
    """
    if session is None:
        return
    if flags.command_line and session != 0:
        raise ValueError("Only one session can be cleared when running from the command line.")
    libproton.clear_data(session, phase)


def pop_flushed(session: Optional[int] = 0):
    """
    Pops one flushed JSON payload from the in-memory periodic flushing buffer.

    Args:
        session (Optional[int]): The session ID of the profiling session, or None if profiling is inactive.

    Returns:
        Optional[Tuple[int, object]]: (phase, parsed_json) or None if empty.
    """
    if session is None:
        return None
    if flags.command_line and session != 0:
        raise ValueError("Only one session can be drained when running from the command line.")
    result = libproton.pop_flushed_data(session)
    if result is None:
        return None
    phase, payload = result
    return phase, json.loads(payload)


def pop_flushed_msgpack(session: Optional[int] = 0):
    """
    Pops one flushed MessagePack payload from the in-memory periodic flushing buffer.

    Args:
        session (Optional[int]): The session ID of the profiling session, or None if profiling is inactive.

    Returns:
        Optional[Tuple[int, bytes]]: (phase, msgpack_bytes) or None if empty.
    """
    if session is None:
        return None
    if flags.command_line and session != 0:
        raise ValueError("Only one session can be drained when running from the command line.")
    return libproton.pop_flushed_data_msgpack(session)


def pop_flushed_path_metrics(session: Optional[int] = 0):
    """
    Pops one flushed per-path metrics payload from the in-memory periodic flushing buffer.

    Returns:
        Optional[Tuple[int, List[Tuple[str, Optional[float], Optional[float]]]]]
    """
    if session is None:
        return None
    if flags.command_line and session != 0:
        raise ValueError("Only one session can be drained when running from the command line.")
    return libproton.pop_flushed_path_metrics(session)


def get_path_metrics(session: Optional[int] = 0, phase: int = 0):
    """
    Retrieves per-path metrics for a given session/phase.

    Returns:
        List[Tuple[str, Optional[float], Optional[float]]]
    """
    if session is None:
        return []
    if flags.command_line and session != 0:
        raise ValueError("Only one session can be retrieved when running from the command line.")
    return libproton.get_path_metrics(session, phase)
