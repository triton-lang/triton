import functools

from triton._C.libproton import proton as libproton
from .hook import register_triton_hook, unregister_triton_hook
from .flags import set_profiling_off, set_profiling_on
from typing import Optional

DEFAULT_PROFILE_NAME = "proton"


def start(
    name: Optional[str] = None,
    *,
    backend: str = "cupti",
    context: str = "shadow",
    data: str = "tree",
    hook: Optional[str] = None,
):
    """
    Start profiling with the given name and backend.

    Usage:

        ```python
        proton.start("my_profile")
        # do something
        proton.finalize()
        ```

    Args:
        name (str): The name (with path) of the profiling session.
                    If not provided, the default name is "~/proton".
        backend (str): The backend to use for profiling.
                       Available options are ["cupti"].
                       Defaults to "cupti".
        context (str): The context to use for profiling.
                       Available options are ["shadow", "python"].
                       Defaults to "shadow".
        data (str): The data structure to use for profiling.
                    Available options are ["tree"].
                    Defaults to "tree".
        hook (str, optional): The hook to use for profiling.
                              Available options are [None, "triton"].
                              Defaults to None.
    Returns:
        session (int): The session ID of the profiling session.
    """
    if name is None:
        name = DEFAULT_PROFILE_NAME

    set_profiling_on()
    if hook and hook == "triton":
        register_triton_hook()
    return libproton.start(name, backend, context, data)


def activate(session: Optional[int] = 0) -> None:
    """
    Activate the specified session.
    The profiling session will be active and data will be recorded.

    Args:
        session (int): The session ID of the profiling session. Defaults to 0 (the first session started.)

    Returns:
        None
    """
    libproton.activate(session)


def deactivate(session: Optional[int] = 0) -> None:
    """
    Stop the specified session.
    The profiling session's data will still be in the memory, but no more data will be recorded.

    Args:
        session (int): The session ID of the profiling session. Defaults to 0 (the first session started.)

    Returns:
        None
    """
    libproton.deactivate(session)


def finalize(session: Optional[int] = None, output_format: str = "hatchet") -> None:
    """
    Finalizes a profiling session.
    Flush and write the profiling data to the file specified by the session name.

    Args:
        session (int, optional): The session ID to finalize. If None, all sessions are finalized. Defaults to None.
        output_format (str, optional): The output format for the profiling results.
                                       Aavailable options are ["hatchet"].

    Returns:
        None
    """
    if session is None:
        set_profiling_off()
        libproton.finalize_all(output_format)
        unregister_triton_hook()
    else:
        libproton.finalize(session, output_format)


def _profiling(
    func,
    name: Optional[str] = None,
    backend: str = "cupti",
    context: str = "shadow",
    data: str = "tree",
    hook: Optional[str] = None,
):
    """
    Context manager for profiling. Internally use only.

    Args:
        See start() for the arguments.

    Returns:
        wrapper (function): The wrapped function.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        session = start(name, backend=backend, context=context, data=data, hook=hook)
        ret = func(*args, **kwargs)
        deactivate(session)
        return ret

    return wrapper


def profile(
    func=None,
    *,
    name: Optional[str] = None,
    backend: str = "cupti",
    context: str = "shadow",
    data: str = "tree",
    hook: Optional[str] = None,
):
    """
    Decorator for profiling.

    Usage:

    ```python
    @proton.profile
    def foo():
        pass
    ```

    Args:
        See start() for the arguments.

    Returns:
        decorator (function): The decorator function.
    """
    if func is None:
        # It's being used with parentheses, so return a decorator
        def decorator(f):
            return _profiling(f, name=name, backend=backend, context=context, data=data)

        return decorator
    else:
        # It's being used without parentheses, so apply the decorator directly
        return _profiling(func, name=name, backend=backend, context=context, data=data)
