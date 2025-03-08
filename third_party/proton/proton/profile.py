import functools
import triton
import os
import pathlib

from triton import knobs
from triton._C.libproton import proton as libproton
from .hook import register_launch_hook, unregister_launch_hook, register_instrumentation_hook, unregister_instrumentation_hook
from .flags import set_profiling_off, set_profiling_on, is_command_line
from typing import Optional

DEFAULT_PROFILE_NAME = "proton"


def _select_backend() -> str:
    backend = triton.runtime.driver.active.get_current_target().backend
    if backend == "cuda":
        return "cupti"
    elif backend == "hip":
        return "roctracer"
    else:
        raise ValueError("No backend is available for the current target.")


def _get_backend_default_path(backend: str) -> str:
    lib_path = ""
    if backend == "cupti":
        # First try to get the path from the environment variable that overrides the default path
        lib_path = knobs.proton.cupti_dir
        if lib_path is None:
            # Get the default path for the cupti backend,
            # which is the most compatible with the current CUPTI header file triton is compiled with
            lib_path = str(pathlib.Path(__file__).parent.parent.absolute() / "backends" / "nvidia" / "lib" / "cupti")
    return lib_path


def _check_env(backend: str) -> None:
    if backend == "roctracer":
        hip_device_envs = ["HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"]
        for env in hip_device_envs:
            if os.getenv(env, None) is not None:
                raise ValueError(
                    f"Proton does not work when the environment variable {env} is set on AMD GPUs. Please unset it and use `ROCR_VISIBLE_DEVICES` instead"
                )


def _check_mode(backend: str, mode: Optional[str]) -> None:
    # TODO(Keren): Need a better mode registration mechanism
    backend_modes = {
        "cupti": ["", "pcsampling"],
        "roctracer": [""],
        "instrumentation": [""],
    }

    if mode not in backend_modes[backend]:
        raise ValueError(f"Invalid mode {mode} for backend {backend}")


def start(
    name: Optional[str] = None,
    *,
    context: Optional[str] = "shadow",
    data: Optional[str] = "tree",
    backend: Optional[str] = None,
    mode: Optional[str] = None,
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
        name (str, optional): The name (with path) of the profiling session.
                              If not provided, the default name is "~/proton.hatchet".
        context (str, optional): The context to use for profiling.
                                 Available options are ["shadow", "python"].
                                 Defaults to "shadow".
        data (str, optional): The data structure to use for profiling.
                              Available options are ["tree"].
                              Defaults to "tree".
        backend (str, optional): The backend to use for profiling.
                                 Available options are [None, "cupti", "roctracer", "instrumentation"].
                                 Defaults to None, which automatically selects the backend matching the current active runtime.
        mode (str, optional): The "mode" to use for profiling, which is specific to the backend.
                              Defaults to None.
                              For "cupti", available options are [None, "pcsampling"].
                              For "roctracer", available options are [None].
                              For "instrumentation", available options are [None].
        hook (str, optional): The hook to use for profiling.
                              Available options are [None, "triton"].
                              Defaults to None.
    Returns:
        session (int): The session ID of the profiling session.
    """
    if is_command_line():
        # Ignore the start() call if the script is run from the command line.
        return

    set_profiling_on()

    name = DEFAULT_PROFILE_NAME if name is None else name
    backend = _select_backend() if backend is None else backend
    backend_path = _get_backend_default_path(backend)
    mode = "" if mode is None else mode

    _check_env(backend)
    _check_mode(backend, mode)

    if hook == "triton":
        register_launch_hook()
    if backend == "instrumentation":
        register_instrumentation_hook()

    return libproton.start(name, context, data, backend, mode, backend_path)


def activate(session: Optional[int] = None) -> None:
    """
    Activate the specified session.
    The profiling session will be active and data will be recorded.

    Args:
        session (int): The session ID of the profiling session. Defaults to None (all sessions)

    Returns:
        None
    """
    if is_command_line() and session != 0:
        raise ValueError("Only one session can be activated when running from the command line.")
    if session is None:
        libproton.activate_all()
    else:
        libproton.activate(session)


def deactivate(session: Optional[int] = None) -> None:
    """
    Stop the specified session.
    The profiling session's data will still be in the memory, but no more data will be recorded.

    Args:
        session (int): The session ID of the profiling session. Defaults to None (all sessions)

    Returns:
        None
    """
    if is_command_line() and session != 0:
        raise ValueError("Only one session can be deactivated when running from the command line.")
    if session is None:
        libproton.deactivate_all()
    else:
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
        unregister_launch_hook()
        unregister_instrumentation_hook()
    else:
        if is_command_line() and session != 0:
            raise ValueError("Only one session can be finalized when running from the command line.")
        libproton.finalize(session, output_format)


def _profiling(
    func,
    name: Optional[str] = None,
    context: Optional[str] = "shadow",
    data: Optional[str] = "tree",
    backend: Optional[str] = None,
    mode: Optional[str] = None,
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
        session = start(name, context=context, data=data, backend=backend, mode=mode, hook=hook)
        ret = func(*args, **kwargs)
        deactivate(session)
        return ret

    return wrapper


def profile(
    func=None,
    *,
    name: Optional[str] = None,
    context: Optional[str] = "shadow",
    data: Optional[str] = "tree",
    backend: Optional[str] = None,
    mode: Optional[str] = None,
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
            return _profiling(f, name=name, context=context, data=data, backend=backend, mode=mode, hook=hook)

        return decorator
    else:
        # It's being used without parentheses, so apply the decorator directly
        return _profiling(func, name=name, context=context, data=data, backend=backend, mode=mode, hook=hook)
