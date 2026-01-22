import functools
import triton

from triton._C.libproton import proton as libproton  # type: ignore
from triton._C.libtriton import getenv  # type: ignore
from .flags import flags
from .hooks import HookManager, LaunchHook, InstrumentationHook
from .hooks.hook import Hook
from .mode import BaseMode
from typing import Optional, Union

DEFAULT_PROFILE_NAME = "proton"


def _select_backend() -> str:
    backend = triton.runtime.driver.active.get_current_target().backend
    if backend == "cuda":
        return "cupti"
    elif backend == "hip":
        return "roctracer"
    else:
        raise ValueError("No backend is available for the current target.")


def _get_mode_str(backend: str, mode: Optional[Union[str, BaseMode]]) -> str:
    if backend == "instrumentation":
        prefix = triton.runtime.driver.active.get_current_target().backend
        return f"{prefix}:{mode}" if mode else prefix
    return str(mode) if mode else ""


def _check_env(backend: str) -> None:
    if backend == "roctracer":
        hip_device_envs = ["HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"]
        for env in hip_device_envs:
            if getenv(env, None) is not None:
                raise ValueError(
                    f"Proton does not work when the environment variable {env} is set on AMD GPUs. Please unset it and use `ROCR_VISIBLE_DEVICES` instead"
                )

    # Ensure default envs are set for Proton knobs if not already set by the user.
    for attr, desc in triton.knobs.proton.knob_descriptors.items():
        key = desc.key
        if getenv(key, None) is None:
            val = getattr(triton.knobs.proton, attr)
            if val is not None:
                if env_val := triton.knobs.toenv(val):
                    triton.knobs.setenv(key, env_val[0])


def start(
    name: Optional[str] = None,
    *,
    context: Optional[str] = "shadow",
    data: Optional[str] = "tree",
    backend: Optional[str] = None,
    mode: Optional[Union[str, BaseMode]] = None,
    hook: Optional[Union[str, Hook]] = None,
) -> Optional[int]:
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
                              If not provided, the default name is "~/proton.<suffix>", where suffix is the default
                              format according to the data type. For example, if data is "tree", the default name is "~/proton.hatchet".
        context (str, optional): The context to use for profiling.
                                 Available options are ["shadow", "python"].
                                 Defaults to "shadow".
        data (str, optional): The data structure to use for profiling.
                              Available options are ["tree", "trace"].
                              Defaults to "tree".
        backend (str, optional): The backend to use for profiling.
                                 Available options are [None, "cupti", "roctracer", "instrumentation"].
                                 Defaults to None, which automatically selects the backend matching the current active runtime.
        mode (Union[str, BaseMode], optional): The "mode" to use for profiling, which is specific to the backend.
                                               Can be a string or an instance of BaseMode (or any subclass thereof).
                                               Defaults to None.
                                               For "cupti", available options are [None, "pcsampling", "periodic_flushing"].
                                               For "roctracer", available options are ["periodic_flushing"].
                                               For "instrumentation", available options are [None].
                                               Each mode has a set of control knobs following with the mode name.
                                               For example, "periodic_flushing" mode has a knob:
                                               - format: The output format of the profiling results. Available options are ["hatchet", "hatchet_msgpack", "chrome_trace"]. Default is "hatchet".
                                               The can be set via `mode="periodic_flushing:format=chrome_trace"`.
        hook (Union[str, Hook], optional): The hook to use for profiling.
                                           You may pass either:
                                           - a string hook name, e.g. "triton" (kernel launch metadata), or
                                           - a custom Hook instance.
                                           Defaults to None.
    Returns:
        session (Optional[int]): The session ID of the profiling session, or None if profiling is disabled.
    """
    if flags.command_line or triton.knobs.proton.disable:
        # Ignore the start() call if the script is run from the command line or profiling is disabled.
        return None

    flags.profiling_on = True

    name = DEFAULT_PROFILE_NAME if name is None else name
    backend = _select_backend() if backend is None else backend
    # Convert mode to its string representation for libproton's runtime
    mode_str = _get_mode_str(backend, mode)

    _check_env(backend)

    session = libproton.start(name, context, data, backend, mode_str)

    if isinstance(hook, Hook):
        HookManager.register(hook, session)
    elif hook == "triton":
        HookManager.register(LaunchHook(), session)
    elif hook is not None:
        raise ValueError(f"Unsupported hook: {hook!r}")
    if backend == "instrumentation":
        HookManager.register(InstrumentationHook(mode), session)

    return session


def activate(session: Optional[int] = None) -> None:
    """
    Activate the specified session.
    The profiling session will be active and data will be recorded.

    Args:
        session (int): The session ID of the profiling session. Defaults to None (all sessions)

    Returns:
        None
    """
    if flags.command_line and session != 0:
        raise ValueError("Only one session can be activated when running from the command line.")

    HookManager.activate(session)

    if session is None:
        libproton.activate_all()
    else:
        libproton.activate(session)


def deactivate(session: Optional[int] = None, flushing: bool = False) -> None:
    """
    Stop the specified session.
    The profiling session's data will still be in the memory, but no more data will be recorded.

    Args:
        session (int): The session ID of the profiling session. Defaults to None (all sessions)
        flushing (bool): Whether to flush the profiling data before deactivating. Defaults to True.

    Returns:
        None
    """
    if flags.command_line and session != 0:
        raise ValueError("Only one session can be deactivated when running from the command line.")

    HookManager.deactivate(session)

    if session is None:
        libproton.deactivate_all(flushing)
    else:
        libproton.deactivate(session, flushing)


def finalize(session: Optional[int] = None, output_format: Optional[str] = "") -> None:
    """
    Finalizes a profiling session.
    Flush and write the profiling data to the file specified by the session name.

    Args:
        session (int, optional): The session ID to finalize. If None, all sessions are finalized. Defaults to None.
        output_format (str, optional): The output format for the profiling results.
                                       Available options are ["hatchet", "hatchet_msgpack", "chrome_trace"].

    Returns:
        None
    """
    HookManager.unregister(session)

    if session is None:
        flags.profiling_on = False
        libproton.finalize_all(output_format)
    else:
        if flags.command_line and session != 0:
            raise ValueError("Only one session can be finalized when running from the command line.")
        libproton.finalize(session, output_format)


def _profiling(
    func,
    name: Optional[str] = None,
    context: Optional[str] = "shadow",
    data: Optional[str] = "tree",
    backend: Optional[str] = None,
    mode: Optional[str] = None,
    hook: Optional[Union[str, Hook]] = None,
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
    hook: Optional[Union[str, Hook]] = None,
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
