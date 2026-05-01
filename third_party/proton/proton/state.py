from functools import wraps
import threading

from triton._C.libproton import proton as libproton
from .flags import flags

COMPUTE_METADATA_SCOPE_NAME = "__proton_launch_metadata"
COMPUTE_METADATA_SCOPE_PREFIX = f"{COMPUTE_METADATA_SCOPE_NAME}:"
_thread_state = threading.local()


class state:
    """
    A context manager and decorator for entering and exiting a state.

    Usage:
        context manager:
        ```python
        with proton.state("test0"):
            foo[1,](x, y)
        ```

        decorator:
        ```python
        @proton.state("test0")
        def foo(x, y):
            ...
        ```

    Args:
        name (str): The name of the state.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def __enter__(self):
        if not flags.profiling_on:
            return self
        enter_state(self.name)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if not flags.profiling_on:
            return
        exit_state()

    def __call__(self, func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            if flags.profiling_on:
                enter_state(self.name)
            try:
                return func(*args, **kwargs)
            finally:
                if flags.profiling_on:
                    exit_state()

        return wrapper


class metadata_state(state):

    def __init__(self) -> None:
        super().__init__(COMPUTE_METADATA_SCOPE_NAME)


def enter_state(name: str) -> None:
    libproton.enter_state(name)
    _thread_state.name = name


def exit_state() -> None:
    libproton.exit_state()
    _thread_state.name = None


def _current_state() -> str | None:
    return getattr(_thread_state, "name", None)


def metadata_state_name(kernel_name=None) -> str:
    if not kernel_name:
        return COMPUTE_METADATA_SCOPE_NAME
    return f"{COMPUTE_METADATA_SCOPE_PREFIX}{kernel_name}"


def enter_metadata_scope(name: str) -> int:
    scope_id = libproton.record_scope()
    libproton.enter_scope(scope_id, name)
    try:
        enter_state(name)
    except Exception:
        libproton.exit_scope(scope_id, name)
        raise
    return scope_id


def exit_metadata_scope(scope_id: int, name: str) -> None:
    try:
        exit_state()
    finally:
        libproton.exit_scope(scope_id, name)


def is_metadata_state_active() -> bool:
    state_name = _current_state()
    return bool(state_name
                and (state_name == COMPUTE_METADATA_SCOPE_NAME or state_name.startswith(COMPUTE_METADATA_SCOPE_PREFIX)))
