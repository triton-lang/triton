from functools import wraps
import threading
from typing import Optional

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
        libproton.enter_state(self.name)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if not flags.profiling_on:
            return
        libproton.exit_state()

    def __call__(self, func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            if flags.profiling_on:
                libproton.enter_state(self.name)
            ret = func(*args, **kwargs)
            if flags.profiling_on:
                libproton.exit_state()
            return ret

        return wrapper


class metadata_state(state):

    def __init__(self) -> None:
        super().__init__(COMPUTE_METADATA_SCOPE_NAME)


def enter_state(name: str) -> None:
    libproton.enter_state(name)
    stack = getattr(_thread_state, "state_stack", None)
    if stack is None:
        stack = []
        _thread_state.state_stack = stack
    stack.append(name)


def exit_state() -> None:
    stack = getattr(_thread_state, "state_stack", None)
    if stack:
        stack.pop()
    if stack:
        libproton.enter_state(stack[-1])
    else:
        libproton.exit_state()


def get_state(session: Optional[int] = 0) -> Optional[str]:
    """
    Get the current state.

    Args:
        session (int): The session ID of the profiling session. Defaults to 0.

    Returns:
        state (str or None): The current state. If profiling is off or no state is active, returns None.
    """
    if not flags.profiling_on:
        return None
    return libproton.get_state(session)


def metadata_state_name(kernel_name=None) -> str:
    if not kernel_name:
        return COMPUTE_METADATA_SCOPE_NAME
    return f"{COMPUTE_METADATA_SCOPE_PREFIX}{kernel_name}"


def is_metadata_state_active() -> bool:
    stack = getattr(_thread_state, "state_stack", None)
    if not stack:
        return False
    state_name = stack[-1]
    return state_name == COMPUTE_METADATA_SCOPE_NAME or state_name.startswith(COMPUTE_METADATA_SCOPE_PREFIX)
