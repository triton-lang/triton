from triton._C.libproton import proton as libproton
from .flags import flags
from functools import wraps
from contextvars import ContextVar

COMPUTE_METADATA_SCOPE_NAME = "__proton_launch_metadata"
COMPUTE_METADATA_SCOPE_PREFIX = f"{COMPUTE_METADATA_SCOPE_NAME}:"
_state_stack = ContextVar("proton_state_stack", default=())
_metadata_scope_stack = ContextVar("proton_metadata_scope_stack", default=())


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
    stack = _state_stack.get()
    _state_stack.set((*stack, name))
    libproton.enter_state(name)


def exit_state() -> None:
    stack = _state_stack.get()
    if not stack:
        libproton.exit_state()
        return

    stack = stack[:-1]
    _state_stack.set(stack)
    if stack:
        libproton.enter_state(stack[-1])
    else:
        libproton.exit_state()


def is_state_active(name: str) -> bool:
    return name in _state_stack.get()


def metadata_state_name(kernel_name=None) -> str:
    if not kernel_name:
        return COMPUTE_METADATA_SCOPE_NAME
    return f"{COMPUTE_METADATA_SCOPE_PREFIX}{kernel_name}"


def enter_metadata_scope(name: str) -> int:
    stack = _metadata_scope_stack.get()
    _metadata_scope_stack.set((*stack, name))
    scope_id = libproton.record_scope()
    libproton.enter_scope(scope_id, name)
    return scope_id


def exit_metadata_scope(scope_id: int, name: str) -> None:
    try:
        libproton.exit_scope(scope_id, name)
    finally:
        stack = _metadata_scope_stack.get()
        if stack:
            _metadata_scope_stack.set(stack[:-1])


def is_metadata_scope_active() -> bool:
    return bool(_metadata_scope_stack.get())


def is_metadata_state_active() -> bool:
    if is_metadata_scope_active():
        return True
    return any(
        name == COMPUTE_METADATA_SCOPE_NAME
        or name.startswith(COMPUTE_METADATA_SCOPE_PREFIX)
        for name in _state_stack.get()
    )


def current_metadata_state_name() -> str:
    for name in reversed(_state_stack.get()):
        if (
            name == COMPUTE_METADATA_SCOPE_NAME
            or name.startswith(COMPUTE_METADATA_SCOPE_PREFIX)
        ):
            return name
    return COMPUTE_METADATA_SCOPE_NAME
