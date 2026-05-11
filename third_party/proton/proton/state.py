from functools import wraps

from triton._C.libproton import proton as libproton
from .flags import flags

COMPUTE_METADATA_SCOPE_NAME = libproton.metadata_scope_name
COMPUTE_METADATA_SCOPE_PREFIX = libproton.metadata_scope_prefix


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

    def __init__(self, kernel_name=None) -> None:
        super().__init__(get_metadata_state_name(kernel_name))


def enter_state(name: str) -> None:
    libproton.enter_state(name)


def exit_state() -> None:
    libproton.exit_state()


def get_metadata_state_name(kernel_name=None) -> str:
    if not kernel_name:
        return COMPUTE_METADATA_SCOPE_NAME
    return f"{COMPUTE_METADATA_SCOPE_PREFIX}{kernel_name}"
