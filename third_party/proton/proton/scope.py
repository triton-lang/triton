import threading
from functools import wraps
from typing import Optional, Union

from .flags import get_profiling_on
from triton._C.libproton import proton as libproton

thread_local_scopes = threading.local()

MetricValueType = Union[float, int]
PropertyValueType = Union[float, int, str]


class scope:
    """
    A context manager and decorator for entering and exiting a scope.

    Usage:
        context manager:
        ```python
        with proton.scope("test0", {metric_name: metric_value}):
            foo[1,](x, y)
        ```

        decorator:
        ```python
        @proton.scope("test0", {metric_name: metric_value})
        def foo(x, y):
            ...
        ```

    Args:
        name (str): The name of the scope.
        metrics (dict[str, float], optional): The metrics of the scope. Default is None.
    """

    def __init__(self, name: str, metrics: Optional[dict[str, MetricValueType]] = None,
                 properties: Optional[dict[str, PropertyValueType]] = None) -> None:
        self.name = name
        self.metrics = metrics
        self.properties = properties

    def __enter__(self):
        if not get_profiling_on():
            return self
        self.id = libproton.record_scope()
        libproton.enter_scope(self.id, self.name)
        if self.metrics:
            libproton.add_metrics(self.id, self.metrics)
        if self.properties:
            libproton.set_properties(self.id, self.properties)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if not get_profiling_on():
            return
        libproton.exit_scope(self.id, self.name)

    def __call__(self, func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            if get_profiling_on():
                id = libproton.record_scope()
                libproton.enter_scope(id, self.name)
                if self.metrics:
                    libproton.add_metrics(id, self.metrics)
                if self.properties:
                    libproton.set_properties(id, self.properties)
            ret = func(*args, **kwargs)
            if get_profiling_on():
                libproton.exit_scope(id, self.name)
            return ret

        return wrapper


def enter_scope(name: str, *, triton_op: bool = False, metrics: Optional[dict[str, MetricValueType]] = None,
                properties: Optional[dict[str, PropertyValueType]] = None) -> int:
    if not get_profiling_on():
        return -1
    id = libproton.record_scope()
    if not hasattr(thread_local_scopes, "scopes"):
        thread_local_scopes.scopes = []
    thread_local_scopes.scopes.append((id, name))
    if triton_op:
        libproton.enter_op(id, name)
    else:
        libproton.enter_scope(id, name)
    if metrics:
        libproton.add_metrics(id, metrics)
    if properties:
        libproton.set_properties(id, properties)
    return id


def exit_scope(triton_op: bool = False) -> int:
    if not get_profiling_on():
        return -1
    id, name = thread_local_scopes.scopes.pop()
    if triton_op:
        libproton.exit_op(id, name)
    else:
        libproton.exit_scope(id, name)
    return id
