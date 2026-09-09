from ..errors import TritonError
from typing import Optional


class InterpreterError(TritonError):
    """Raised when an error occurs while executing a kernel in the interpreter (:code:`TRITON_INTERPRET=1`)."""

    def __init__(self, error_message: Optional[str] = None):
        self.error_message = error_message

    def __str__(self) -> str:
        return self.error_message or ""


class OutOfResources(TritonError):
    """
    Raised when a kernel needs more of a hardware resource than is available,
    for example more shared memory or registers than the device provides.

    Reducing block sizes or :code:`num_stages` often resolves it. The resource is
    named by :attr:`name`, with the requested and available amounts in
    :attr:`required` and :attr:`limit`.
    """

    def __init__(self, required, limit, name):
        self.required = required
        self.limit = limit
        self.name = name

    def __str__(self) -> str:
        return f"out of resource: {self.name}, Required: {self.required}, Hardware limit: {self.limit}. Reducing block sizes or `num_stages` may help."

    def __reduce__(self):
        # this is necessary to make CompilationError picklable
        return (type(self), (self.required, self.limit, self.name))


class PTXASError(TritonError):

    def __init__(self, error_message: Optional[str] = None):
        self.error_message = error_message

    def __str__(self) -> str:
        error_message = self.error_message or ""
        return f"PTXAS error: {error_message}"


class AutotunerError(TritonError):

    def __init__(self, error_message: Optional[str] = None):
        self.error_message = error_message

    def __str__(self) -> str:
        error_message = self.error_message or ""
        return f"Autotuner error: {error_message}"
