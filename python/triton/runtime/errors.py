from ..errors import TritonError
from typing import Optional


class InterpreterError(TritonError):

    def __init__(self, error_message: Optional[str] = None):
        self.error_message = error_message

    def __str__(self) -> str:
        return self.error_message or ""


class OutOfResources(TritonError):

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
