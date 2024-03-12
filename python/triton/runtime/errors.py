from ..errors import TritonError


class InterpreterError(TritonError):

    def __init__(self, error_message: str | None = None):
        super().__init__(error_message)


class OutOfResources(TritonError):

    def _format_message(self) -> str:
        message = f"out of resource: {self.name}, " f"Required: {self.required}, " f"Hardware limit: {self.limit}"
        message += ". Reducing block sizes or `num_stages` may help."

    def __init__(self, required, limit, name):
        self.required = required
        self.limit = limit
        self.name = name
        super().__init__(None)

    def __reduce__(self):
        # this is necessary to make CompilationError picklable
        return (type(self), (self.required, self.limit, self.name))
