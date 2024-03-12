class TritonError(Exception):

    def _format_message(self) -> str:
        return self.error_message

    def __init__(self, error_message: str | None):
        self.error_message = error_message
        self.message = self._format_message()

    def __str__(self):
        return self.message

    def __repr__(self):
        return "{}({!r})".format(type(self).__name__, self.message)
