class TritonError(Exception):

    def __init__(self, error_message: str | None):
        self.error_message = error_message

    def __str__(self) -> str:
        return self.error_message or ""

    def __repr__(self) -> str:
        return "{}({!r})".format(type(self).__name__, self.error_message)
