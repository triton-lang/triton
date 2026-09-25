from contextlib import contextmanager
from contextvars import ContextVar

_compilation_target = ContextVar("gluon_compilation_target", default=None)


@contextmanager
def target_context(target):
    token = _compilation_target.set(target)
    try:
        yield
    finally:
        _compilation_target.reset(token)


def get_compilation_target():
    return _compilation_target.get()
