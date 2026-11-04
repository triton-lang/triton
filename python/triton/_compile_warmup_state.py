from contextvars import ContextVar

_COMPILE_WARMUP_ACTIVE = ContextVar("triton_compile_warmup_active", default=False)


def is_compile_warmup():
    return _COMPILE_WARMUP_ACTIVE.get()
