from ._allocator import ShareableHandleType, configure, create_mem_pool, freeze_config, get_allocator
from ._stream_sync import warmup_gsan_kernels

__all__ = [
    "ShareableHandleType",
    "configure",
    "create_mem_pool",
    "freeze_config",
    "get_allocator",
    "warmup_gsan_kernels",
]

_LAZY_LOAD_MODULES = {"symmetric_memory"}


def __getattr__(name):
    if name in _LAZY_LOAD_MODULES:
        import importlib
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__} has no attribute {name}")
