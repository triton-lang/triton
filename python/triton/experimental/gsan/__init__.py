from ._allocator import create_mem_pool, get_allocator

__all__ = ["create_mem_pool", "get_allocator"]

_LAZY_LOAD_MODULES = {"symmetric_memory"}


def __getattr__(name):
    if name in _LAZY_LOAD_MODULES:
        import importlib
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__} has no attribute {name}")
