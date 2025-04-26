import importlib
import inspect
import sys
from dataclasses import dataclass
from .driver import DriverBase
from .compiler import BaseBackend

if sys.version_info >= (3, 10):
    from importlib.metadata import entry_points
else:
    from importlib_metadata import entry_points


def _find_concrete_subclasses(module, base_class):
    ret = []
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if isinstance(attr, type) and issubclass(attr, base_class) and not inspect.isabstract(attr):
            ret.append(attr)
    if len(ret) == 0:
        raise RuntimeError(f"Found 0 concrete subclasses of {base_class} in {module}: {ret}")
    if len(ret) > 1:
        raise RuntimeError(f"Found >1 concrete subclasses of {base_class} in {module}: {ret}")
    return ret[0]


@dataclass(frozen=True)
class Backend:
    compiler: BaseBackend = None
    driver: DriverBase = None


def _discover_backends():
    backends = dict()
    for ep in entry_points().select(group="triton.backends"):
        compiler = importlib.import_module(f"{ep.value}.compiler")
        driver = importlib.import_module(f"{ep.value}.driver")
        backends[ep.name] = Backend(_find_concrete_subclasses(compiler, BaseBackend),
                                    _find_concrete_subclasses(driver, DriverBase))
    return backends


backends = _discover_backends()
