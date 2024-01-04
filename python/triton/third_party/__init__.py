from functools import lru_cache
import os
import importlib
import inspect
from dataclasses import dataclass
from .driver import DriverBase
from .compiler import BaseBackend

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name[:-3], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def find_concrete_subclasses(module, base_class):
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


def initialize():
    backends = []
    root = os.path.dirname(__file__)
    for backend in os.listdir(root):
        if not os.path.isdir(os.path.join(root, backend)):
            continue
        if backend.startswith('__'):
            continue
        compiler = load_module(backend[:-3], os.path.join(root, backend, 'compiler.py'))
        driver = load_module(backend[:-3], os.path.join(root, backend, 'driver.py'))
        backends.append(Backend(find_concrete_subclasses(compiler, BaseBackend),
                                find_concrete_subclasses(driver, DriverBase)))
    return backends
    

backends = initialize()

