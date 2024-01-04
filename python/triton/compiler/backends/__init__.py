from functools import lru_cache
import os
import importlib
from ...common.backend import BaseBackend


@lru_cache
def make_backend(target):
    backends = list()
    backends_dir = os.path.dirname(__file__)
    for filename in os.listdir(backends_dir):
        if filename.endswith('.py') and not filename.startswith('__'):
            file_path = os.path.join(backends_dir, filename)
            spec = importlib.util.spec_from_file_location(filename[:-3], file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            backend = getattr(module, "backend")
            assert issubclass(backend, BaseBackend)
            backends.append(backend)
    ret = [backend for backend in backends if backend.name == target[0]]
    if len(ret) == 0:
        raise RuntimeError(f"Found 0 matchin backend for target '{target}'")
    if len(ret) > 1:
        raise RuntimeError(f"Found more than 1 matching backend for target '{target}'")
    return ret[0](target)
