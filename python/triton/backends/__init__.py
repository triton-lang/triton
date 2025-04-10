import os
import importlib.util
import inspect
from dataclasses import dataclass
from typing import Type, TypeVar
from types import ModuleType
from .driver import DriverBase
from .compiler import BaseBackend


def _load_module(name: str, path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if not spec:
        raise RuntimeError(f"Unable to load {name} from {path}, ModuleSpec could not be created")
    module = importlib.util.module_from_spec(spec)
    loader = spec.loader
    if not loader:
        raise RuntimeError(f"Unable to load {name} from {path}, no Loader on the ModuleSpec")
    loader.exec_module(module)
    return module


T = TypeVar("T", bound=BaseBackend | DriverBase)


def _find_concrete_subclasses(module: ModuleType, base_class: Type[T]) -> Type[T]:
    ret: list[Type[T]] = []
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
    compiler: Type[BaseBackend]
    driver: Type[DriverBase]


def _discover_backends() -> dict[str, Backend]:
    backends = dict()
    root = os.path.dirname(__file__)
    for name in os.listdir(root):
        if not os.path.isdir(os.path.join(root, name)):
            continue
        if name.startswith('__'):
            continue
        compiler = _load_module(name, os.path.join(root, name, 'compiler.py'))
        driver = _load_module(name, os.path.join(root, name, 'driver.py'))
        backends[name] = Backend(_find_concrete_subclasses(compiler, BaseBackend),  # type: ignore
                                 _find_concrete_subclasses(driver, DriverBase))  # type: ignore
    return backends


backends: dict[str, Backend] = _discover_backends()
