import pkgutil
from importlib.util import module_from_spec
from sys import modules

_backends = []
for module_finder, module_name, is_pkg in pkgutil.iter_modules(
        __path__,
        prefix=__name__ + ".",
):
    # skip .py files (like libdevice.py)
    if not is_pkg:
        continue

    # import backends (like cuda and hip) that are included during setup.py
    spec = module_finder.find_spec(module_name)
    if spec is None or spec.loader is None:
        continue
    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    _backends.append(module_name)
    modules[module_name] = module

__all__ = _backends

del _backends
