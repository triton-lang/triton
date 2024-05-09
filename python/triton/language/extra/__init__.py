from ...runtime.driver import driver
import os
import importlib

_backend = driver.active.get_current_target().backend
if _backend == 'cuda':
    from . import cuda
    __all__ = ['cuda']
elif _backend == 'hip':
    from . import hip
    __all__ = ['hip']
else:
    raise RuntimeError('unknown backend')

_curr_dir = os.path.dirname(os.path.realpath(__file__))
_module_path = os.path.join(_curr_dir, _backend, 'libdevice.py')
_module_spec = importlib.util.spec_from_file_location('triton.language.extra.libdevice', _module_path)
libdevice = importlib.util.module_from_spec(_module_spec)
_module_spec.loader.exec_module(libdevice)
