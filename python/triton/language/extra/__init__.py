from ...runtime.driver import driver

backend = driver.active.get_current_target().backend
if backend== 'cuda':
  from .cuda import libdevice
  from . import cuda
  __all__ = ['cuda']
elif backend == 'hip':
  from .hip import libdevice
  from . import hip
  __all__ = ['hip']
else:
  raise RuntimeError('unknown backend')
