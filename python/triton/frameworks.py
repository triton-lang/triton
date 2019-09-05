import sys
import os
import libtriton

torch_id = 'torch'
tensorflow_id = 'tensorflow'

torch = None
tensorflow = None
tf_extra_ops = None


def _import_torch():
  global torch
  if torch is None:
    import torch

def _import_tensorflow():
  global tensorflow
  if tensorflow is None:
    import tensorflow

def _import_tf_extra_ops():
  global tf_extra_ops
  if tf_extra_ops is None:
    path = os.path.dirname(libtriton.__file__)
    path = os.path.join(path, 'libextra_tf_ops.so')
    _import_tensorflow()
    tf_extra_ops = tensorflow.load_op_library(path)


def _find_framework(default = None):
    is_tf_imported = 'tensorflow' in sys.modules
    is_torch_imported = 'torch' in sys.modules
    if default:
      if default not in [tensorflow_id, torch_id]:
        raise ValueError('unsupported framework')
      else:
        return default
    elif is_tf_imported and not is_torch_imported:
      return tensorflow_id
    elif is_torch_imported and not is_tf_imported:
      return torch_id
    else:
      raise ValueError('cannot determine imported framework, '
                       'please provide framework argument')