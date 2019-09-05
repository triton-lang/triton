import sys
import os
import libtriton

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

def has_tensorflow():
  return 'tensorflow' in sys.modules

def has_torch():
  return 'torch' in sys.modules