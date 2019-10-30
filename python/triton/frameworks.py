import sys
import os
import triton._C.libtriton as libtriton

torch = None
tensorflow = None

def _import_torch():
  global torch
  if torch is None:
    import torch

def _import_tensorflow():
  global tensorflow
  if tensorflow is None:
    import tensorflow

def has_tensorflow():
  result = 'tensorflow' in sys.modules
  if result:
    _import_tensorflow()
  return result

def has_torch():
  result = 'torch' in sys.modules
  if result:
    _import_torch()
  return result