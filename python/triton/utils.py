import triton.frameworks as fw
import triton._C.libtriton as libtriton
import numpy as np

def cdiv(a, b):
    return -(-a // b)

class tf_empty_proxy:

  def __init__(self, shape, dtype):
    self.shape = shape
    self.dtype = dtype
    self.tensor = None
  
  def to_tensor(self):
    assert self.tensor is not None
    return self.tensor

def empty(shape, dtype):
  if fw.has_tensorflow():
    shape = [fw.tensorflow.constant(x) for x in shape]
    shape = fw.tensorflow.stack(shape)
    return tf_empty_proxy(shape, dtype)
    #return fw.tf_extra_ops.alloc_empty(args, T = dtype)
  elif fw.has_torch():
    return fw.torch.empty(*shape).cuda()

def shape(A) :
  if fw.has_tensorflow():
    return A.shape.as_list()
  elif fw.has_torch():
    return A.shape
  else:
    assert False


class id_dict:

  # Lazy entry for e.g., tensorflow, when value of benchmark is
  # not known at graph compile time
  class lazy_entry:
    def __init__(self, id):
      self.id = id

    def get(self):
      return libtriton.retrieve_scalar(self.id)


  def __init__(self):
    self.data = dict()

  def __delitem__(self, key):
    del self.data[id(key)]

  def __getitem__(self, key):
    ret = self.data[id(key)]
    if isinstance(ret, id_dict.lazy_entry):
      return ret.get()
    return ret

  def __len__(self):
    return len(self.data)

  def __setitem__(self, key, value):
    self.data[id(key)] = value