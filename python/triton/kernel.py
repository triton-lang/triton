import triton._C.libtriton as libtriton
import os
import time
from struct import pack
import torch

codes = {
  libtriton.arg_type.int1:   'B',
  libtriton.arg_type.int8:   'B',
  libtriton.arg_type.int32:  'I',
  libtriton.arg_type.int64:  'Q',
  libtriton.arg_type.half:   'H',
  libtriton.arg_type.float:  'f',
  libtriton.arg_type.double: 'd',
  libtriton.arg_type.buffer: 'P'
}

sizes = {
  libtriton.arg_type.int1:   1,
  libtriton.arg_type.int8:   1,
  libtriton.arg_type.int32:  4,
  libtriton.arg_type.int64:  8,
  libtriton.arg_type.half:   2,
  libtriton.arg_type.float:  4,
  libtriton.arg_type.double: 8,
  libtriton.arg_type.buffer: 8
}


def th_to_triton(obj):
  tys = {
    torch.int8: 'char',
    torch.int16: 'short',
    torch.int32: 'int',
    torch.int64: 'long',
    torch.float16: 'half',
    torch.float32: 'float',
    torch.float64: 'double'
  }
  if isinstance(obj, torch.dtype):
    return [tys[obj]]
  if isinstance(obj, list):
    return [th_to_triton(x)[0] for x in obj]
  return [str(obj)]


def cdiv(a, b):
    return (a + b - 1) // b

def cdiv_sum(a, b):
    return torch.ops.triton.cdiv_sum(a, b)

class kernel:

  def __init__(self, src, defines = dict(), num_warps = [2, 4, 8]):
    self.src = src
    self.opt = libtriton.options_space()
    self.opt.defines = [(k, th_to_triton(v)) for k, v in defines.items()]
    self.opt.num_warps = num_warps
    self.op_id = libtriton.make_op_id()
    self.registered = set()
    arg_types = libtriton.get_fn_signature(self.src, self.opt)
    size = sum([sizes[x] for x in arg_types])
    self.tys = ''.join([codes[x] for x in arg_types])

  def set_constant(self, device, name, value):
    libtriton.register_cst((self.op_id, device), name, value)

  def __call__(self, *args, **kwargs):
    for x in args:
      if isinstance(x, torch.Tensor):
        device = x.device.index
        device = -1 if device is None else device
        break
    # lazily register function for device
    if device not in self.registered:
      self.registered.add(device)
      libtriton.register_fn((self.op_id, device), self.src, self.opt, os.path.realpath(libtriton.__file__))
    # launch grid
    if 'grid' not in kwargs:
      raise RuntimeError('Must provide grid for kernel launch')
    grid = kwargs['grid']
    libtriton.register_grid((self.op_id, device), grid)
    # launch
    #print(self.tys)
    params = pack(self.tys, *[x.data_ptr() if isinstance(x, torch.Tensor) else x for x in args])
    torch.cuda.synchronize()
    torch.ops.triton.launch_kernel(self.op_id, device, params)