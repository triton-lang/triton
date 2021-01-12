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
  
def synchronize(device):
    dev_id = device.index
    dev_id = -1 if dev_id is None else dev_id
    torch.ops.triton.synchronize(dev_id)

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

  def asm(self, mode, device, **kwargs):
    dev_id = device.index
    # assembly mode
    supported = {
      'ptx': libtriton.asm_mode.ptx,
      'sass': libtriton.asm_mode.sass,
    }
    if mode not in supported:
      raise('ASM mode must be in ', supported.keys())
    mode = supported[mode]
    # disambiguates #defines
    libtriton.register_fn((self.op_id, dev_id), self.src, self.opt)
    def _single_value_or_err(x, key):
      if isinstance(x, list) and len(x) == 1:
        return x[0]
      if isinstance(x, list) and len(x) > 1:
        if key in kwargs:
          return kwargs[key]
        raise ValueError(f'Parameter {key}={x} was auto-tuned during kernel creation. '
                          'Please supply an explicit value as a keyword argument.')
      return str(x)
    defines = dict()
    for (D, V) in self.opt.defines:
      defines[D] = _single_value_or_err(V, D)
    opt = libtriton.options()
    opt.num_warps = _single_value_or_err(self.opt.num_warps, 'num_warps')
    opt.defines = defines
    # run
    return libtriton.get_fn_asm((self.op_id, dev_id), mode, opt)

  def __call__(self, *args, **kwargs):
    if 'TRITON_DEBUG_MODE' in os.environ:
      _args = args
      args = [x.clone() if isinstance(x, torch.Tensor) else x for x in _args]
      for i in range(len(args)):
        if isinstance(args[i], torch.Tensor):
          args[i] = torch.ops.triton.cuda_empty_like(args[i])
          args[i].copy_(_args[i])
      torch.cuda.synchronize()
    for x in args:
      if isinstance(x, torch.Tensor):
        device = x.device.index
        device = -1 if device is None else device
        break
    # lazily register function for device
    libtriton.register_fn((self.op_id, device), self.src, self.opt)
    # launch grid
    if 'grid' not in kwargs:
      raise RuntimeError('Must provide grid for kernel launch')
    grid = kwargs['grid']
    libtriton.register_grid((self.op_id, device), grid)
    # re-allocate buffers for auto-tuning
    if 'autotune_buf' in kwargs:
      pass
    # launch
    params    = pack(self.tys, *[x.data_ptr() if isinstance(x, torch.Tensor) else x for x in args])
    names     = list(kwargs['constants'].keys()) if 'constants' in kwargs else []
    constants = list(kwargs['constants'].values()) if 'constants' in kwargs else []
    torch.ops.triton.launch_kernel(self.op_id, device, params, names, constants)
    if 'TRITON_DEBUG_MODE' in os.environ:
      torch.cuda.synchronize()
      for i in range(len(args)):
        if isinstance(args[i], torch.Tensor):
          _args[i].copy_(args[i].clone())
      args = _args