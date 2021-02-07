import os
import struct
import torch
# C bindings
import triton._C.libtriton.triton as _triton
import triton._C.libtriton.torch_utils as _torch_utils
# Make sure internal C resources are cleaned up upon exit
import atexit
@atexit.register
def cleanup():
  _triton.cleanup()

codes = {
  _triton.arg_type.int1:   'B',
  _triton.arg_type.int8:   'B',
  _triton.arg_type.int32:  'I',
  _triton.arg_type.int64:  'Q',
  _triton.arg_type.half:   'H',
  _triton.arg_type.float:  'f',
  _triton.arg_type.double: 'd',
  _triton.arg_type.buffer: 'P'
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
    return tys[obj]
  return str(obj)

def cdiv(a, b):
    return (a + b - 1) // b

def synchronize(device):
    dev_id = device.index
    dev_id = -1 if dev_id is None else dev_id
    _torch_utils.synchronize(dev_id)

def read(path, kernel_names=[]):
  with open(path, 'r') as f:
    source = f.read()
    source = _triton.extract_kernels(source, kernel_names)
  return source

class kernel:

  def __init__(self, src, device, defines = dict(), num_warps = 4, autotune_vals = [], autotune_key = []):
    # check if src is empty
    if src == '':
      raise ValueError('Kernel source code is empty')
    self.src = src
    self.opt = _triton.options()
    self.opt.defines = {k: th_to_triton(v) for k, v in defines.items()}
    self.opt.num_warps = num_warps
    # device
    assert device.type in ['cuda', 'cpu']
    if device.type == 'cuda':
      self.device = torch.cuda.current_device() if device.index is None else device.index
    if device.type == 'cpu':
      self.device = -1
    _torch_utils.register_device(self.device)
    _torch_utils.register_stream(self.device)
    # C++ function wrapper
    self.op_id = _triton.make_op_id()
    _triton.register_fn(self.op_id, self.device, self.src, self.opt, autotune_vals, autotune_key)
    # debug mode
    self.is_debug = 'TRITON_DEBUG' in os.environ
    # signature
    arg_types = _triton.get_fn_signature(self.op_id)
    self.tys = ''.join([codes[x] for x in arg_types])

  def __call__(self, *args, grid):
    _torch_utils.set_device(self.device)
    # pack parameters into a byte buffer
    params = struct.pack(self.tys, *args)
    opt = _triton.autotune(self.op_id, self.device, params, grid)
    # run kernel
    grid = grid(opt)
    grid_0 = grid[0]
    grid_1 = 1 if len(grid) < 2 else grid[1]
    grid_2 = 1 if len(grid) < 3 else grid[2]
    _triton.launch_kernel(self.op_id, self.device, params, grid_0, grid_1, grid_2)