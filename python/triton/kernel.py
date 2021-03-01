import os
import struct
from typing import Optional, Dict, List

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
    _triton.arg_type.int1: 'B', _triton.arg_type.int8: 'B', _triton.arg_type.int32: 'I', _triton.arg_type.int64: 'Q',
    _triton.arg_type.half: 'H', _triton.arg_type.float: 'f', _triton.arg_type.double: 'd', _triton.arg_type.buffer: 'P'
}

def th_to_triton(obj):
    tys = {
        torch.int8: 'char', torch.int16: 'short', torch.int32: 'int', torch.int64: 'long', torch.float16: 'half',
        torch.float32: 'float', torch.float64: 'double'
    }
    if isinstance(obj, torch.dtype):
        return tys[obj]
    return str(obj)

def cdiv(a, b):
    return (a + b - 1) // b

def read(path, kernel_names: Optional[List] = None):
    if kernel_names is None:
        kernel_names = []
    with open(path, 'r') as f:
        source = f.read()
        source = _triton.extract_kernels(source, kernel_names)
    return source

class kernel:
    def __init__(self, src, device, defines: Optional[Dict] = None, num_warps: int = 4,
                 autotune_vals: Optional[List] = None, autotune_key: Optional[List] = None):
        if defines is None:
            defines = {}
        if autotune_vals is None:
            autotune_vals = []
        if autotune_key is None:
            autotune_key = []
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
            self.device_id = torch.cuda.current_device() if device.index is None else device.index
            self.device = _triton.cu_device(_torch_utils.cu_device(self.device_id), False)
            self.stream = _triton.cu_stream(_torch_utils.cu_stream(self.device_id), False)
        if device.type == 'cpu':
            self.device_id = -1
            self.device = _triton.host_device()
            self.device = _triton.host_stream()
        _torch_utils.set_device(self.device_id)
        # function
        self.fn = _triton.function(self.src, self.opt, self.device, autotune_vals, autotune_key)
        self.tys = ''.join([codes[x] for x in self.fn.signature()])

    def __call__(self, *args, grid):
        _torch_utils.set_device(self.device_id)
        # pack parameters into a byte buffer
        params = struct.pack(self.tys, *args)
        opt = self.fn.autotune(self.stream, params, grid)
        # run kernel
        grid = grid(opt)
        grid_0 = grid[0]
        grid_1 = 1 if len(grid) < 2 else grid[1]
        grid_2 = 1 if len(grid) < 3 else grid[2]
        self.fn.run(self.stream, params, grid_0, grid_1, grid_2)
