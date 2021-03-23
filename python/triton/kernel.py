import os
import struct
from typing import Optional, Dict, List
import torch
# C bindings
import triton._C.libtriton.triton as _triton

codes = {
    _triton.runtime.arg_type.int1: 'B',
    _triton.runtime.arg_type.int8: 'B',
    _triton.runtime.arg_type.int32: 'I',
    _triton.runtime.arg_type.int64: 'Q',
    _triton.runtime.arg_type.half: 'H',
    _triton.runtime.arg_type.float: 'f',
    _triton.runtime.arg_type.double: 'd',
    _triton.runtime.arg_type.buffer: 'P'
}


def th_to_triton(obj):
    tys = {
        torch.int8: 'char', torch.int16: 'short', torch.int32: 'int', torch.int64: 'long',\
        torch.float16: 'half', torch.float32: 'float', torch.float64: 'double'
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
        source = _triton.tools.extract_kernels(source, kernel_names)
    return source


config = _triton.runtime.config


class kernel:
    def __init__(
        self,
        src,
        device,
        defines: Optional[Dict] = None,
        num_warps: int = 4,
        autotune_vals: Optional[List] = None,
        autotune_key: Optional[List] = None
    ):
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
        # device
        assert device.type in ['cuda', 'cpu']
        if device.type == 'cuda':
            self.device_id = torch.cuda.current_device() if device.index is None else device.index
            self.device = _triton.driver.cu_device(self.device_id, False)
            cu_stream = torch.cuda.current_stream(self.device_id).cuda_stream
            self.stream = _triton.driver.cu_stream(cu_stream, False)
        if device.type == 'cpu':
            self.device_id = -1
            self.device = _triton.driver.host_device()
            self.device = _triton.driver.host_stream()
        torch.cuda.set_device(self.device_id)
        # function
        self.opt = _triton.runtime.options()
        self.opt.defines = {k: th_to_triton(v) for k, v in defines.items()}
        self.opt.num_warps = num_warps
        # autotune_vals = [({}, 4)]
        self.fn = _triton.runtime.function(self.src, self.opt, self.device, autotune_vals, autotune_key)
        self.tys = ''.join([codes[x] for x in self.fn.signature()])

    def __call__(self, *args, grid):
        # make sure that the executing thread is on the right device
        torch.cuda.set_device(self.device_id)
        # pack parameters into a byte buffer
        params = struct.pack(self.tys, *args)
        kernel = self.fn.autotune(params, grid, self.stream)
        # run kernel
        grid = grid(kernel.opt)
        kernel(params, self.stream, grid)
