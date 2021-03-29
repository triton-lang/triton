import os
import struct
from typing import Optional, Dict, List, Callable
import torch
import triton._C.libtriton.triton as _triton

codes = {
    _triton.runtime.arg_type.int1: 'B',
    _triton.runtime.arg_type.int8: 'B',
    _triton.runtime.arg_type.int32: 'I',
    _triton.runtime.arg_type.int64: 'Q',
    _triton.runtime.arg_type.half: 'H',
    _triton.runtime.arg_type.float: 'f',
    _triton.runtime.arg_type.double: 'd',
    _triton.runtime.arg_type.buffer: 'P',
}


def th_to_triton(obj):
    """ Convert a `torch.dtype` to a Triton-C type string. """
    tys = {
        torch.int8: 'char',
        torch.int16: 'short',
        torch.int32: 'int',
        torch.int64: 'long',
        torch.float16: 'half',
        torch.float32: 'float',
        torch.float64: 'double',
    }
    if isinstance(obj, torch.dtype):
        return tys[obj]
    return str(obj)


def cdiv(a: int, b: int) -> int:
    """ Ceil division (a + b - 1) // b"""
    return (a + b - 1) // b


def read(path: str, kernel_names: Optional[List] = None) -> str:
    """ Extracts the source code for `kernel_names` from the given `path` file."""
    if kernel_names is None:
        kernel_names = []
    with open(path, 'r') as f:
        source = f.read()
        source = _triton.tools.extract_kernels(source, kernel_names)
    return source


config = _triton.runtime.config


class kernel:
    """
    A class used to represent a Triton kernel.
    """
    def __init__(
        self,
        src: str,
        device: torch.device,
        defines: Optional[Dict] = None,
        num_warps: int = 4,
        autotune_configs: Optional[List] = None,
        autotune_key: Optional[List] = None
    ):
        """
        :param src: The source code of the kernel.
        :param device: The device to compile the kernel for.
        :param defines: A dictionary of preprocessor #define for the compiler.
        :param num_warps: Optimization flag for the compiler's internal auto-parallelization engine.
        :param autotune_configs: A list of triton.config objects for the autotuner to try.
        :param autotune_key: A list of kernel argument names whose change in value should trigger the autotuner to re-run.
        """

        if defines is None:
            defines = {}
        if autotune_configs is None:
            autotune_configs = []
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
        # autotune_configs = [({}, 4)]
        self.fn = _triton.runtime.function(self.src, self.opt, self.device, autotune_configs, autotune_key)
        self.tys = ''.join([codes[x] for x in self.fn.signature()])

    def __call__(self, *args, grid: Callable[[_triton.runtime.options], tuple]):
        """
        Runs the kernel on the given arguments and launch grid.
        :param args: The arguments to the kernel in the orders that they appear in the Triton-C source.
        :param grid: The launch grid for the kernel, i.e., callable that transform compilation options into a tuple of at most 3 integers.
        :return: None
        """
        # make sure that the executing thread is on the right device
        torch.cuda.set_device(self.device_id)
        # pack parameters into a byte buffer
        params = struct.pack(self.tys, *args)
        kernel = self.fn.autotune(params, grid, self.stream)
        # run kernel
        grid = grid(kernel.opt)
        kernel(params, self.stream, grid)
