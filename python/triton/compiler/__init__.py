from .compiler import (CompiledKernel, compile, get_arch_default_num_stages, get_arch_default_num_warps,
                       instance_descriptor)
from .errors import CompilationError
from .backends.cuda import CUDABackend
from ..common.backend import register_backend

__all__ = [
    "compile", "instance_descriptor", "CompiledKernel", "CompilationError", "get_arch_default_num_warps",
    "get_arch_default_num_stages"
]

register_backend("cuda", CUDABackend)
