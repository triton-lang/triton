from __future__ import annotations

import hashlib
import json
import re

from .._C.libtriton.triton import (get_env_vars, ir)
from ..common.build import is_hip
# from ..runtime import driver, jit, JITFunction
# TODO: runtime.errors
from ..runtime.autotuner import OutOfResources
from ..runtime.cache import get_cache_manager
from ..runtime.jit import (get_cuda_stream)
from .utils import (TensorMapManager)
from .backends.cuda import CUDABackend

from ..runtime.driver import driver
import torch
from dataclasses import dataclass
from .code_generator import ast_to_ttir
from pathlib import Path


class LazyDict(dict):

    def __getitem__(self, key):
        val = dict.__getitem__(self, key)
        if callable(val):
            return val()
        return val


# ------------------------------------------------------------------------------
# compiler
# ------------------------------------------------------------------------------


def convert_type_repr(x):
    # Currently we only capture the pointer type and assume the pointer is on global memory.
    # TODO: Capture and support shared memory space
    match = re.search(r'!tt\.ptr<([^,]+)', x)
    if match is not None:
        return '*' + convert_type_repr(match.group(1))
    return x


def make_hash(fn, env_vars, device_backend, specialization, options):
    version_key = device_backend.get_version_key()
    env_vars_list = [f"{env_vars[k]}" for k in sorted(env_vars.keys())]
    key = f"{fn.cache_key}-{version_key}-{specialization.hash()}-{options.hash()}-{env_vars_list}"
    return hashlib.md5(key.encode("utf-8")).hexdigest()


# - ^\s*tt\.func\s+ : match the start of the string, any leading whitespace, the keyword func,
#    and any following whitespace
# - (public\s+)? : optionally match the keyword public and any following whitespace
# - (@\w+) : match an @ symbol followed by one or more word characters
#   (letters, digits, or underscores), and capture it as group 1 (the function name)
# - (\((?:%\w+: \S+(?: \{\S+ = \S+ : \S+\})?(?:, )?)*\)) : match a pair of parentheses enclosing
#   zero or more arguments separated by commas, and capture it as group 2 (the argument list)
# - (attributes \{[\S\s]+\})? : optionally match attributes enclosed in braces and capture it as group 3
mlir_prototype_pattern = r"^\s*tt\.func\s+(?:public\s+)?(@\w+)(\((?:%\w+: [\S\s]+(?: \{\S+ = \S+ : \S+\})?(?:, )?)*\))\s*(attributes \{[\S\s]+\})?\s+\{\s*$"
ptx_prototype_pattern = r"\.(?:visible|extern)\s+\.(?:entry|func)\s+(\w+)\s*\(([^)]*)\)"
prototype_pattern = {
    "ttir": mlir_prototype_pattern,
    "ttgir": mlir_prototype_pattern,
    "ptx": ptx_prototype_pattern,
}

# - ((?:[^,\s<]+|<[^>]+>)+): Capturing group that matches one or more of either:
#   [^,\s<]+: One or more characters that are not a comma, whitespace, or the < symbol.
#   |: OR
#   <[^>]+>: A string that starts with < and ends with >, containing any characters except > in between.
mlir_arg_type_pattern = r'%\w+: ((?:[^,\s<]+|<[^>]+>)+),?'
ptx_arg_type_pattern = r"\.param\s+\.(\w+)"
arg_type_pattern = {
    "ttir": mlir_arg_type_pattern,
    "ttgir": mlir_arg_type_pattern,
    "ptx": ptx_arg_type_pattern,
}
if is_hip():
    ttgir_num_warps_pattern = r'"triton_gpu_rocm.num-warps"\s?=\s?(\d+)\s?:'
else:
    ttgir_num_warps_pattern = r'"triton_gpu.num-warps"\s?=\s?(\d+)\s?:'


def parse_mlir_module(path, context):
    module = ir.parse_mlir_module(path, context)
    module.context = context  # module takes ownership of the context
    return module


@dataclass
class InstanceDescriptor:
    divisible_by_16: set = None
    equal_to_1: set = None
    ids_of_folded_args: set = None
    divisible_by_8: set = None

    def hash(self):
        key = str([sorted(x) for x in self.__dict__.values()])
        return hashlib.md5(key.encode("utf-8")).hexdigest()


@dataclass
class SpecializationDescriptor:
    config: InstanceDescriptor
    signature: dict
    constants: dict

    def __post_init__(self):
        if isinstance(self.signature, str):
            self.signature = {k: v.strip() for k, v in enumerate(self.signature.split(","))}
        if self.constants is None:
            self.constants = dict()

    def hash(self):
        key = f"{self.config.hash()}-{self.signature.values()}-{self.constants}"
        return hashlib.md5(key.encode("utf-8")).hexdigest()


def compile(src, device_type=("cuda", None), signature=None, config=InstanceDescriptor(), constants=None,
            extern_libs=None, **kwargs):
    # TODO (backward-breaking):
    #   - merge InstanceDescriptor and SpecializationDescriptor
    #   - extern_libs => linker_flags: dict
    #   - **kwargs -> compiler_flags: dict

    # create backend handler
    backend = CUDABackend(device_type)
    options = backend.parse_options(**kwargs)
    specialization = SpecializationDescriptor(config, signature, constants)

    # create cache manager
    hash = make_hash(src, get_env_vars(), backend, specialization, options=options)
    fn_cache_manager = get_cache_manager(hash)
    name = src.__name__
    metadata_filename = f"{name}.json"
    metadata_group = fn_cache_manager.get_group(metadata_filename) or {}
    metadata_path = metadata_group.get(metadata_filename)
    if metadata_path is not None:
        # cache hit!
        metadata = json.loads(Path(metadata_path).read_text())
        so_path = backend.make_launcher_stub(src, metadata, name, specialization)
        return CompiledKernel(so_path, metadata_path)

    # initialize metadata
    metadata = {
        "constants": constants,
        "device_type": device_type,
        "ids_of_folded_args": tuple([int(k) for k in config.ids_of_folded_args]),
        **options.__dict__,
        **get_env_vars(),
    }
    # run compilation pipeline  and populate metadata
    stages = dict()
    backend.add_stages(extern_libs, stages, options)
    first_stage = list(stages.keys()).index("ttir")
    module = ast_to_ttir(src, specialization, options=options)
    for ext, compile_ir in list(stages.items())[first_stage:]:
        next_module = compile_ir(module, metadata)
        metadata_group[f"{name}.{ext}"] = fn_cache_manager.put(next_module, f"{name}.{ext}")
        module = next_module
    # write-back metadata
    metadata_group[metadata_filename] = fn_cache_manager.put(json.dumps(metadata, default=vars), metadata_filename,
                                                             binary=False)
    fn_cache_manager.put_group(metadata_filename, metadata_group)
    so_path = backend.make_launcher_stub(src, metadata, name, specialization)
    # return handle to compiled kernel
    return CompiledKernel(so_path, metadata_group.get(metadata_filename))


class RuntimeCudaBackend:

    def __init__(self) -> None:
        pass

    def get_load_binary_fn(self):
        return driver.utils.load_binary

    def get_stream(self):
        return get_cuda_stream()

    def get_device_properties(self, device):
        return driver.utils.get_device_properties(device)

    def get_current_device(self):
        return torch.cuda.current_device()

    def set_current_device(self, device):
        torch.cuda.set_device(device)

    def get_kernel_bin(self):
        return "cubin"


class CompiledKernel:

    # Hooks for external tools to monitor the execution of triton kernels
    launch_enter_hook = None
    launch_exit_hook = None
    tensormap_manager = TensorMapManager()

    @staticmethod
    def read_text_or_bytes(path):
        try:
            return path.read_text()
        except UnicodeDecodeError:
            return path.read_bytes()

    def __init__(self, so_path, metadata_path):
        metadata_path = Path(metadata_path)
        self.driver = RuntimeCudaBackend()
        # initialize launcher
        import importlib.util
        spec = importlib.util.spec_from_file_location("__triton_launcher", so_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.c_wrapper = getattr(mod, "launch")
        # initialize metadata
        self.metadata = json.loads(metadata_path.read_text())
        for key, val in self.metadata.items():
            setattr(self, key, val)
        # stores the text of each level of IR that was generated during compilation
        asm_files = [file for file in metadata_path.parent.glob(f'{metadata_path.stem}.*') if file.suffix != '.json']
        self.asm = {file.suffix[1:]: self.read_text_or_bytes(file) for file in asm_files}
        # binaries are lazily initialized
        # because it involves doing runtime things
        # (e.g., checking amount of shared memory on current device)
        self.cu_module = None
        self.cu_function = None

    def _init_handles(self):
        if self.cu_module is not None:
            return

        device = self.driver.get_current_device()
        bin_path = self.driver.get_kernel_bin()
        max_shared = self.driver.get_device_properties(device)["max_shared_mem"]
        fn_load_binary = self.driver.get_load_binary_fn()

        if self.shared > max_shared:
            raise OutOfResources(self.shared, max_shared, "shared memory")

        mod, func, n_regs, n_spills = fn_load_binary(self.metadata["name"], self.asm[bin_path], self.shared, device)

        self.n_spills = n_spills
        self.n_regs = n_regs
        self.cu_module = mod
        self.cu_function = func

    def __getattribute__(self, name):
        if name == 'c_wrapper':
            self._init_handles()
        return super().__getattribute__(name)

    # capture args and expand args with cutensormap*
    def assemble_tensormap_to_arg(self, args):
        args_with_tma = list(args)
        if hasattr(self, 'tensormaps_info'):
            # tuple for hashable
            args_ptr = tuple([arg.data_ptr() if hasattr(arg, 'data_ptr') else arg for arg in args])
            for i, e in enumerate(self.tensormaps_info):
                args_with_tma.append(CompiledKernel.tensormap_manager[(e, args_ptr)])
        return args_with_tma

    def __getitem__(self, grid):
        self._init_handles()

        def runner(*args, stream=None):
            args_expand = self.assemble_tensormap_to_arg(args)
            if stream is None:
                stream = self.driver.get_stream(None)
            self.c_wrapper(grid[0], grid[1], grid[2], self.num_warps, self.num_ctas, self.cluster_dims[0],
                           self.cluster_dims[1], self.cluster_dims[2], self.shared, stream, self.cu_function,
                           CompiledKernel.launch_enter_hook, CompiledKernel.launch_exit_hook, self, *args_expand)

        return runner
