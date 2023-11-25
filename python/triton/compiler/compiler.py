from __future__ import annotations

import hashlib
import json

from .._C.libtriton.triton import (get_env_vars)
# from ..runtime import driver, jit, JITFunction
# TODO: runtime.errors
from ..runtime.autotuner import OutOfResources
from ..runtime.cache import get_cache_manager
from ..runtime.jit import get_current_device
from ..runtime.driver import driver
from .backends.cuda import CUDABackend
from dataclasses import dataclass
from .code_generator import ast_to_ttir
from pathlib import Path


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

    # create backend
    backend = CUDABackend(device_type)
    options = backend.parse_options(**kwargs)
    specialization = SpecializationDescriptor(config, signature, constants)

    # create cache manager
    key = f"{src.cache_key}-{backend.hash()}-{specialization.hash()}-{options.hash()}-{frozenset(get_env_vars().items())}"
    hash = hashlib.md5(key.encode("utf-8")).hexdigest()
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


class CompiledKernel:

    # Hooks for external tools to monitor the execution of triton kernels
    # TODO: move out of this namespace since it's a runtime thing
    launch_enter_hook = None
    launch_exit_hook = None

    def __init__(self, so_path, metadata_path):
        metadata_path = Path(metadata_path)
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
        self.asm = {
            file.suffix[1:]: file.read_bytes() if file.suffix[1:] == driver.binary_ext else file.read_text()
            for file in asm_files
        }
        self.kernel = self.asm[driver.binary_ext]
        # binaries are lazily initialized
        # because it involves doing runtime things
        # (e.g., checking amount of shared memory on current device)
        self.module = None
        self.function = None

    def _init_handles(self):
        if self.module is not None:
            return
        device = get_current_device()
        # not enough shared memory to run the kernel
        max_shared = driver.utils.get_device_properties(device)["max_shared_mem"]
        if self.shared > max_shared:
            raise OutOfResources(self.shared, max_shared, "shared memory")
        # TODO: n_regs, n_spills should be metadata generated when calling `ptxas`
        self.module, self.function, self.n_regs, self.n_spills = driver.utils.load_binary(
            self.name, self.kernel, self.shared, device)

    def __getattribute__(self, name):
        if name == 'c_wrapper':
            self._init_handles()
        return super().__getattribute__(name)
