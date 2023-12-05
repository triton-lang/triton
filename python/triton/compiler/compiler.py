from __future__ import annotations

import hashlib
import json

from .._C.libtriton.triton import (get_env_vars, ir)
# from ..runtime import driver, jit, JITFunction
# TODO: runtime.errors
from ..runtime.autotuner import OutOfResources
from ..runtime.cache import get_cache_manager
from ..runtime.driver import driver
from .utils import InfoFromBackendForTensorMap
from .backends.cuda import CUDABackend
from dataclasses import dataclass
from .code_generator import ast_to_ttir
from pathlib import Path
import re


@dataclass
class AttrsDescriptor:
    divisible_by_16: set = None
    equal_to_1: set = None
    ids_of_folded_args: set = None
    divisible_by_8: set = None

    def __post_init__(self):
        if self.divisible_by_16 is None:
            self.divisible_by_16 = set()
        if self.equal_to_1 is None:
            self.equal_to_1 = set()
        if self.ids_of_folded_args is None:
            self.ids_of_folded_args = set()
        if self.divisible_by_8 is None:
            self.divisible_by_8 = set()

    def hash(self):
        key = str([sorted(x) for x in self.__dict__.values()])
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

mlir_arg_type_pattern = r'%\w+: ((?:[^,\s<]+|<[^>]+>)+),?'
ptx_arg_type_pattern = r"\.param\s+\.(\w+)"
arg_type_pattern = {
    "ttir": mlir_arg_type_pattern,
    "ttgir": mlir_arg_type_pattern,
    "ptx": ptx_arg_type_pattern,
}


def convert_type_repr(x):
    # Currently we only capture the pointer type and assume the pointer is on global memory.
    # TODO: Capture and support shared memory space
    match = re.search(r'!tt\.ptr<([^,]+)', x)
    if match is not None:
        return '*' + convert_type_repr(match.group(1))
    return x


def _get_num_warps_from_ir_str(src: str):
    ttgir_num_warps_pattern = r'"triton_gpu.num-warps"\s?=\s?(\d+)\s?:'
    # TODO(jlebar): Using a regex to get num-warps is a hack, and will break if
    # e.g. someone has an instruction (not module) attribute named "num-warps".
    num_warps_matches = re.findall(ttgir_num_warps_pattern, src)
    assert len(num_warps_matches) == 1, "Expected exactly one match for num_warps"
    num_warps = int(num_warps_matches[0])

    # If warp specialization is enabled, the true number of warps from
    # the perspective of e.g. CUDA is num-warps times the number of
    # specialized groups.
    num_warp_groups_matches = re.findall(r'"triton_gpu.num-warp-groups-per-cta"\s?=\s?(\d+)\s?:', src)
    assert len(num_warp_groups_matches) == 0 or len(num_warp_groups_matches) == 1, \
      "Expected triton_gpu.num-warp-groups-per-cta attribute to appear 0 or 1 times"
    if num_warp_groups_matches:
        num_warps *= int(num_warp_groups_matches[0])

    return num_warps


class ASTSource:

    def __init__(self, fn, signature, constants=None, attrs=None) -> None:
        self.fn = fn
        self.ext = "ttir"
        self.name = fn.__name__
        self.signature = signature
        self.constants = constants
        self.attrs = attrs
        if isinstance(self.signature, str):
            self.signature = {k: v.strip() for k, v in enumerate(self.signature.split(","))}
        if self.constants is None:
            self.constants = dict()
        if self.attrs is None:
            self.attrs = AttrsDescriptor()

    def hash(self):
        key = f"{self.fn.cache_key}-{self.attrs.hash()}-{self.signature.values()}-{self.constants}"
        return hashlib.md5(key.encode("utf-8")).hexdigest()

    def make_ir(self, options):
        return ast_to_ttir(self.fn, self, options=options)

    def metadata(self):
        # TODO: remove once TMA support is cleaned up
        return {"ids_of_folded_args": tuple([int(k) for k in self.attrs.ids_of_folded_args])}

    def parse_options(self):
        return dict()


class IRSource:

    def __init__(self, path):
        self.path = path
        path = Path(path)
        self.ext = path.suffix[1:]
        self.src = path.read_text()
        match = re.search(prototype_pattern[self.ext], self.src, re.MULTILINE)
        self.name = match.group(1)
        signature = match.group(2)
        types = re.findall(arg_type_pattern[self.ext], signature)
        self.signature = {k: convert_type_repr(ty) for k, ty in enumerate(types)}

    def hash(self):
        return hashlib.md5(self.src.encode("utf-8")).hexdigest()

    def make_ir(self, options):
        context = ir.context()
        module = ir.parse_mlir_module(self.path, context)
        module.context = context
        return module

    def metadata(self):
        return dict()

    def parse_options(self):
        if self.ext == "ttgir":
            return {'num_warps': _get_num_warps_from_ir_str(self.src)}
        return dict()


def compile(src, target=None, options=None):
    if target is None:
        target = driver.get_current_target()
    backend = CUDABackend(target)
    # create backend
    if not isinstance(src, ASTSource):
        assert isinstance(src, str), "source must be either AST or a filepath"
        src = IRSource(src)
    extra_options = src.parse_options()
    options = backend.parse_options(dict(options or dict(), **extra_options))
    # create cache manager
    key = f"{src.hash()}-{backend.hash()}-{options.hash()}-{frozenset(sorted(get_env_vars().items()))}"
    hash = hashlib.md5(key.encode("utf-8")).hexdigest()
    fn_cache_manager = get_cache_manager(hash)
    metadata_filename = f"{src.name}.json"
    metadata_group = fn_cache_manager.get_group(metadata_filename) or {}
    metadata_path = metadata_group.get(metadata_filename)
    if metadata_path is not None:
        # cache hit!
        metadata = json.loads(Path(metadata_path).read_text())
        so_path = backend.make_launcher_stub(src, metadata)
        return CompiledKernel(so_path, metadata_path)
    # initialize metadata
    metadata = {
        "target": target,
        **options.__dict__,
        **get_env_vars(),
        **src.metadata(),
    }
    # run compilation pipeline  and populate metadata
    stages = dict()
    backend.add_stages(stages, options)
    first_stage = list(stages.keys()).index(src.ext)
    module = src.make_ir(options)
    for ext, compile_ir in list(stages.items())[first_stage:]:
        next_module = compile_ir(module, metadata)
        metadata_group[f"{src.name}.{ext}"] = fn_cache_manager.put(next_module, f"{src.name}.{ext}")
        module = next_module
    # write-back metadata
    metadata_group[metadata_filename] = fn_cache_manager.put(json.dumps(metadata, default=vars), metadata_filename,
                                                             binary=False)
    fn_cache_manager.put_group(metadata_filename, metadata_group)
    so_path = backend.make_launcher_stub(src, metadata)
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
        self.run = getattr(mod, "launch")
        # initialize metadata
        self.metadata = json.loads(metadata_path.read_text())
        self.metadata['tensormaps_info'] = [InfoFromBackendForTensorMap(e) for e in self.metadata['tensormaps_info']
                                            ] if 'tensormaps_info' in self.metadata else []
        for i, _ in enumerate(self.metadata["tensormaps_info"]):
            self.metadata["tensormaps_info"][i].ids_of_folded_args = tuple(self.metadata["ids_of_folded_args"])
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
        device = driver.get_current_device()
        # not enough shared memory to run the kernel
        max_shared = driver.utils.get_device_properties(device)["max_shared_mem"]
        if self.shared > max_shared:
            raise OutOfResources(self.shared, max_shared, "shared memory")
        # TODO: n_regs, n_spills should be metadata generated when calling `ptxas`
        self.module, self.function, self.n_regs, self.n_spills = driver.utils.load_binary(
            self.name, self.kernel, self.shared, device)

    def __getattribute__(self, name):
        if name == 'run':
            self._init_handles()
        return super().__getattribute__(name)

    def __getitem__(self, grid):
        self._init_handles()

        def runner(*args, stream=None):
            args_expand = driver.assemble_tensormap_to_arg(self.tensormaps_info, args)
            if stream is None:
                device = driver.get_current_device()
                stream = driver.get_current_stream(device)
            self.run(grid[0], grid[1], grid[2], self.num_warps, self.num_ctas, self.cluster_dims[0],
                     self.cluster_dims[1], self.cluster_dims[2], self.shared, stream, self.function,
                     CompiledKernel.launch_enter_hook, CompiledKernel.launch_exit_hook, self, *args_expand)

        return runner
