from __future__ import annotations

import functools
import hashlib
import json
import os
import re
import subprocess
import tempfile
from collections import namedtuple
from pathlib import Path
from typing import Any, Tuple

import torch

import triton
import triton._C.libtriton.triton as _triton
# TODO: runtime.errors
from ..runtime.autotuner import OutOfResources
from ..runtime.cache import get_cache_manager
from ..runtime.driver import get_cuda_utils, get_hip_utils
from ..tools.disasm import extract
from .code_generator import ast_to_ttir
from .make_launcher import make_stub


def inline_triton_ir(mod):
    pm = _triton.ir.pass_manager(mod.context)
    pm.enable_debug()
    pm.add_inliner_pass()
    pm.run(mod)
    return mod


def ttir_compute_capability_rewrite(mod, arch):
    # For hardware without support, we must rewrite all load/store
    # with block (tensor) pointers into tensors of pointers
    pm = _triton.ir.pass_manager(mod.context)
    pm.enable_debug()
    if _is_cuda(arch):
        pm.add_rewrite_tensor_pointer_pass(arch)
    pm.run(mod)
    return mod


def optimize_ttir(mod, arch):
    mod = inline_triton_ir(mod)
    mod = ttir_compute_capability_rewrite(mod, arch)
    pm = _triton.ir.pass_manager(mod.context)
    pm.enable_debug()
    pm.add_inliner_pass()
    pm.add_triton_combine_pass()
    pm.add_canonicalizer_pass()
    pm.add_cse_pass()
    pm.add_licm_pass()
    pm.add_symbol_dce_pass()
    pm.run(mod)
    return mod


def ttir_to_ttgir(mod, num_warps):
    pm = _triton.ir.pass_manager(mod.context)
    pm.add_convert_triton_to_tritongpu_pass(num_warps)
    pm.run(mod)
    return mod


def optimize_ttgir(mod, num_stages, arch):
    pm = _triton.ir.pass_manager(mod.context)
    pm.enable_debug()
    pm.add_tritongpu_coalesce_pass()
    pm.add_tritongpu_remove_layout_conversions_pass()
    if isinstance(arch, int):
        pm.add_tritongpu_accelerate_matmul_pass(arch)
    pm.add_tritongpu_remove_layout_conversions_pass()
    pm.add_tritongpu_optimize_dot_operands_pass()
    pm.add_tritongpu_pipeline_pass(num_stages)
    pm.add_tritongpu_prefetch_pass()
    pm.add_tritongpu_optimize_dot_operands_pass()
    pm.add_tritongpu_remove_layout_conversions_pass()
    pm.add_tritongpu_decompose_conversions_pass()
    pm.add_tritongpu_reorder_instructions_pass()
    pm.add_cse_pass()
    pm.add_symbol_dce_pass()
    pm.run(mod)
    return mod


def _add_external_libs(mod, libs):
    for name, path in libs.items():
        if len(name) == 0 or len(path) == 0:
            return
    _triton.add_external_libs(mod, list(libs.keys()), list(libs.values()))


def ttgir_to_llir(mod, extern_libs, arch):
    if extern_libs:
        _add_external_libs(mod, extern_libs)
    # TODO: separate tritongpu_to_llvmir for different backends
    if _is_cuda(arch):
        return _triton.translate_triton_gpu_to_llvmir(mod, arch, False)
    else:
        return _triton.translate_triton_gpu_to_llvmir(mod, 0, True)


# PTX translation

@functools.lru_cache
def ptx_get_version(cuda_version) -> int:
    '''
    Get the highest PTX version supported by the current CUDA driver.
    '''
    assert isinstance(cuda_version, str)
    major, minor = map(int, cuda_version.split('.'))
    if major == 12:
        return 80 + minor
    if major == 11:
        return 70 + minor
    if major == 10:
        return 63 + minor
    raise RuntimeError("Triton only support CUDA 10.0 or higher")


def path_to_ptxas():
    base_dir = os.path.join(os.path.dirname(__file__), os.pardir)
    paths = [
        os.environ.get("TRITON_PTXAS_PATH", ""),
        os.path.join(base_dir, "third_party", "cuda", "bin", "ptxas")
    ]

    for ptxas in paths:
        if os.path.exists(ptxas) and os.path.isfile(ptxas):
            result = subprocess.check_output([ptxas, "--version"], stderr=subprocess.STDOUT)
            if result is not None:
                version = re.search(r".*release (\d+\.\d+).*", result.decode("utf-8"), flags=re.MULTILINE)
                if version is not None:
                    return ptxas, version.group(1)
    raise RuntimeError("Cannot find ptxas")


def llir_to_ptx(mod: Any, arch: int, ptx_version: int = None) -> str:
    '''
    Translate TritonGPU module to PTX code.
    :param mod: a TritonGPU dialect module
    :return: PTX code
    '''
    if ptx_version is None:
        _, cuda_version = path_to_ptxas()
        ptx_version = ptx_get_version(cuda_version)
    return _triton.translate_llvmir_to_ptx(mod, arch, ptx_version)


def ptx_to_cubin(ptx: str, arch: int):
    '''
    Compile TritonGPU module to cubin.
    :param ptx: ptx code
    :param compute_capability: compute capability
    :return: str
    '''
    ptxas, _ = path_to_ptxas()
    return _triton.compile_ptx_to_cubin(ptx, ptxas, arch)


# AMDGCN translation

def get_amdgcn_bitcode_paths(arch):
    gpu_arch_agnostic_bitcode_libraries = ["opencl.bc",
                                           "ocml.bc",
                                           "ockl.bc",
                                           "oclc_finite_only_off.bc",
                                           "oclc_daz_opt_off.bc",
                                           "oclc_correctly_rounded_sqrt_on.bc",
                                           "oclc_unsafe_math_off.bc",
                                           "oclc_wavefrontsize64_on.bc"]

    gfx_arch = arch[1]
    gfx_arch_id = re.search('gfx(\\w+)', gfx_arch).group(1).strip()

    gpu_arch_specific_bitcode_library = 'oclc_isa_version_' + gfx_arch_id + ".bc"
    bitcode_path_dir = os.path.join(Path(__file__).parent.resolve(), "third_party/rocm/lib/bitcode/")

    amdgcn_bitcode_paths = {}
    i = 1
    for bc_lib in gpu_arch_agnostic_bitcode_libraries:
        bc_path = bitcode_path_dir + bc_lib
        if os.path.exists(bc_path):
            amdgcn_bitcode_paths['library_' + str(i)] = bc_path
            i += 1
    bc_gfx_path = bitcode_path_dir + gpu_arch_specific_bitcode_library
    if os.path.exists(bc_gfx_path):
        amdgcn_bitcode_paths['library_' + str(i)] = bc_gfx_path

    return amdgcn_bitcode_paths


def get_amdgpu_arch_fulldetails():
    """
    get the amdgpu fulll ISA details for compiling:
    i.e., arch_triple: amdgcn-amd-amdhsa; arch_name: gfx906; arch_features: sramecc+:xnack-
    """
    try:
        # TODO: package rocm.cc with Triton
        rocm_path_dir = os.getenv("ROCM_PATH", default="/opt/rocm")
        rocminfo = subprocess.check_output(rocm_path_dir + '/bin/rocminfo').decode()
        gfx_arch_details = re.search('amd.*', rocminfo).group(0).strip().split('--')
        arch_triple = gfx_arch_details[0]
        arch_name_features = gfx_arch_details[1].split(':')
        arch_name = arch_name_features[0]
        arch_features = ""

        if (len(arch_name_features) == 3):
            arch_features = "+" + re.search('\\w+', arch_name_features[1]).group(0) + ","\
                            "-" + re.search('\\w+', arch_name_features[2]).group(0)
        return [arch_triple, arch_name, arch_features]
    except BaseException:
        return None


def llir_to_amdgcn_and_hsaco(mod: Any, gfx_arch: str, gfx_triple: str, gfx_features: str) -> Tuple[str, str]:
    '''
    Translate TritonGPU module to HSACO code based on full details of gpu architecture.
    :param mod: a TritonGPU dialect module
    :return:
        - AMDGCN code
        - Path to HSACO object
    '''
    return _triton.translate_llvmir_to_hsaco(mod, gfx_arch, gfx_triple, gfx_features)


# ------------------------------------------------------------------------------
# compiler
# ------------------------------------------------------------------------------
def get_kernel_name(src: str, pattern: str) -> str:
    '''
    Get kernel name from PTX code.
    This Kernel name is required when launching the kernel.
    '''
    # There is a name mangling in PTX codegen, so the original kernel names in Triton IR are not available in PTX/cubin.
    assert src
    for line in src.split('\n'):
        line = line.strip()
        if line.startswith(pattern):
            return line.split()[-1]


def convert_type_repr(x):
    match = re.search(r'!tt\.ptr<(.*)>', x)
    if match is not None:
        return '*' + convert_type_repr(match.group(1))
    return x


def make_hash(fn, **kwargs):
    if isinstance(fn, triton.runtime.JITFunction):
        configs = kwargs["configs"]
        signature = kwargs["signature"]
        constants = kwargs.get("constants", dict())
        num_warps = kwargs.get("num_warps", 4)
        num_stages = kwargs.get("num_stages", 3)
        debug = kwargs.get("debug", False)
        # Get unique key for the compiled code
        get_conf_key = lambda conf: (sorted(conf.divisible_by_16), sorted(conf.equal_to_1))
        configs_key = [get_conf_key(conf) for conf in configs]
        key = f"{fn.cache_key}-{''.join(signature.values())}-{configs_key}-{constants}-{num_warps}-{num_stages}-{debug}"
        return hashlib.md5(key.encode("utf-8")).hexdigest()
    assert isinstance(fn, str)
    return hashlib.md5((Path(fn).read_text() + triton.runtime.jit.version_key()).encode("utf-8")).hexdigest()


# - ^\s*tt\.func\s+ : match the start of the string, any leading whitespace, the keyword func,
#    and any following whitespace
# - (public\s+)? : optionally match the keyword public and any following whitespace
# - (@\w+) : match an @ symbol followed by one or more word characters
#   (letters, digits, or underscores), and capture it as group 1 (the function name)
# - (\((?:%\w+: \S+(?: \{\S+ = \S+ : \S+\})?(?:, )?)*\)) : match a pair of parentheses enclosing
#   zero or more arguments separated by commas, and capture it as group 2 (the argument list)
mlir_prototype_pattern = r'^\s*tt\.func\s+(?:public\s+)?(@\w+)(\((?:%\w+: \S+(?: \{\S+ = \S+ : \S+\})?(?:, )?)*\))\s*\{\s*$'
ptx_prototype_pattern = r"\.(?:visible|extern)\s+\.(?:entry|func)\s+(\w+)\s*\(([^)]*)\)"
prototype_pattern = {
    "ttir": mlir_prototype_pattern,
    "ttgir": mlir_prototype_pattern,
    "ptx": ptx_prototype_pattern,
}

mlir_arg_type_pattern = r'%\w+: ([^,^\)\s]+)(?: \{\S+ = \S+ : \S+\})?,?'
ptx_arg_type_pattern = r"\.param\s+\.(\w+)"
arg_type_pattern = {
    "ttir": mlir_arg_type_pattern,
    "ttgir": mlir_arg_type_pattern,
    "ptx": ptx_arg_type_pattern,
}


def _get_jsonable_constants(constants):
    def _is_jsonable(x):
        try:
            json.dumps(x)
            return True
        except (TypeError, OverflowError):
            return False
    serialized_constants = {}
    for constant in constants:
        if _is_jsonable(constants[constant]):
            serialized_constants[constant] = constants[constant]
    return serialized_constants


def parse_mlir_module(path, context):
    module = _triton.ir.parse_mlir_module(path, context)
    # module takes ownership of the context
    module.context = context
    return module


instance_descriptor = namedtuple("instance_descriptor", ["divisible_by_16", "equal_to_1"], defaults=[set(), set()])


# TODO: architecture descriptor class
def _is_cuda(arch):
    return isinstance(arch, int)


def get_architecture_descriptor(capability):
    if capability is None:
        if torch.version.hip is None:
            device = triton.runtime.jit.get_current_device()
            capability = triton.runtime.jit.get_device_capability(device)
            capability = capability[0] * 10 + capability[1]
        else:
            capability = get_amdgpu_arch_fulldetails()
    return capability


def add_rocm_stages(arch, extern_libs, stages):
    extern_libs.update(get_amdgcn_bitcode_paths(arch))

    for key in list(extern_libs):
        if extern_libs[key] == '' or extern_libs[key] is None:
            extern_libs.pop(key)

    gfx_arch_full_details = arch
    gfx_arch = os.environ.get('MI_GPU_ARCH', gfx_arch_full_details[1])
    if gfx_arch is None:
        raise RuntimeError('gfx_arch is None (not specified)')
    stages["amdgcn"] = (lambda path: Path(path).read_text(),
                        lambda src: llir_to_amdgcn_and_hsaco(src, gfx_arch,
                                                             gfx_arch_full_details[0],
                                                             gfx_arch_full_details[2]))


def add_cuda_stages(arch, extern_libs, stages):

    stages["ptx"] = (lambda path: Path(path).read_text(),
                     lambda src: llir_to_ptx(src, arch))
    stages["cubin"] = (lambda path: Path(path).read_bytes(),
                       lambda src: ptx_to_cubin(src, arch))


def compile(fn, **kwargs):
    arch = get_architecture_descriptor(kwargs.get("cc", None))
    is_cuda = _is_cuda(arch)
    context = _triton.ir.context()
    asm = dict()
    constants = kwargs.get("constants", dict())
    num_warps = kwargs.get("num_warps", 4)
    num_stages = kwargs.get("num_stages", 3 if is_cuda and arch >= 75 else 2)
    extern_libs = kwargs.get("extern_libs", dict())
    if extern_libs is None:
        extern_libs = dict()
    debug = kwargs.get("debug", False)
    # build compilation stages
    stages = dict()
    stages["ast"] = (lambda path: fn, None)
    stages["ttir"] = (lambda path: parse_mlir_module(path, context),
                      lambda src: optimize_ttir(ast_to_ttir(src, signature, configs[0], constants, debug=debug), arch))
    stages["ttgir"] = (lambda path: parse_mlir_module(path, context),
                       lambda src: optimize_ttgir(ttir_to_ttgir(src, num_warps), num_stages, arch))
    stages["llir"] = (lambda path: Path(path).read_text(),
                      lambda src: ttgir_to_llir(src, extern_libs, arch))
    if is_cuda:
        add_cuda_stages(arch, extern_libs, stages)
    else:
        add_rocm_stages(arch, extern_libs, stages)

    # find out the signature of the function
    if isinstance(fn, triton.runtime.JITFunction):
        configs = kwargs.get("configs", None)
        signature = kwargs["signature"]
        if configs is None:
            configs = [instance_descriptor()]
        assert len(configs) == 1
        kwargs["configs"] = configs
        name = fn.__name__
        first_stage = 0
        if isinstance(signature, str):
            signature = {k: v.strip() for k, v in enumerate(signature.split(","))}
        kwargs["signature"] = signature
    else:
        assert isinstance(fn, str)
        _, ir = os.path.basename(fn).split(".")
        src = Path(fn).read_text()
        import re
        match = re.search(prototype_pattern[ir], src, re.MULTILINE)
        name, signature = match.group(1), match.group(2)
        types = re.findall(arg_type_pattern[ir], signature)
        param_tys = [convert_type_repr(ty) for ty in types]
        signature = {k: v for k, v in enumerate(param_tys)}
        first_stage = list(stages.keys()).index(ir)

    # cache manager
    so_path = make_stub(name, signature, constants)
    # create cache manager
    fn_cache_manager = get_cache_manager(make_hash(fn, **kwargs))
    # determine name and extension type of provided function
    if isinstance(fn, triton.runtime.JITFunction):
        name, ext = fn.__name__, "ast"
    else:
        name, ext = os.path.basename(fn).split(".")

    # load metadata if any
    metadata = None
    metadata_filename = f"{name}.json"

    # The group is addressed by the metadata
    metadata_group = fn_cache_manager.get_group(
        metadata_filename
    ) or {}

    metadata_path = metadata_group.get(metadata_filename)

    if metadata_path is not None:
        with open(metadata_path) as f:
            metadata = json.load(f)
    else:
        metadata = {"num_warps": num_warps,
                    "num_stages": num_stages,
                    "constants": _get_jsonable_constants(constants),
                    "debug": debug}
        if ext == "ptx":
            assert "shared" in kwargs, "ptx compilation must provide shared memory size"
            metadata["shared"] = kwargs["shared"]

    first_stage = list(stages.keys()).index(ext)
    asm = dict()
    module = fn
    # run compilation pipeline  and populate metadata
    for ir, (parse, compile_kernel) in list(stages.items())[first_stage:]:
        ir_filename = f"{name}.{ir}"

        if ir == ext:
            next_module = parse(fn)
        else:
            path = metadata_group.get(ir_filename)
            if path is None:
                next_module = compile_kernel(module)
                if ir == "amdgcn":
                    extra_file_name = f"{name}.hsaco_path"
                    metadata_group[ir_filename] = fn_cache_manager.put(next_module[0], ir_filename)
                    metadata_group[extra_file_name] = fn_cache_manager.put(next_module[1], extra_file_name)
                else:
                    metadata_group[ir_filename] = fn_cache_manager.put(next_module, ir_filename)
                    fn_cache_manager.put(next_module, ir_filename)
            else:
                if ir == "amdgcn":
                    extra_file_name = f"{name}.hsaco_path"
                    hasco_path = metadata_group.get(extra_file_name)
                    assert hasco_path is not None, "Expected to have hsaco_path in metadata when we have the amdgcn"
                    next_module = (parse(path), parse(hasco_path))
                else:
                    next_module = parse(path)

        if ir == "cubin":
            asm[ir] = next_module
        elif ir == "amdgcn":
            asm[ir] = str(next_module[0])
        else:
            asm[ir] = str(next_module)
        if ir == "llir" and "shared" not in metadata:
            metadata["shared"] = _triton.get_shared_memory_size(module)
        if ir == "ptx":
            metadata["name"] = get_kernel_name(next_module, pattern='// .globl')
        if ir == "amdgcn":
            metadata["name"] = get_kernel_name(next_module[0], pattern='.globl')
            asm["hsaco_path"] = next_module[1]
        module = next_module
    # write-back metadata, if it didn't come from the cache
    if metadata_path is None:
        metadata_group[metadata_filename] = fn_cache_manager.put(json.dumps(metadata), metadata_filename, binary=False)
        fn_cache_manager.put_group(metadata_filename, metadata_group)

    # return handle to compiled kernel
    return CompiledKernel(fn, so_path, metadata, asm)


class CompiledKernel:

    # Hooks for external tools to monitor the execution of triton kernels
    launch_enter_hook = None
    launch_exit_hook = None

    def __init__(self, fn, so_path, metadata, asm):
        # initialize launcher
        import importlib.util
        spec = importlib.util.spec_from_file_location("__triton_launcher", so_path)
        mod = importlib.util.module_from_spec(spec)
        self.fn = fn
        spec.loader.exec_module(mod)
        self.c_wrapper = getattr(mod, "launch")
        # initialize metadata
        self.shared = metadata["shared"]
        self.num_warps = metadata["num_warps"]
        self.num_stages = metadata["num_stages"]
        self.constants = metadata["constants"]
        # initialize asm dict
        self.asm = asm
        # binaries are lazily initialized
        # because it involves doing runtime things
        # (e.g., checking amount of shared memory on current device)
        self.metadata = metadata
        self.cu_module = None
        self.cu_function = None
        self.is_hip = "amdgcn" in asm

    def _init_handles(self):
        if self.cu_module is not None:
            return
        device = triton.runtime.jit.get_current_device()
        if self.is_hip:
            hip_utils = get_hip_utils()
            max_shared = hip_utils.get_device_properties(device)["max_shared_mem"]
            if self.shared > max_shared:
                raise OutOfResources(self.shared, max_shared, "shared memory")
            mod, func, n_regs, n_spills = hip_utils.load_binary(self.metadata["name"], self.asm["hsaco_path"], self.shared, device)
        else:
            cuda_utils = get_cuda_utils()
            max_shared = cuda_utils.get_device_properties(device)["max_shared_mem"]
            if self.shared > max_shared:
                raise OutOfResources(self.shared, max_shared, "shared memory")
            mod, func, n_regs, n_spills = cuda_utils.load_binary(self.metadata["name"], self.asm["cubin"], self.shared, device)

        self.n_spills = n_spills
        self.n_regs = n_regs
        self.cu_module = mod
        self.cu_function = func

    def __getattribute__(self, name):
        if name == 'c_wrapper':
            self._init_handles()
        return super().__getattribute__(name)

    def __getitem__(self, grid):
        self._init_handles()

        def runner(*args, stream=None):
            if stream is None:
                stream = triton.runtime.jit.get_cuda_stream()
            self.c_wrapper(grid[0], grid[1], grid[2], self.num_warps, self.shared, stream, self.cu_function,
                           CompiledKernel.launch_enter_hook, CompiledKernel.launch_exit_hook, self, *args)
        return runner

    def get_sass(self, fun=None):
        if 'sass' in self.asm:
            return self.asm['sass']
        fd, path = tempfile.mkstemp()
        try:
            with open(fd, 'wb') as cubin:
                cubin.write(self.asm['cubin'])
            self.sass = extract(path, fun)
        finally:
            os.remove(path)
        self.asm['sass'] = self.sass
        return self.sass
