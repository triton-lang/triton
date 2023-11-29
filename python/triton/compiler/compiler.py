from __future__ import annotations

import functools
import hashlib
import json
import os
import re
from collections import namedtuple
from pathlib import Path
from typing import Any

from dataclasses import dataclass

from .._C.libtriton.triton import (ClusterInfo, TMAInfos, add_external_libs, compile_ptx_to_cubin, get_env_vars,
                                   get_num_warps, get_shared_memory_size, ir, runtime, translate_llvmir_to_ptx,
                                   translate_triton_gpu_to_llvmir)
from ..common.backend import get_backend, get_cuda_version_key, path_to_ptxas
from ..common.build import is_hip
# from ..runtime import driver, jit, JITFunction
# TODO: runtime.errors
from ..runtime.autotuner import OutOfResources
from ..runtime.cache import get_cache_manager, get_dump_manager, get_override_manager
from ..runtime.driver import driver
from ..runtime.jit import (JITFunction, get_cuda_stream, get_current_device, get_device_capability)
from ..tools.disasm import get_sass
from .code_generator import ast_to_ttir
from .make_launcher import make_stub
from .utils import (InfoFromBackendForTensorMap, TensorMapManager, get_ids_of_tensormaps, parse_tma_info)


@dataclass
class CudaTargetDescriptor:
    capability: int
    num_warps: int
    enable_fp_fusion: bool


def _is_cuda(target):
    return isinstance(target, CudaTargetDescriptor)


class LazyDict(dict):

    def __getitem__(self, key):
        val = dict.__getitem__(self, key)
        if callable(val):
            return val()
        return val


def inline_triton_ir(mod):
    pm = ir.pass_manager(mod.context)
    pm.enable_debug()
    pm.add_inliner_pass()
    pm.run(mod)
    return mod


def ttir_compute_capability_rewrite(mod, target):
    # For hardware without support, we must rewrite all load/store
    # with block (tensor) pointers into tensors of pointers
    pm = ir.pass_manager(mod.context)
    pm.enable_debug()
    if _is_cuda(target):
        pm.add_rewrite_tensor_pointer_pass(target.capability)
    pm.run(mod)
    return mod


def optimize_ttir(mod, target):
    mod = inline_triton_ir(mod)
    mod = ttir_compute_capability_rewrite(mod, target)
    pm = ir.pass_manager(mod.context)
    pm.enable_debug()
    pm.add_inliner_pass()
    pm.add_triton_combine_pass()
    pm.add_canonicalizer_pass()
    pm.add_reorder_broadcast_pass()
    pm.add_cse_pass()
    pm.add_licm_pass()
    pm.add_symbol_dce_pass()
    pm.run(mod)
    return mod


def ttir_to_ttgir(mod, num_warps, num_ctas, target):
    pm = ir.pass_manager(mod.context)
    pm.enable_debug()
    pm.add_convert_triton_to_tritongpu_pass(num_warps, 32, num_ctas, target.capability)
    pm.run(mod)
    return mod


def optimize_ttgir(mod, num_stages, num_warps, num_ctas, target, cluster_info, enable_warp_specialization,
                   enable_persistent, optimize_epilogue):
    is_cuda = _is_cuda(target)
    if is_cuda:
        capability = target.capability
    pm = ir.pass_manager(mod.context)
    pm.enable_debug()
    pm.add_tritongpu_coalesce_pass()
    # TODO(Qingyi): Move PlanCTAPass to the front of CoalescePass
    pm.add_plan_cta_pass(cluster_info)
    if is_cuda:
        pm.add_tritongpu_rewrite_tensor_pointer_pass(capability)
        pm.add_plan_cta_pass(cluster_info)
    pm.add_tritongpu_remove_layout_conversions_pass()
    if is_cuda:
        pm.add_tritongpu_accelerate_matmul_pass(capability)
    pm.add_tritongpu_remove_layout_conversions_pass()
    if optimize_epilogue:
        pm.add_tritongpu_optimize_epilogue_pass()
    pm.add_tritongpu_optimize_dot_operands_pass()
    pm.add_cse_pass()
    ws_enabled = False
    # `num_warps` does not mean the total number of warps of a CTA when
    # warp specialization is enabled.
    # it's the responsibility of the compiler to figure out the exact
    # `num_warps` to use.
    # TODO: support the case where `num_warps` from user is not 4.
    if capability // 10 >= 9 and enable_warp_specialization and num_warps == 4:
        pm.add_tritongpu_ws_feasibility_checking_pass(capability)
        pm.run(mod)
        ws_enabled = ir.is_ws_supported(mod)
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
    if ws_enabled:
        pm.add_tritongpu_wsdecomposing_pass(capability)
        pm.add_tritongpu_wspipeline_pass(num_stages, num_warps, capability)
        pm.add_tritongpu_wsmutex_pass(capability)
        pm.add_tritongpu_wsmaterialization_pass(capability)
        pm.add_licm_pass()
        pm.add_cse_pass()
    else:
        pm.add_tritongpu_pipeline_pass(num_stages, num_warps, num_ctas, capability)
    pm.add_tritongpu_materialize_load_store_pass(num_warps, capability)
    if capability // 10 <= 8:
        pm.add_tritongpu_prefetch_pass()
    pm.add_tritongpu_optimize_dot_operands_pass()
    pm.add_tritongpu_remove_layout_conversions_pass()
    pm.add_tritongpu_decompose_conversions_pass()
    pm.add_tritongpu_ws_fixup_missing_attrs_pass()
    pm.add_tritongpu_reorder_instructions_pass()
    pm.add_cse_pass()
    pm.add_symbol_dce_pass()
    if capability // 10 >= 9:
        pm.add_tritongpu_fence_insertion_pass()
    pm.add_tritongpu_ws_fixup_missing_attrs_pass()
    pm.add_tritongpu_optimize_thread_locality_pass()
    pm.add_canonicalizer_pass()
    pm.run(mod)
    return mod


def _add_external_libs(mod, libs):
    for name, path in libs.items():
        if len(name) == 0 or len(path) == 0:
            return
    add_external_libs(mod, list(libs.keys()), list(libs.values()))


def ttgir_to_llir(mod, extern_libs, target, tma_infos):
    if extern_libs:
        _add_external_libs(mod, extern_libs)
    # TODO: separate tritongpu_to_llvmir for different backends
    if _is_cuda(target):
        return translate_triton_gpu_to_llvmir(mod, target.capability, tma_infos, runtime.TARGET.NVVM)
    else:
        return translate_triton_gpu_to_llvmir(mod, 0, TMAInfos(), runtime.TARGET.ROCDL)


# PTX translation


@functools.lru_cache()
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


def llir_to_ptx(mod: Any, target: CudaTargetDescriptor, ptx_version: int = None) -> str:
    '''
    Translate TritonGPU module to PTX code.
    :param mod: a TritonGPU dialect module
    :return: PTX code
    '''
    if ptx_version is None:
        _, cuda_version = path_to_ptxas()
        ptx_version = ptx_get_version(cuda_version)
    return translate_llvmir_to_ptx(mod, target.capability, ptx_version, target.enable_fp_fusion)


def ptx_to_cubin(ptx: str, target: CudaTargetDescriptor):
    '''
    Compile TritonGPU module to cubin.
    :param ptx: ptx code
    :param compute_capability: compute capability
    :return: str
    '''
    ptxas, _ = path_to_ptxas()
    return compile_ptx_to_cubin(ptx, ptxas, target.capability, target.enable_fp_fusion)


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
    # Currently we only capture the pointer type and assume the pointer is on global memory.
    # TODO: Capture and support shared memory space
    match = re.search(r'!tt\.ptr<([^,]+)', x)
    if match is not None:
        return '*' + convert_type_repr(match.group(1))
    return x


def make_hash(fn, target, env_vars, device_backend, **kwargs):
    if device_backend is None:
        version_key = get_cuda_version_key()
    else:
        version_key = device_backend.get_version_key()
    if isinstance(fn, JITFunction):
        configs = kwargs["configs"]
        signature = kwargs["signature"]
        constants = kwargs.get("constants", dict())
        num_warps = kwargs.get("num_warps", 4)
        num_ctas = kwargs.get("num_ctas", 1)
        num_stages = kwargs.get("num_stages", 3)
        enable_warp_specialization = kwargs.get("enable_warp_specialization", False)
        enable_persistent = kwargs.get("enable_persistent", False)
        debug = kwargs.get("debug", False)
        # Get unique key for the compiled code
        get_conf_key = lambda conf: (sorted(conf.divisible_by_16), sorted(conf.equal_to_1),
                                     sorted(conf.ids_of_folded_args), sorted(conf.divisible_by_8))
        configs_key = [get_conf_key(conf) for conf in configs]
        env_vars_list = [f"{env_vars[k]}" for k in sorted(env_vars.keys())]
        key = f"{fn.cache_key}-{version_key}-{''.join(signature.values())}-{configs_key}-{constants}-{num_warps}-{num_stages}-{num_ctas}-{num_stages}-{enable_warp_specialization}-{enable_persistent}-{debug}-{target}-{env_vars_list}"
        return hashlib.md5(key.encode("utf-8")).hexdigest()
    assert isinstance(fn, str)
    ignore_version = kwargs.get('ignore_version', False)
    if (ignore_version):
        return hashlib.md5((Path(fn).read_text()).encode("utf-8")).hexdigest()
    return hashlib.md5((Path(fn).read_text() + version_key).encode("utf-8")).hexdigest()


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


def _get_num_warps_from_ir_str(src: str):
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


def parse_mlir_module(path, context):
    module = ir.parse_mlir_module(path, context)
    # module takes ownership of the context
    module.context = context
    return module


instance_descriptor = namedtuple("instance_descriptor",
                                 ["divisible_by_16", "equal_to_1", "ids_of_folded_args", "divisible_by_8"],
                                 defaults=[set(), set(), set(), set()])


def get_cuda_capability(capability):
    if capability is None:
        device = get_current_device()
        capability = get_device_capability(device)
        capability = capability[0] * 10 + capability[1]
    return capability


def get_arch_default_num_warps(device_type):
    if device_type in ["cuda", "hip"]:
        num_warps = 4
    else:
        _device_backend = get_backend(device_type)
        assert _device_backend
        arch = _device_backend.get_architecture_descriptor()
        num_warps = arch["num_warps"]
    return num_warps


def get_arch_default_num_stages(device_type, capability=None):
    if device_type == "cuda":
        num_stages = 3 if get_cuda_capability(capability) >= 75 else 2
    else:
        _device_backend = get_backend(device_type)
        assert _device_backend
        arch = _device_backend.get_architecture_descriptor()
        num_stages = arch["num_stages"]

    return num_stages


def add_cuda_stages(target, extern_libs, stages):

    stages["ptx"] = (lambda path: Path(path).read_text(), lambda src: llir_to_ptx(src, target))
    stages["cubin"] = (lambda path: Path(path).read_bytes(), lambda src: ptx_to_cubin(src, target))


def compile(fn, **kwargs):
    # Get device type to decide which backend should be used
    device_type = kwargs.get("device_type", "cuda")
    capability = kwargs.get("cc", None)

    if is_hip():
        device_type = "hip"
    is_cuda = device_type == "cuda"
    if is_hip():
        is_cuda = False

    context = ir.context()
    constants = kwargs.get("constants", dict())
    num_warps = kwargs.get("num_warps", get_arch_default_num_warps(device_type))
    assert num_warps > 0 and (num_warps & (num_warps - 1)) == 0, "num_warps must be a power of 2"
    num_ctas = kwargs.get("num_ctas", 1)
    num_stages = kwargs.get("num_stages", get_arch_default_num_stages(device_type, capability=capability))
    enable_fp_fusion = kwargs.get("enable_fp_fusion", True)
    # TODO[shuhaoj]: Default should be to enable warp specialization once possible
    enable_warp_specialization = kwargs.get("enable_warp_specialization", False)
    # TODO[shuhaoj]: persistent can be decoupled with warp specialization
    enable_persistent = kwargs.get("enable_persistent", enable_warp_specialization)
    extern_libs = kwargs.get("extern_libs", dict())
    if extern_libs is None:
        extern_libs = dict()
    debug = kwargs.get("debug", False)
    # Flag to control whether to store mma layout directly
    optimize_epilogue = False
    if os.environ.get('OPTIMIZE_EPILOGUE', '') == '1':
        optimize_epilogue = True
    #
    cluster_info = ClusterInfo()
    if "clusterDims" in kwargs:
        cluster_info.clusterDimX = kwargs["clusterDims"][0]
        cluster_info.clusterDimY = kwargs["clusterDims"][1]
        cluster_info.clusterDimZ = kwargs["clusterDims"][2]
    tma_infos = TMAInfos()
    # build architecture descriptor
    if device_type == "cuda":
        _device_backend = get_backend(device_type)
        target = CudaTargetDescriptor(capability=get_cuda_capability(capability), num_warps=num_warps,
                                      enable_fp_fusion=enable_fp_fusion)
    else:
        _device_backend = get_backend(device_type)
        assert _device_backend
        target = _device_backend.get_architecture_descriptor(**kwargs)
    # build compilation stages
    stages = dict()
    stages["ast"] = (lambda path: fn, None)
    stages["ttir"] = (lambda path: parse_mlir_module(path, context), lambda src: optimize_ttir(
        ast_to_ttir(src, signature, configs[0], constants, debug=debug, target=target), target))
    if is_cuda:
        stages["ttgir"] = (lambda path: parse_mlir_module(path, context), lambda src: optimize_ttgir(
            ttir_to_ttgir(src, num_warps, num_ctas, target), num_stages, num_warps, num_ctas, target, cluster_info,
            enable_warp_specialization, enable_persistent, optimize_epilogue))
        stages["llir"] = (lambda path: Path(path).read_text(),
                          lambda src: ttgir_to_llir(src, extern_libs, target, tma_infos))
        add_cuda_stages(target, extern_libs, stages)
    elif device_type == "hip":
        _device_backend.add_stages(target, extern_libs, stages, num_warps=num_warps, num_stages=num_stages)
    else:
        # pass the user's configuration to the backend device.
        target["num_warps"] = num_warps
        target["num_stages"] = num_stages
        target["num_ctas"] = num_ctas
        _device_backend.add_stages(target, extern_libs, stages)

    # find out the signature of the function
    if isinstance(fn, JITFunction):
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
        _, ir_name = os.path.basename(fn).split(".")
        src = Path(fn).read_text()
        import re
        match = re.search(prototype_pattern[ir_name], src, re.MULTILINE)
        # TODO: support function attributes at group 3 (e.g., device function)
        name, signature = match.group(1), match.group(2)
        types = re.findall(arg_type_pattern[ir_name], signature)
        if ir_name == 'ttgir':
            num_warps_from_ir = _get_num_warps_from_ir_str(src)
            assert "num_warps" not in kwargs or num_warps_from_ir == num_warps, "num_warps in ttgir does not match num_warps in compile"
            num_warps = num_warps_from_ir

        param_tys = [convert_type_repr(ty) for ty in types]
        signature = {k: v for k, v in enumerate(param_tys)}
        first_stage = list(stages.keys()).index(ir_name)

    # create cache manager
    fn_cache_manager = get_cache_manager(make_hash(fn, target, get_env_vars(), _device_backend, **kwargs))
    # managers used to dump and override IR for debugging
    enable_override = os.environ.get("TRITON_KERNEL_OVERRIDE", "0") == "1"
    fn_override_manager = get_override_manager(
        make_hash(fn, target, get_env_vars(), _device_backend, **kwargs, ignore_version=True))
    fn_dump_manager = get_dump_manager(
        make_hash(fn, target, get_env_vars(), _device_backend, **kwargs, ignore_version=True))

    # determine name and extension type of provided function
    if isinstance(fn, JITFunction):
        name, ext = fn.__name__, "ast"
    else:
        name, ext = os.path.basename(fn).split(".")

    # load metadata if any
    metadata = None
    metadata_filename = f"{name}.json"

    # The group is addressed by the metadata
    metadata_group = fn_cache_manager.get_group(metadata_filename) or {}

    metadata_path = metadata_group.get(metadata_filename)

    if metadata_path is not None:
        with open(metadata_path) as f:
            metadata = json.load(f)
            if 'tensormaps_info' in metadata:
                metadata['tensormaps_info'] = [InfoFromBackendForTensorMap(e) for e in metadata['tensormaps_info']]
    else:
        metadata = {
            "num_warps": num_warps,
            "num_ctas": num_ctas,
            "num_stages": num_stages,
            "enable_warp_specialization": enable_warp_specialization,
            "enable_persistent": enable_persistent,
            "constants": _get_jsonable_constants(constants),
            "debug": debug,
            "target": target,
        }
        metadata.update(get_env_vars())
        if ext == "ptx":
            assert "shared" in kwargs, "ptx compilation must provide shared memory size"
            metadata["shared"] = kwargs["shared"]

    # Add device type to meta information
    metadata["device_type"] = device_type

    first_stage = list(stages.keys()).index(ext)
    asm = LazyDict()
    module = fn
    # run compilation pipeline  and populate metadata
    for ir_name, (parse, compile_kernel) in list(stages.items())[first_stage:]:
        ir_filename = f"{name}.{ir_name}"

        if ir_name == ext:
            next_module = parse(fn)
        else:
            path = metadata_group.get(ir_filename)
            if path is None:
                next_module = compile_kernel(module)
                if ir_name == "amdgcn":
                    extra_file_name = f"{name}.hsaco_path"
                    metadata_group[ir_filename] = fn_cache_manager.put(next_module[0], ir_filename)
                    metadata_group[extra_file_name] = fn_cache_manager.put(next_module[1], extra_file_name)
                else:
                    metadata_group[ir_filename] = fn_cache_manager.put(next_module, ir_filename)
                    fn_dump_manager.put(next_module, ir_filename)
                    if (enable_override and fn_override_manager.has_file(ir_filename)):
                        print(f"\nOverriding kernel with file {ir_filename}")
                        full_name = fn_override_manager.get_file(ir_filename)
                        next_module = parse(full_name)
            else:
                if ir_name == "amdgcn":
                    extra_file_name = f"{name}.hsaco_path"
                    hasco_path = metadata_group.get(extra_file_name)
                    assert hasco_path is not None, "Expected to have hsaco_path in metadata when we have the amdgcn"
                    next_module = (parse(path), parse(hasco_path))
                else:
                    next_module = parse(path)

        if ir_name == "cubin":
            asm[ir_name] = next_module
            asm["sass"] = lambda: get_sass(next_module)
        elif ir_name == "amdgcn":
            asm[ir_name] = str(next_module[0])
        else:
            asm[ir_name] = str(next_module)
        if ir_name == "llir" and "shared" not in metadata:
            if is_hip():
                metadata["shared"] = _device_backend.get_shared_memory_size(module)
            else:
                metadata["shared"] = get_shared_memory_size(module)
        if ir_name == "ttgir":
            if is_hip():
                metadata["num_warps"] = _device_backend.get_num_warps(next_module)
            else:
                metadata["enable_warp_specialization"] = ir.is_ws_supported(next_module)
                if metadata["enable_warp_specialization"]:
                    metadata["num_warps"] = get_num_warps(next_module)
        if ir_name == "ptx":
            metadata["name"] = get_kernel_name(next_module, pattern='// .globl')
        if ir_name == "amdgcn":
            metadata["name"] = get_kernel_name(next_module[0], pattern='.globl')
            asm["hsaco_path"] = next_module[1]
        if not is_cuda and not is_hip():
            _device_backend.add_meta_info(ir_name, module, next_module, metadata, asm)
        module = next_module

    ids_of_folded_args = tuple([int(k) for k in configs[0].ids_of_folded_args]) if isinstance(fn, JITFunction) else ()
    if "clusterDims" not in metadata:
        metadata["clusterDims"] = [cluster_info.clusterDimX, cluster_info.clusterDimY, cluster_info.clusterDimZ]

    if len(tma_infos) > 0:
        metadata["tensormaps_info"] = parse_tma_info(tma_infos, ids_of_folded_args)
    # set constant
    if "tensormaps_info" in metadata:
        for i, _ in enumerate(metadata["tensormaps_info"]):
            metadata["tensormaps_info"][i].ids_of_folded_args = ids_of_folded_args

    ids_of_tensormaps = get_ids_of_tensormaps(metadata.get("tensormaps_info", None))
    if isinstance(fn, JITFunction) and "tensormaps_info" in metadata:
        fn.tensormaps_info = metadata["tensormaps_info"]

    ids_of_const_exprs = tuple(fn.constexprs) if isinstance(fn, JITFunction) else ()
    ids = {
        "ids_of_tensormaps": ids_of_tensormaps, "ids_of_folded_args": ids_of_folded_args, "ids_of_const_exprs":
        ids_of_const_exprs
    }
    # cache manager
    if is_cuda:
        so_path = make_stub(name, signature, constants, ids, enable_warp_specialization=enable_warp_specialization)
    else:
        so_path = _device_backend.make_launcher_stub(name, signature, constants, ids)
    # write-back metadata, if it didn't come from the cache
    if metadata_path is None:
        metadata_group[metadata_filename] = fn_cache_manager.put(json.dumps(metadata, default=vars), metadata_filename,
                                                                 binary=False)
    fn_cache_manager.put_group(metadata_filename, metadata_group)

    # return handle to compiled kernel
    return CompiledKernel(fn, so_path, metadata, asm)


class CompiledKernel:

    # Hooks for external tools to monitor the execution of triton kernels
    launch_enter_hook = None
    launch_exit_hook = None
    tensormap_manager = TensorMapManager()

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
        if "threads_per_warp" in metadata:
            self.threads_per_warp = metadata["threads_per_warp"]
        self.num_ctas = metadata["num_ctas"]
        self.num_stages = metadata["num_stages"]
        self.clusterDims = metadata["clusterDims"]
        if "tensormaps_info" in metadata:
            self.tensormaps_info = metadata["tensormaps_info"]
        self.constants = metadata["constants"]
        self.device_type = metadata["device_type"]
        self.device_backend = get_backend(self.device_type) if self.device_type not in ["cuda"] else None
        # initialize asm dict
        self.asm = asm
        # binaries are lazily initialized
        # because it involves doing runtime things
        # (e.g., checking amount of shared memory on current device)
        self.metadata = metadata
        self.cu_module = None
        self.cu_function = None

    def _init_handles(self):
        if self.cu_module is not None:
            return

        if self.device_type in ["cuda"]:
            device = get_current_device()
            bin_path = {driver.HIP: "hsaco_path", driver.CUDA: "cubin"}[driver.backend]
            max_shared = driver.utils.get_device_properties(device)["max_shared_mem"]
            fn_load_binary = driver.utils.load_binary
        else:
            assert self.device_backend
            device = self.device_backend.get_current_device()
            bin_path = self.device_backend.get_kernel_bin()
            max_shared = self.device_backend.get_device_properties(device)["max_shared_mem"]
            fn_load_binary = self.device_backend.get_load_binary_fn()

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
                if self.device_type in ["cuda"]:
                    stream = get_cuda_stream()
                else:
                    stream = get_backend(self.device_type).get_stream(None)
            self.c_wrapper(grid[0], grid[1], grid[2], self.num_warps, self.num_ctas, self.clusterDims[0],
                           self.clusterDims[1], self.clusterDims[2], self.shared, stream, self.cu_function,
                           CompiledKernel.launch_enter_hook, CompiledKernel.launch_exit_hook, self, *args_expand)

        return runner
