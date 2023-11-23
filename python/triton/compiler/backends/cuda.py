from triton.common.backend import BaseBackend
from pathlib import Path
from dataclasses import dataclass
from ..._C.libtriton.triton import ClusterInfo, get_num_warps, TMAInfos, translate_triton_gpu_to_llvmir, get_shared_memory_size, translate_llvmir_to_ptx, compile_ptx_to_cubin, add_external_libs
from ...common.backend import get_cuda_version_key, path_to_ptxas
from ..._C.libtriton.triton import ir, runtime
import functools
from typing import Any
from ...runtime.jit import JITFunction
from ..utils import get_ids_of_tensormaps, parse_tma_info
from ..make_launcher import make_stub
from ...tools.disasm import get_sass
import hashlib


def ttir_to_ttgir(mod, num_warps, num_ctas, capability):
    pm = ir.pass_manager(mod.context)
    pm.enable_debug()
    pm.add_convert_triton_to_tritongpu_pass(num_warps, 32, num_ctas, capability)
    pm.run(mod)
    return mod


def parse_mlir_module(path, context):
    module = ir.parse_mlir_module(path, context)
    # module takes ownership of the context
    module.context = context
    return module


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


def optimize_ttgir(mod, num_stages, num_warps, num_ctas, capability, cluster_info, enable_warp_specialization,
                   enable_persistent, optimize_epilogue):
    pm = ir.pass_manager(mod.context)
    pm.enable_debug()
    pm.add_tritongpu_coalesce_pass()
    # TODO(Qingyi): Move PlanCTAPass to the front of CoalescePass
    pm.add_plan_cta_pass(cluster_info)
    if capability // 10 < 9:
        pm.add_tritongpu_rewrite_tensor_pointer_pass()
    pm.add_plan_cta_pass(cluster_info)
    pm.add_tritongpu_remove_layout_conversions_pass()
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


def ttgir_to_llir(mod, extern_libs, capability, tma_infos):
    if extern_libs:
        _add_external_libs(mod, extern_libs)
    return translate_triton_gpu_to_llvmir(mod, capability, tma_infos, runtime.TARGET.NVVM)


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


def llir_to_ptx(mod: Any, enable_fp_fusion: bool, capability: int, ptx_version: int = None) -> str:
    '''
    Translate TritonGPU module to PTX code.
    :param mod: a TritonGPU dialect module
    :return: PTX code
    '''
    if ptx_version is None:
        _, cuda_version = path_to_ptxas()
        ptx_version = ptx_get_version(cuda_version)
    return translate_llvmir_to_ptx(mod, capability, ptx_version, enable_fp_fusion)


def ptx_to_cubin(ptx: str, capability: int, enable_fp_fusion: bool):
    '''
    Compile TritonGPU module to cubin.
    :param ptx: ptx code
    :param compute_capability: compute capability
    :return: str
    '''
    ptxas, _ = path_to_ptxas()
    return compile_ptx_to_cubin(ptx, ptxas, capability, enable_fp_fusion)


@dataclass
class CUDAOptions:
    num_warps: int = 4
    num_ctas: int = 1
    num_stages: int = 3
    cluster_dims: list = None
    enable_warp_specialization: bool = False
    enable_persistent: bool = False
    optimize_epilogue: bool = False
    enable_fp_fusion: bool = True
    extern_libs = None
    allow_fp8e4nv: bool = False
    max_num_imprecise_acc: bool = None

    debug: bool = False

    def hash(self):
        key = '-'.join([str(x) for x in self.__dict__])
        return hashlib.md5(key.encode("utf-8")).hexdigest()


class CUDABackend(BaseBackend):

    def __init__(self, device_type: tuple) -> None:
        super().__init__(device_type)
        self.capability = device_type[1]
        assert isinstance(self.capability, int)

    def parse_options(self, **opts) -> Any:
        options = CUDAOptions(**opts)
        options.allow_fp8e4nv = self.capability >= 89
        options.max_num_imprecise_acc = 0 if self.capability >= 89 else None
        return options

    def add_stages(self, extern_libs, stages, opt):

        context = ir.context()
        cluster_info = ClusterInfo()
        if opt.cluster_dims is not None:
            cluster_info.clusterDimX = opt.cluster_dims[0]
            cluster_info.clusterDimY = opt.cluster_dims[1]
            cluster_info.clusterDimZ = opt.cluster_dims[2]

        # TTIR -> TTGIR stage
        def create_ttgir(src):
            ttgir = ttir_to_ttgir(src, opt.num_warps, opt.num_ctas, self.capability)
            return optimize_ttgir(ttgir, opt.num_stages, opt.num_warps, opt.num_ctas, self.capability, cluster_info,
                                  opt.enable_warp_specialization, opt.enable_persistent, opt.optimize_epilogue)

        stages["ttgir"] = (lambda path: parse_mlir_module(path, context), create_ttgir)
        # TTGIR -> LLIR stage
        tma_infos = TMAInfos()

        def create_llir(src):
            return ttgir_to_llir(src, opt.extern_libs, self.capability, tma_infos)

        stages["llir"] = (lambda path: Path(path).read_text(), create_llir)

        # LLIR -> PTX stage
        def create_ptx(src):
            return llir_to_ptx(src, opt.enable_fp_fusion, self.capability)

        stages["ptx"] = (lambda path: Path(path).read_text(), create_ptx)

        # PTX -> CUBIN stage
        def create_cubin(src):
            return ptx_to_cubin(src, self.capability, opt.enable_fp_fusion)

        stages["cubin"] = (lambda path: Path(path).read_bytes(), create_cubin)
        self.tma_infos = tma_infos

    def add_meta_info(self, ir_name, cur_module, next_module, metadata, asm):
        if ir_name == "cubin":
            asm[ir_name] = next_module
            asm["sass"] = lambda: get_sass(next_module)
        if ir_name == "llir" and "shared" not in metadata:
            metadata["shared"] = get_shared_memory_size(cur_module)
        if ir_name == "ttgir":
            metadata["enable_warp_specialization"] = ir.is_ws_supported(next_module)
            metadata["num_warps"] = get_num_warps(next_module)
        if ir_name == "ptx":
            metadata["name"] = get_kernel_name(next_module, pattern='// .globl')

    def get_version_key(self):
        return f'{get_cuda_version_key()}-{self.capability}'

    def make_launcher_stub(self, fn, configs, metadata, name, signature, constants):
        ids_of_folded_args = tuple([int(k)
                                    for k in configs[0].ids_of_folded_args]) if isinstance(fn, JITFunction) else ()
        if "cluster_dims" not in metadata:
            metadata["cluster_dims"] = [1, 1, 1]
        if len(self.tma_infos) > 0:
            metadata["tensormaps_info"] = parse_tma_info(self.tma_infos, ids_of_folded_args)
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
        enable_warp_specialization = False

        return make_stub(name, signature, constants, ids, enable_warp_specialization=enable_warp_specialization)

    @classmethod
    def create_backend(cls, device_type: str):
        return cls(device_type)
