from triton.common.backend import BaseBackend
from dataclasses import dataclass
from ..._C.libtriton.triton import ClusterInfo, get_num_warps, TMAInfos, translate_triton_gpu_to_llvmir, get_shared_memory_size, translate_llvmir_to_ptx, compile_ptx_to_cubin, add_external_libs
from ...common.backend import get_cuda_version_key, path_to_ptxas
from ..._C.libtriton.triton import ir, runtime
import functools
from typing import Any
from ..utils import get_ids_of_tensormaps, parse_tma_info
from ..make_launcher import make_stub
import hashlib


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


@dataclass(frozen=True)
class CUDAOptions:
    num_warps: int = 4
    num_ctas: int = 1
    num_stages: int = 3
    cluster_dims: tuple = (1, 1, 1)
    ptx_version: int = None
    enable_warp_specialization: bool = False
    enable_persistent: bool = False
    optimize_epilogue: bool = False
    enable_fp_fusion: bool = True
    allow_fp8e4nv: bool = False
    max_num_imprecise_acc_default: bool = None
    extern_libs: dict = None
    debug: bool = False

    def __post_init__(self):
        # TODO: change API
        if isinstance(self.extern_libs, dict):
            extern_libs = tuple([(k, v) for k, v in self.extern_libs.items() if v])
            object.__setattr__(self, 'extern_libs', extern_libs)
        assert self.num_warps > 0 and (self.num_warps & (self.num_warps - 1)) == 0, \
               "num_warps must be a power of 2"

    def hash(self):
        key = '_'.join([f'{name}-{val}' for name, val in self.__dict__.items()])
        return hashlib.md5(key.encode("utf-8")).hexdigest()


class CUDABackend(BaseBackend):

    def __init__(self, device_type: tuple) -> None:
        super().__init__(device_type)
        self.capability = device_type[1]
        assert isinstance(self.capability, int)

    def parse_options(self, opts) -> Any:
        args = {k: opts[k] for k in CUDAOptions.__dataclass_fields__.keys() if k in opts}
        args["allow_fp8e4nv"] = self.capability >= 89
        args["max_num_imprecise_acc_default"] = 0 if self.capability >= 89 else None
        return CUDAOptions(**args)

    @staticmethod
    def make_ttir(mod, metadata, opt):
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

    @staticmethod
    def make_ttgir(mod, metadata, opt, capability):
        cluster_info = ClusterInfo()
        if opt.cluster_dims is not None:
            cluster_info.clusterDimX = opt.cluster_dims[0]
            cluster_info.clusterDimY = opt.cluster_dims[1]
            cluster_info.clusterDimZ = opt.cluster_dims[2]
        # TTIR -> TTGIR
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        pm.add_convert_triton_to_tritongpu_pass(opt.num_warps, 32, opt.num_ctas, capability)
        # optimize TTGIR
        pm.add_tritongpu_coalesce_pass()
        # TODO(Qingyi): Move PlanCTAPass to the front of CoalescePass
        pm.add_plan_cta_pass(cluster_info)
        pm.add_tritongpu_rewrite_tensor_pointer_pass(capability)
        pm.add_plan_cta_pass(cluster_info)
        pm.add_tritongpu_remove_layout_conversions_pass()
        pm.add_tritongpu_accelerate_matmul_pass(capability)
        pm.add_tritongpu_remove_layout_conversions_pass()
        if opt.optimize_epilogue:
            pm.add_tritongpu_optimize_epilogue_pass()
        pm.add_tritongpu_optimize_dot_operands_pass()
        pm.add_cse_pass()
        ws_enabled = False
        # `num_warps` does not mean the total number of warps of a CTA when
        # warp specialization is enabled.
        # it's the responsibility of the compiler to figure out the exact
        # `num_warps` to use.
        # TODO: support the case where `num_warps` from user is not 4.
        if capability // 10 >= 9 and opt.enable_warp_specialization and opt.num_warps == 4:
            pm.add_tritongpu_ws_feasibility_checking_pass(capability)
            pm.run(mod)
            ws_enabled = ir.is_ws_supported(mod)
            pm = ir.pass_manager(mod.context)
            pm.enable_debug()
        if ws_enabled:
            pm.add_tritongpu_wsdecomposing_pass(capability)
            pm.add_tritongpu_wspipeline_pass(opt.num_stages, opt.num_warps, capability)
            pm.add_tritongpu_wsmutex_pass(capability)
            pm.add_tritongpu_wsmaterialization_pass(capability)
            pm.add_licm_pass()
            pm.add_cse_pass()
        else:
            pm.add_tritongpu_pipeline_pass(opt.num_stages, opt.num_warps, opt.num_ctas, capability)
        pm.add_tritongpu_materialize_load_store_pass(opt.num_warps, capability)
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
        metadata["cluster_dims"] = (cluster_info.clusterDimX, cluster_info.clusterDimY, cluster_info.clusterDimZ)
        return mod

    @staticmethod
    def make_llir(src, metadata, options, capability):
        metadata["enable_warp_specialization"] = ir.is_ws_supported(src)
        metadata["num_warps"] = get_num_warps(src)
        tma_infos = TMAInfos()
        # link libraries
        if options.extern_libs:
            names = [lib[0] for lib in options.extern_libs]
            paths = [lib[1] for lib in options.extern_libs]
            add_external_libs(src, names, paths)
        # TritonGPU -> LLVM-IR
        ret = translate_triton_gpu_to_llvmir(src, capability, tma_infos, runtime.TARGET.NVVM)
        if len(tma_infos) > 0:
            metadata["tensormaps_info"] = parse_tma_info(tma_infos, metadata["ids_of_folded_args"])
            for i, _ in enumerate(metadata["tensormaps_info"]):
                metadata["tensormaps_info"][i].ids_of_folded_args = metadata["ids_of_folded_args"]
        metadata["ids_of_tensormaps"] = get_ids_of_tensormaps(metadata.get("tensormaps_info", None))
        metadata["shared"] = get_shared_memory_size(src)
        return ret

    @staticmethod
    def make_ptx(src, metadata, opt, capability):
        ptx_version = opt.ptx_version
        if ptx_version is None:
            _, cuda_version = path_to_ptxas()
            ptx_version = ptx_get_version(cuda_version)
        return translate_llvmir_to_ptx(src, capability, ptx_version, opt.enable_fp_fusion)

    @staticmethod
    def make_cubin(src, metadata, opt, capability):
        metadata["name"] = get_kernel_name(src, pattern='// .globl')
        ptxas, _ = path_to_ptxas()
        return compile_ptx_to_cubin(src, ptxas, capability, opt.enable_fp_fusion)

    def add_stages(self, stages, options):
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["ttgir"] = lambda src, metadata: self.make_ttgir(src, metadata, options, self.capability)
        stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options, self.capability)
        stages["ptx"] = lambda src, metadata: self.make_ptx(src, metadata, options, self.capability)
        stages["cubin"] = lambda src, metadata: self.make_cubin(src, metadata, options, self.capability)

    def hash(self):
        return f'{get_cuda_version_key()}-{self.capability}'

    def make_launcher_stub(self, src, metadata):
        ids = {
            "ids_of_tensormaps": metadata.get("ids_of_tensormaps", tuple()), "ids_of_folded_args":
            metadata.get("ids_of_folded_args",
                         tuple()), "ids_of_const_exprs": src.fn.constexprs if hasattr(src, "fn") else tuple()
        }
        constants = src.constants if hasattr(src, "constants") else dict()
        enable_warp_specialization = False

        # set constant
        return make_stub(src.name, src.signature, constants, ids, enable_warp_specialization=enable_warp_specialization)

    @classmethod
    def create_backend(cls, device_type: str):
        return cls(device_type)
