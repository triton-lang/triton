from triton.common.backend import BaseBackend
from dataclasses import dataclass
from ..._C.libtriton.translation import ClusterInfo, get_num_warps, TMAInfos, translate_triton_gpu_to_llvmir, get_shared_memory_size, translate_llvmir_to_asm, add_external_libs
from ...common.backend import get_cuda_version_key, path_to_ptxas
from ..._C.libtriton import ir, passes
import functools
from typing import Any
from ..utils import get_ids_of_tensormaps, parse_tma_info
from ..make_launcher import make_stub
import hashlib
import re
import tempfile
import signal
import os
import subprocess


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
        args["max_num_imprecise_acc_default"] = 2**30 if self.capability == 90 else 0
        return CUDAOptions(**args)

    @staticmethod
    def make_ttir(mod, metadata, opt):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.ttir.add_combine(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.common.add_licm(pm)
        passes.common.add_symbol_dce(pm)
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
        passes.ttir.add_convert_to_ttgpuir(pm, opt.num_warps, 32, opt.num_ctas, capability)
        # optimize TTGIR
        passes.ttgpuir.add_coalesce(pm)
        # TODO(Qingyi): Move PlanCTAPass to the front of CoalescePass
        passes.ttnvgpuir.add_plan_cta(pm, cluster_info)
        passes.ttnvgpuir.add_rewrite_tensor_pointer(pm, capability)
        passes.ttnvgpuir.add_plan_cta(pm, cluster_info)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_optimize_thread_locality(pm)
        passes.ttgpuir.add_accelerate_matmul(pm, capability)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        if opt.optimize_epilogue:
            passes.ttgpuir.add_optimize_epilogue(pm)
        passes.ttgpuir.add_optimize_dot_operands(pm)
        passes.common.add_cse(pm)
        # `num_warps` does not mean the total number of warps of a CTA when
        # warp specialization is enabled.
        # it's the responsibility of the compiler to figure out the exact
        # `num_warps` to use.
        # TODO: support the case where `num_warps` from user is not 4.
        ws_enabled = False
        if capability // 10 >= 9 and opt.enable_warp_specialization and opt.num_warps == 4:
            passes.ttnvgpuir.add_wsfeasibility_checking(pm, capability)
            pm.run(mod)
            ws_enabled = passes.ttnvgpuir.is_ws_supported(mod)
            pm = ir.pass_manager(mod.context)
            pm.enable_debug()
        if ws_enabled:
            passes.ttnvgpuir.add_wsdecomposing(pm, capability)
            passes.ttnvgpuir.add_wspipeline(pm, opt.num_stages, opt.num_warps, capability)
            passes.ttnvgpuir.add_wsmutex(pm, capability)
            passes.ttnvgpuir.add_wsmaterialization(pm, capability)
            passes.common.add_licm(pm)
            passes.common.add_cse(pm)
        else:
            passes.ttgpuir.add_pipeline(pm, opt.num_stages, opt.num_warps, opt.num_ctas, capability)
        passes.ttnvgpuir.add_materialize_load_store(pm, opt.num_warps, capability)
        if capability // 10 <= 8:
            passes.ttgpuir.add_prefetch(pm)
        passes.ttgpuir.add_optimize_dot_operands(pm)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_decompose_conversions(pm)
        passes.ttnvgpuir.add_wsfixup_missing_attrs(pm)
        passes.ttgpuir.add_reorder_instructions(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        if capability // 10 >= 9:
            passes.ttnvgpuir.add_fence_insertion(pm)
        passes.ttnvgpuir.add_wsfixup_missing_attrs(pm)
        passes.common.add_canonicalizer(pm)
        pm.run(mod)
        metadata["cluster_dims"] = (cluster_info.clusterDimX, cluster_info.clusterDimY, cluster_info.clusterDimZ)
        return mod

    @staticmethod
    def make_llir(src, metadata, options, capability):
        metadata["enable_warp_specialization"] = passes.ttnvgpuir.is_ws_supported(src)
        metadata["num_warps"] = get_num_warps(src)
        tma_infos = TMAInfos()
        # link libraries
        if options.extern_libs:
            names = [lib[0] for lib in options.extern_libs]
            paths = [lib[1] for lib in options.extern_libs]
            add_external_libs(src, names, paths)
        # TritonGPU -> LLVM-IR
        ret = translate_triton_gpu_to_llvmir(src, capability, tma_infos)
        if len(tma_infos) > 0:
            metadata["tensormaps_info"] = parse_tma_info(tma_infos, metadata["ids_of_folded_args"])
            for i, _ in enumerate(metadata["tensormaps_info"]):
                metadata["tensormaps_info"][i].ids_of_folded_args = metadata["ids_of_folded_args"]
        metadata["ids_of_tensormaps"] = get_ids_of_tensormaps(metadata.get("tensormaps_info", None))
        metadata["shared"] = get_shared_memory_size(src)
        return ret

    @staticmethod
    def make_ptx(src, metadata, opt, capability):
        proc = 'sm_90a' if capability == 90 else f'sm_{capability}'
        ret, name = translate_llvmir_to_asm(src, 'nvptx64-nvidia-cuda', proc, '', ['nvptx-short-ptr'],
                                            opt.enable_fp_fusion, False)
        metadata["name"] = name
        # post-process
        ptx_version = opt.ptx_version
        if ptx_version is None:
            _, cuda_version = path_to_ptxas()
            ptx_version = ptx_get_version(cuda_version)
        ptx_version = f'{ptx_version//10}.{ptx_version%10}'
        ret = re.sub(r'\.version \d+\.\d+', f'.version {ptx_version}', ret, flags=re.MULTILINE)
        return ret

    @staticmethod
    def make_cubin(src, metadata, opt, capability):
        ptxas, _ = path_to_ptxas()
        with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.ptx') as fsrc, \
            tempfile.NamedTemporaryFile(delete=False, mode='r', suffix='.log') as flog:
            fsrc.write(src)
            fsrc.flush()
            fbin = fsrc.name + '.o'

            line_info = '' if os.environ.get('TRITON_DISABLE_LINE_INFO') else ' -lineinfo'
            fmad = '' if opt.enable_fp_fusion else ' --fmad=false'
            suffix = 'a ' if capability == 90 else ' '
            cmd = f'{ptxas}{line_info}{fmad} -v --gpu-name=sm_{capability}{suffix}{fsrc.name} -o {fbin} 2> {flog.name}'

            try:
                subprocess.run(cmd, shell=True, check=True)
            except subprocess.CalledProcessError as e:
                with open(flog.name) as log_file:
                    log = log_file.read()
                if e.returncode == 255:
                    raise RuntimeError(f'Internal Triton PTX codegen error: \n{log}')
                elif e.returncode == 128 + signal.SIGSEGV:
                    raise RuntimeError(
                        f'Please run `ptxas {fsrc.name}` to confirm that this is a bug in `ptxas`\n{log}')
                else:
                    raise RuntimeError(f'`ptxas` failed with error code {e.returncode}: \n{log}')
            finally:
                if os.path.exists(fsrc.name):
                    os.remove(fsrc.name)
                if os.path.exists(flog.name):
                    os.remove(flog.name)

            with open(fbin, 'rb') as f:
                cubin = f.read()
            if os.path.exists(fbin):
                os.remove(fbin)
        return cubin

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
