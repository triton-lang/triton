from triton.common.backend import BaseBackend
from ..._C.libtriton.triton import ir, runtime
from ..._C.libtriton.triton import get_num_warps, TMAInfos, translate_triton_gpu_to_llvmir, get_shared_memory_size, add_external_libs, translate_llvmir_to_hsaco
# from ..._C.libtriton.amd_triton import amd_ir
from dataclasses import dataclass
from ...common.backend import get_cuda_version_key
from typing import Any
import hashlib


@dataclass(frozen=True)
class HIPOptions:
    num_warps: int = 4
    num_stages: int = 3
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


class HIPBackend(BaseBackend):

    def __init__(self, device_type: tuple) -> None:
        super().__init__(device_type)
        self.capability = device_type[1]
        assert isinstance(self.capability, int)

    def parse_options(self, opts) -> Any:
        args = {k: opts[k] for k in HIPOptions.__dataclass_fields__.keys() if k in opts}
        return HIPOptions(**args)

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
    def make_ttgir(mod, metadata, opt):
        pm = ir.pass_manager(mod.context)
        pm.add_convert_triton_to_tritongpu_pass(opt.num_warps, 32, 1, 90)
        pm.run(mod)
        return mod

        # pm = amd_ir.pass_manager(mod.context)
        # pm.enable_debug()
        # pm.add_tritonamdgpu_coalesce_pass()
        # pm.add_tritonamdgpu_accelerate_matmul_pass(opt.matrix_core_version, opt.matrix_inst_shape)
        # pm.add_tritonamdgpu_remove_layout_conversions_pass()
        # pm.add_tritonamdgpu_optimize_epilogue_pass()
        # pm.add_tritonamdgpu_optimize_dot_operands_pass()
        # if opt.num_stages == 0 and opt.matrix_core_version != 0:
        #     pm.add_tritonamdgpu_stream_pipeline_pass()
        #     pm.add_canonicalizer_pass()
        # pm.add_tritonamdgpu_pipeline_pass(opt.num_stages, opt.num_warps, opt.num_ctas, 0)
        # pm.add_tritonamdgpu_materialize_load_store_pass(opt.num_warps, 0)
        # pm.add_tritonamdgpu_optimize_dot_operands_pass()
        # pm.add_tritonamdgpu_remove_layout_conversions_pass()
        # pm.add_tritonamdgpu_decompose_conversions_pass()
        # # do we even need the old pipeliner anymore?
        # if opt.num_stages != 0:
        #     pm.add_tritonamdgpu_reorder_instructions_pass()
        # pm.add_cse_pass()
        # pm.add_symbol_dce_pass()
        # pm.run(mod)
        # return mod

    @staticmethod
    def make_llir(src, metadata, options, capability):
        metadata["num_warps"] = get_num_warps(src)
        # link libraries
        if options.extern_libs:
            names = [lib[0] for lib in options.extern_libs]
            paths = [lib[1] for lib in options.extern_libs]
            add_external_libs(src, names, paths)
        # TritonGPU -> LLVM-IR
        ret = translate_triton_gpu_to_llvmir(src, capability, TMAInfos(), runtime.TARGET.ROCDL)
        metadata["shared"] = get_shared_memory_size(src)
        return ret

    @staticmethod
    def make_amdgcn(src, metadata, options):
        ret = translate_llvmir_to_hsaco(src, options.gfx_arch, options.archgfx_triple, options.gfx_features)
        return ret

    def add_stages(self, stages, options):
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["ttgir"] = lambda src, metadata: self.make_ttgir(src, metadata, options)
        stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options, 90)
        stages["amdgcn"] = lambda src, metadata: self.make_amdgcn(src, metadata, options)

    def hash(self):
        return f'{get_cuda_version_key()}-{self.capability}'

    def make_launcher_stub(self, src, metadata):
        assert False

    @classmethod
    def create_backend(cls, device_type: str):
        return cls(device_type)
