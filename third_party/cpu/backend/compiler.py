import functools
import hashlib
import os

from dataclasses import dataclass
from typing import Any, Tuple

from triton._C.libtriton import cpu, ir, llvm, passes
from triton.backends.compiler import BaseBackend, GPUTarget


@dataclass(frozen=True)
class CPUOptions:
    # GPU-specific options are used in several places.
    # For now, we just provide dummy values.
    num_warps: int = 0
    num_stages: int = 0
    num_ctas: int = 0
    cluster_dims: tuple = (1, 1, 1)
    extern_libs: dict = None
    debug: bool = False
    allowed_dot_input_precisions: Tuple[str] = ("ieee", )
    allow_fp8e4nv: bool = False
    enable_fp_fusion: bool = True

    # TODO: We may introduce CPU-specific options like # of cores.

    def __post_init__(self):
        pass

    def hash(self):
        hash_dict = dict(self.__dict__)
        key = "_".join([f"{name}-{val}" for name, val in sorted(hash_dict.items())])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


class CPUBackend(BaseBackend):

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == "cpu"

    def __init__(self, target: tuple) -> None:
        super().__init__(target)
        self.binary_ext = "asm"

    def parse_options(self, opts) -> Any:
        args = {k: opts[k] for k in CPUOptions.__dataclass_fields__.keys() if k in opts}
        return CPUOptions(**args)

    def pack_metadata(self, metadata):
        return metadata

    def get_codegen_implementation(self):
        codegen_fns = dict()
        return codegen_fns

    def load_dialects(self, ctx):
        cpu.load_dialects(ctx)

    @staticmethod
    def make_ttir(mod, metadata, opt):
        # This is the same as the Nvidia backend.
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
    def make_ttcir(mod, metadata, opt):
        # TTIR -> TTCIR
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        cpu.passes.ttcpuir.add_triton_to_triton_cpu_pipeline(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        passes.common.add_canonicalizer(pm)
        pm.run(mod)
        metadata["cluster_dims"] = (opt.cluster_dims[0], opt.cluster_dims[1], opt.cluster_dims[2])
        return mod

    @staticmethod
    def make_llir(src, metadata, options):
        # warp-specialization mutates num_warps
        num_warp_groups = src.get_int_attr("triton_gpu.num-warp-groups-per-cta")
        if num_warp_groups is not None:
            metadata["num_warps"] *= num_warp_groups
        metadata["threads_per_warp"] = 1
        mod = src
        # TritonCPU -> LLVM-IR (MLIR)
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        cpu.passes.ttcpuir.add_lower_vector_multi_dim(pm)
        cpu.passes.ttcpuir.add_vector_to_scf(pm, True, 1, False)
        cpu.passes.ttcpuir.add_lower_affine(pm)
        passes.convert.add_scf_to_cf(pm)
        passes.convert.add_index_to_llvmir(pm)
        cpu.passes.ttcpuir.add_triton_cpu_to_llvmir_pipeline(pm)
        passes.convert.add_math_to_llvmir(pm)
        cpu.passes.ttcpuir.add_math_to_libm(pm)
        cpu.passes.ttcpuir.add_vector_to_llvmir(pm)
        cpu.passes.ttcpuir.add_memref_to_llvmir(pm)
        passes.convert.add_arith_to_llvmir(pm)
        cpu.passes.ttcpuir.add_func_to_llvmir(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        if os.environ.get("TRITON_DISABLE_LINE_INFO", "0") == "0":
            passes.llvmir.add_di_scope(pm)
        pm.run(mod)

        # Find kernel fn
        kernel_names = cpu.find_kernel_names(mod)
        assert len(kernel_names) == 1, f"expected exactly 1 kernel in a module, got {kernel_names}"

        # LLVM-IR (MLIR) -> LLVM-IR (LLVM)
        llvm.init_targets()
        context = llvm.context()
        llvm_mod = llvm.to_module(mod, context)
        llvm.set_host_target(llvm_mod)
        #if options.extern_libs:
        #    paths = [path for (name, path) in options.extern_libs]
        #   llvm.link_extern_libs(llvm_mod, paths)
        llvm.optimize_module(llvm_mod, llvm.OPTIMIZE_O3)
        # Get some metadata
        metadata["shared"] = 0
        metadata["name"] = kernel_names[0]
        ret = str(llvm_mod)
        del llvm_mod
        del context
        return ret

    @staticmethod
    def make_asm(src, metadata, options):
        return llvm.translate_to_host_asm(src, options.enable_fp_fusion)

    def add_stages(self, stages, options):
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["ttcir"] = lambda src, metadata: self.make_ttcir(src, metadata, options)
        stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options)
        stages["asm"] = lambda src, metadata: self.make_asm(src, metadata, options)

    @functools.lru_cache()
    def hash(self):
        # TODO: Get more detailed CPU info like raw brand name with supported ISAs.
        # Right now it would only return a simple string like "x86_64" or "aarch64".
        import platform

        return f"{platform.machine()}"
