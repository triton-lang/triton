from triton.backends.compiler import BaseBackend, GPUTarget
from triton._C.libtriton import ir, passes, llvm, iluvatar

from dataclasses import dataclass
import functools
from typing import Any, Tuple, Optional
import hashlib
import re
import tempfile
import signal
import os
import subprocess
from pathlib import Path
from triton.backends.iluvatar.driver import cuda_home_dirs


@functools.lru_cache(None)
def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


@dataclass(frozen=False)
class CUDAOptions:
    num_warps: int = 4
    num_ctas: int = 1
    num_stages: int = 3
    # maxnreg corresponds to the ptx parameter .maxnreg, which controls the
    # maximum number of 32-bit registers used by one thread.
    maxnreg: Optional[int] = None
    cluster_dims: tuple = (1, 1, 1)
    enable_fp_fusion: bool = True
    allow_fp8e4nv: bool = False
    allow_fp8e4b15: bool = False
    default_dot_input_precision: str = "tf32"
    allowed_dot_input_precisions: Tuple[str] = ("tf32", "tf32x3", "ieee")
    max_num_imprecise_acc_default: bool = None
    extern_libs: dict = None
    debug: bool = False
    backend_name: str = 'cuda'
    use_sme: int = 0
    enable_sme: bool = True
    num_vgpr: int = 0

    def __post_init__(self):
        default_libdir = cuda_home_dirs() + "/nvvm/libdevice/"
        extern_libs = {} if self.extern_libs is None else dict(self.extern_libs)
        if not extern_libs.get('libdevice', None):
            extern_libs['libdevice'] = os.getenv("TRITON_LIBDEVICE_PATH",
                                                 str(default_libdir + 'libdevice.compute_bi.10.bc'))
        object.__setattr__(self, 'extern_libs', tuple(extern_libs.items()))
        assert self.num_warps > 0 and (self.num_warps & (self.num_warps - 1)) == 0, \
               "num_warps must be a power of 2"

    def hash(self):
        hash_dict = dict(self.__dict__)
        hash_dict["extern_libs"] = tuple((k, file_hash(v)) for k, v in sorted(hash_dict["extern_libs"]))
        key = "_".join([f"{name}-{val}" for name, val in sorted(hash_dict.items())])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


class CUDABackend(BaseBackend):

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == 'cuda'

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        self.capability = target.arch
        assert isinstance(self.capability, int)
        self.binary_ext = "cubin"

    def parse_options(self, opts) -> Any:
        args = {k: opts[k] for k in CUDAOptions.__dataclass_fields__.keys() if k in opts}
        # args["allow_fp8e4nv"] = self.capability >= 89
        # args["allow_fp8e4b15"] = self.capability < 90
        args["allow_fp8e4nv"] = False
        args["allow_fp8e4b15"] = False
        args["max_num_imprecise_acc_default"] = 2**30 if self.capability == 90 else 0
        return CUDAOptions(**args)

    def pack_metadata(self, metadata):
        return (
            metadata.num_warps,
            metadata.num_ctas,
            metadata.shared,
            metadata.cluster_dims[0],
            metadata.cluster_dims[1],
            metadata.cluster_dims[2],
        )

    def get_codegen_implementation(self):
        codegen_fns = dict()
        return codegen_fns

    def load_dialects(self, ctx):
        iluvatar.load_dialects(ctx)

    @staticmethod
    def make_ttir(mod, metadata, opt):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.ttir.add_rewrite_tensor_pointer(pm)
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
        # TTIR -> TTGIR
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.ttir.add_convert_to_ttgpuir(pm, f"cuda:{capability}", opt.num_warps, 64, opt.num_ctas)
        # optimize TTGIR
        passes.ttgpuir.add_coalesce(pm)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_optimize_thread_locality(pm)
        iluvatar.passes.ttgpuir.add_accelerate_matmul(pm, capability, opt.use_sme)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_optimize_dot_operands(pm, True)
        passes.common.add_cse(pm)
        iluvatar.passes.ttgpuir.add_matmul_load(pm, capability)  # only MR(71) support sme
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_optimize_dot_operands(pm, True)
        passes.common.add_cse(pm)
        passes.ttgpuir.add_pipeline(pm, opt.num_stages)
        passes.ttgpuir.add_prefetch(pm)
        passes.ttgpuir.add_optimize_dot_operands(pm, True)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        iluvatar.passes.ttgpuir.add_matmul_mmastore(pm, capability)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        iluvatar.passes.ttgpuir.add_mmareduce(pm, capability)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_reduce_data_duplication(pm)
        passes.ttgpuir.add_reorder_instructions(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        passes.common.add_canonicalizer(pm)
        pm.run(mod)
        return mod

    @staticmethod
    def make_llir(src, metadata, options, capability):
        mod = src
        # TritonGPU -> LLVM-IR (MLIR)
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        iluvatar.passes.ttgpuir.add_decompose_unsupported_conversions(pm)
        passes.convert.add_scf_to_cf(pm)
        passes.convert.add_index_to_llvmir(pm)
        passes.ttgpuir.add_allocate_shared_memory(pm)
        iluvatar.passes.ttgpuir.add_to_llvmir(pm, capability)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)

        passes.convert.add_scf_to_cf(pm)
        passes.convert.add_cf_to_llvmir(pm)
        passes.convert.add_arith_to_llvmir(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        if os.environ.get("TRITON_DISABLE_LINE_INFO", "0") == "0":
            passes.llvmir.add_di_scope(pm)
        pm.run(mod)
        # LLVM-IR (MLIR) -> LLVM-IR (LLVM)
        llvm.init_targets()
        context = llvm.context()
        llvm_mod = llvm.to_module(mod, context)
        iluvatar.set_nvvm_reflect_ftz(llvm_mod)

        # Set maxnreg on all kernels, if it was provided.
        if options.maxnreg is not None:
            for k in llvm_mod.get_functions():
                if not k.is_declaration() and k.is_external_linkage():
                    k.set_nvvm_maxnreg(options.maxnreg)

        # Set kernel attributes first given this may affect later optimizations.
        fns = [fn for fn in llvm_mod.get_functions() if not fn.is_declaration()]
        # The public kernel should be kernel 0.
        fns[0].set_calling_conv(iluvatar.CALLING_CONV_ILUVATAR_KERNEL)
        if (options.num_vgpr > 0):
            fns[0].add_fn_attr("iluvatar-num-vgpr", f"{options.num_vgpr}")

        if options.extern_libs:
            paths = [path for (name, path) in options.extern_libs]
            llvm.link_extern_libs(llvm_mod, paths)

        llvm.optimize_module(llvm_mod, llvm.OPTIMIZE_O3, iluvatar.TARGET_TRIPLE)

        # Get some metadata
        metadata["shared"] = src.get_int_attr("triton_gpu.shared")
        ret = str(llvm_mod)
        del llvm_mod
        del context
        return ret

    @staticmethod
    def make_cubin(src, metadata, options, capability):
        names = re.findall(r"define iluvatar_kernel void @([a-zA-Z_][a-zA-Z0-9_]*)", src)
        assert len(names) == 1
        metadata["name"] = names[0]

        triple = "bi-iluvatar-ilurt"
        proc = "ivcore11"
        if capability == 70:
            proc = "ivcore10"
        elif capability == 71:
            proc = "ivcore11"
        elif capability == 80:
            proc = "ivcore20"
        else:
            print("iluvatar not support current compute capability", capability)
        cubin = iluvatar.translate_llvmir_to_cubin(src, triple, proc, '', [], options.enable_fp_fusion, False)
        return cubin

    def add_stages(self, stages, options):
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["ttgir"] = lambda src, metadata: self.make_ttgir(src, metadata, options, self.capability)
        stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options, self.capability)
        stages["cubin"] = lambda src, metadata: self.make_cubin(src, metadata, options, self.capability)

    @functools.lru_cache()
    def hash(self):
        version = ''
        return f'{version}-{self.capability}'
