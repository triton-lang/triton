from triton.backends.compiler import BaseBackend, GPUTarget
from triton._C.libtriton import ir, passes, llvm, mthreads

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
import shutil


def get_kernel_name(src: str, pattern: str) -> str:
    assert src
    for line in src.split('\n'):
        line = line.strip()
        if line.startswith(pattern):
            return line.split()[-1]


@functools.lru_cache()
def get_musa_version():
    version = subprocess.check_output(["/usr/local/musa/bin/musa_toolkits_version"]).decode("utf-8")
    return version


@functools.lru_cache(None)
def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


@dataclass(frozen=True)
class MUSAOptions:
    num_warps: int = 4
    num_ctas: int = 1
    num_stages: int = 3
    # maxnreg corresponds to the ptx parameter .maxnreg, which controls the
    # maximum number of 32-bit registers used by one thread.
    maxnreg: Optional[int] = None
    cluster_dims: tuple = (1, 1, 1)
    capability: int = None
    enable_fp_fusion: bool = True
    allow_fp8e4nv: bool = False
    allow_fp8e4b15: bool = False
    default_dot_input_precision: str = "tf32"
    allowed_dot_input_precisions: Tuple[str] = ("tf32", "tf32x3", "ieee")
    max_num_imprecise_acc_default: bool = None
    extern_libs: dict = None
    debug: bool = False
    backend_name: str = 'musa'

    def __post_init__(self):
        extern_libs = {} if self.extern_libs is None else dict(self.extern_libs)
        if not extern_libs.get('libdevice', None):
            if self.capability >= 31:
                default_libdir = "/usr/local/musa/mtgpu/bitcode/libdevice.31.bc"
            else:
                default_libdir = "/usr/local/musa/mtgpu/bitcode/libdevice.bc"
            # here we add an new ENV: MUSA_LIBDEVICE_PATH for MUSA,
            # which represents the path of libdevice.bc
            musa_env_path = os.environ.get("MUSA_LIBDEVICE_PATH", default_libdir)
            extern_libs['libdevice'] = musa_env_path
        object.__setattr__(self, 'extern_libs', tuple(extern_libs.items()))
        assert self.num_warps > 0 and (self.num_warps & (self.num_warps - 1)) == 0, \
               "num_warps must be a power of 2"

    def hash(self):
        hash_dict = dict(self.__dict__)
        hash_dict["extern_libs"] = tuple((k, file_hash(v)) for k, v in sorted(hash_dict["extern_libs"]))
        key = "_".join([f"{name}-{val}" for name, val in sorted(hash_dict.items())])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


class MUSABackend(BaseBackend):

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == 'musa'

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        self.capability = target.arch
        self.warp_size = target.warp_size
        assert isinstance(self.capability, int)
        self.binary_ext = "mubin"

    def parse_options(self, opts) -> Any:
        opts["capability"] = self.capability
        opts["allow_fp8e4nv"] = self.capability >= 31
        args = {k: opts[k] for k in MUSAOptions.__dataclass_fields__.keys() if k in opts}
        return MUSAOptions(**args)

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
        import triton.language.extra.musa as musa
        codegen_fns = {
            "convert_custom_types": None,
        }
        return codegen_fns

    def load_dialects(self, ctx):
        mthreads.load_dialects(ctx)

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
    def make_ttgir(mod, metadata, opt, capability, warp_size):
        # TTIR -> TTGIR
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.ttir.add_convert_to_ttgpuir(pm, f"musa:{capability}", opt.num_warps, warp_size, opt.num_ctas)
        # optimize TTGIR
        passes.ttgpuir.add_coalesce(pm)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.common.add_cse(pm)
        passes.ttgpuir.add_combine_tensor_select_and_if(pm)
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
        # warp-specialization mutates num_warps
        num_warp_groups = src.get_int_attr("triton_gpu.num-warp-groups-per-cta")
        if num_warp_groups is not None:
            metadata["num_warps"] *= num_warp_groups
        mod = src
        # TritonGPU -> LLVM-IR (MLIR)
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.convert.add_scf_to_cf(pm)
        passes.convert.add_index_to_llvmir(pm)
        passes.ttgpuir.add_allocate_shared_memory(pm)
        mthreads.passes.ttgpuir.add_to_llvmir(pm, capability)
        passes.convert.add_arith_to_llvmir(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)

        if os.environ.get("TRITON_DISABLE_LINE_INFO", "0") == "0":
            passes.llvmir.add_di_scope(pm)

        # FIXME: shall we consider to use load/store with robust instruction to support ld/st with predicate
        mthreads.passes.ttgpuir.add_mtgpu_builtin_func_to_llvmir(pm)
        pm.run(mod)

        # LLVM-IR (MLIR) -> LLVM-IR (LLVM)
        llvm.init_targets()
        context = llvm.context()

        llvm_mod = llvm.to_module(mod, context)
        mthreads.attach_datalayout(llvm_mod)

        if options.extern_libs:
            paths = [path for (name, path) in options.extern_libs]
            llvm.link_extern_libs(llvm_mod, paths)

        llvm.optimize_module(llvm_mod, llvm.OPTIMIZE_O3)

        # Get some metadata
        metadata["shared"] = src.get_int_attr("triton_gpu.shared")
        ret = str(llvm_mod)
        del llvm_mod
        del context
        return ret

    @staticmethod
    def make_mubin(src, metadata, opt, capability):
        '''
        Translate TritonGPU module to MUSA binary code.
        '''
        if (os.environ.get("LLVM_IR_ENABLE_DUMP", "0") == "1"):
            print("// -----// LLVM IR")
            print(src)

        opt_option = "-mtgpu-enable-const-calc=1"
        if (os.environ.get("MUSA_ENABLE_LLC_OPT", "0") == "1"):
            opt_option = "-mtgpu-opt-level=1"

        ret = mthreads.translate_llvmir_to_mubin(src, opt_option, capability, 0)
        if (os.environ.get("MUSA_ASM_ENABLE_DUMP", "0") == "1"):
            print("// -----// MTGPU ASM")
            print(ret[0])

        mubin_save_path = os.environ.get("MUBIN_SAVE_PATH", "")
        if mubin_save_path != "":
            mubin_file_name = os.path.join(mubin_save_path, "test.out")
            shutil.copy2(ret[1], mubin_file_name)

        metadata["name"] = get_kernel_name(ret[0], pattern='.globl')
        return ret

    def add_stages(self, stages, options):
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["ttgir"] = lambda src, metadata: self.make_ttgir(src, metadata, options, self.capability, self.warp_size)
        stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options, self.capability)
        stages["mubin"] = lambda src, metadata: self.make_mubin(src, metadata, options, self.capability)

    @functools.lru_cache()
    def hash(self):
        version = get_musa_version()
        return f'{version}-{self.capability}'
