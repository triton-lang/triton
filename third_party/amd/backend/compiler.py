from triton.backends.compiler import BaseBackend
from triton._C.libtriton import ir, passes, llvm, amd
from dataclasses import dataclass
from typing import Any
import hashlib
import tempfile
import os
import re
import subprocess
import functools
from pathlib import Path

@dataclass(frozen=True)
class HIPOptions:
    num_warps: int = 4
    waves_per_eu: int = 1
    num_stages: int = 0
    num_ctas: int = 1
    extern_libs: dict = None
    cluster_dims: tuple = (1, 1, 1)
    debug: bool = False
    arch: str = None
    allow_fp8e4nv: bool = False
    enable_fp_fusion: bool = True
    capability: int = None
    # TODO:
    matrix_core_version: int = -1
    matrix_inst_shape: int = 0
    max_num_imprecise_acc_default: int = 0

    @staticmethod
    def get_warp_size(arch: str) -> int:
        # 64 is not supported for RDNA for now
        if 'gfx10' in arch or 'gfx11' in arch:
            return 32
        if 'gfx9' in arch:
            return 64
        print("Warning: Unexpected device. Wave Size is set to 64.")
        return 64 # Default value

    @staticmethod
    def get_matrix_core_version(arch: str) -> int:
        """ Determine matrix core type available on current GPU.
            0 means no tensor cores are available
            1 corresponds to MFMA in CDNA 1 architecture
            2 corresponds to MFMA in CDNA 2 architecture
            3 corresponds to MFMA in CDNA 3 architecture
        """
        if arch in ['gfx908']:
            return 1
        if arch in ['gfx90a']:
            return 2
        if arch in ['gfx940', 'gfx941', 'gfx942']:
            return 3
        return 0

    def __post_init__(self):
        default_libdir = Path(__file__).parent / 'lib'
        extern_libs = dict() if self.extern_libs is None else dict(self.extern_libs)
        # Ignore user-defined warp size for gfx9
        warp_size = 32 if 'gfx10' in self.arch or 'gfx11' in self.arch else 64
        object.__setattr__(self, 'warp_size', warp_size)
        oclc_wavefrontsize_lib = "oclc_wavefrontsize64_on" if self.warp_size == 64 else "oclc_wavefrontsize64_off"
        libs = [
            "cuda2gcn", "opencl", "ocml", "ockl", "oclc_finite_only_off", "oclc_daz_opt_off",
            "oclc_correctly_rounded_sqrt_on", "oclc_unsafe_math_off", oclc_wavefrontsize_lib, "oclc_abi_version_400"
        ]
        libs += ['oclc_isa_version_' + self.arch.replace('gfx', '')]
        for lib in libs:
            extern_libs[lib] = str(default_libdir / f'{lib}.bc')
        object.__setattr__(self, 'extern_libs', tuple(extern_libs.items()))
        assert self.num_warps > 0 and (self.num_warps & (self.num_warps - 1)) == 0, \
               "num_warps must be a power of 2"
        if(self.matrix_core_version == -1):
            object.__setattr__(self, 'matrix_core_version', self.get_matrix_core_version(self.arch))

    def hash(self):
        key = '_'.join([f'{name}-{val}' for name, val in self.__dict__.items()])
        return hashlib.md5(key.encode("utf-8")).hexdigest()


class HIPBackend(BaseBackend):

    @staticmethod
    def supports_target(target: tuple):
        return target[0] == 'hip'

    def __init__(self, target: tuple) -> None:
        super().__init__(target)
        assert isinstance(target, tuple) and len(target) == 2
        assert isinstance(target[1], str)

    def parse_options(self, opts) -> Any:
        args = {'arch': self.target[1]}
        args.update({k: opts[k] for k in HIPOptions.__dataclass_fields__.keys() if k in opts})
        return HIPOptions(**args)

    def load_dialects(self, ctx):
        amd.load_dialects(ctx)

    @staticmethod
    def path_to_rocm_lld():
        return "/opt/rocm/llvm/bin/ld.lld"

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
    def make_ttgir(mod, metadata, opt):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        # TODO: capability
        passes.ttir.add_convert_to_ttgpuir(pm, opt.num_warps, opt.warp_size, opt.num_ctas, 90)
        pm.run(mod)
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.ttgpuir.add_coalesce(pm)
        amd.passes.ttgpuir.add_remove_layout_conversions(pm)
        amd.passes.ttgpuir.add_accelerate_matmul(pm, opt.matrix_core_version, opt.matrix_inst_shape)
        amd.passes.ttgpuir.add_remove_layout_conversions(pm)
        amd.passes.ttgpuir.add_optimize_epilogue(pm)
        passes.ttgpuir.add_optimize_dot_operands(pm)
        if opt.num_stages == 0 and opt.matrix_core_version != 0:
            amd.passes.ttgpuir.add_stream_pipeline(pm)
            passes.common.add_canonicalizer(pm)
        passes.ttgpuir.add_pipeline(pm, opt.num_stages, opt.num_warps, opt.num_ctas, 0)
        passes.ttgpuir.add_optimize_dot_operands(pm)
        amd.passes.ttgpuir.add_remove_layout_conversions(pm)
        amd.passes.ttgpuir.add_decompose_conversions(pm)
        if opt.num_stages != 0:
            amd.passes.ttgpuir.add_reorder_instructions(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        pm.run(mod)
        return mod

    @staticmethod
    def make_llir(src, metadata, options, capability):
        mod = src
        # TritonGPU -> LLVM-IR (MLIR)
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.convert.add_scf_to_cf(pm)
        passes.convert.add_index_to_llvmir(pm)
        pm.run(mod)
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        amd.passes.ttgpuir.add_to_llvmir(pm)
        pm.run(mod)
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
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
        if options.extern_libs:
            for name, path in options.extern_libs:
                llvm.link_extern_lib(llvm_mod, path)
        llvm.optimize_module(llvm_mod, llvm.OPTIMIZE_O3)
        # Set kernel attributes
        kernels = [fn for fn in llvm_mod.get_functions() if not fn.is_declaration()]
        # The public kernel should be kernel 0
        kernels[0].set_calling_conv(amd.CALLING_CONV_AMDGPU_KERNEL)
        kernels[0].add_fn_attr("amdgpu-flat-work-group-size", f"1, {options.num_warps*options.warp_size}")
        kernels[0].add_fn_attr("amdgpu-waves-per-eu", f"{options.waves_per_eu}")
        kernels[0].add_fn_attr("denormal-fp-math-f32", "preserve-sign")
        # Get some metadata
        metadata["shared"] = src.get_int_attr("triton_gpu.shared")
        ret = str(llvm_mod)
        return ret

    @staticmethod
    def make_hsaco(src, metadata, options):
        # Find kernel names (there should only be one)
        # We get the name at the last possible step to accomodate `triton.compile`
        # on user-provided LLVM
        names = re.findall(r"define amdgpu_kernel void @([a-zA-Z_][a-zA-Z0-9_]*)", src)
        assert len(names) == 1
        metadata["name"] = names[0]
        # llvm -> hsaco
        hsaco = llvm.translate_to_asm(src, 'amdgcn-amd-amdhsa', options.arch, '', [], options.enable_fp_fusion, True)
        if os.environ.get("AMDGCN_ENABLE_DUMP", "0") == "1":
            hsaco_str = llvm.translate_to_asm(src, 'amdgcn-amd-amdhsa', options.arch, '', [], options.enable_fp_fusion, False)
            print("// -----// AMDGCN Dump //----- //")
            print(hsaco_str)
        import subprocess
        rocm_path = HIPBackend.path_to_rocm_lld()
        with tempfile.NamedTemporaryFile() as tmp_out:
            with tempfile.NamedTemporaryFile() as tmp_in:
                with open(tmp_in.name, 'wb') as fd_in:
                    fd_in.write(hsaco)
                subprocess.check_call([rocm_path, '-flavor', 'gnu', '-shared', tmp_in.name, '-o', tmp_out.name])
            with open(tmp_out.name, 'rb') as fd_out:
                ret = fd_out.read()
        return ret

    def add_stages(self, stages, options):
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["ttgir"] = lambda src, metadata: self.make_ttgir(src, metadata, options)
        stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options, 90)
        # TODO: first amdgcn, then hsaco
        # stages["amdgcn"] = lambda src, metadata: self.make_amdgcn(src, metadata, options)
        stages["hsaco"] = lambda src, metadata: self.make_hsaco(src, metadata, options)

    @functools.lru_cache()
    def hash(self):
        version = subprocess.check_output([HIPBackend.path_to_rocm_lld(), "--version"])
        return f'{version}-{self.target}'
