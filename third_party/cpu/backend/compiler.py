import functools
import hashlib
import os
import tempfile
from pathlib import Path

from dataclasses import dataclass
from typing import Any, Tuple

from triton._C.libtriton import cpu, ir, llvm, passes
from triton.backends.compiler import BaseBackend, GPUTarget
from triton.runtime.build import _build
import triton.backends.cpu.driver as cpu_driver


def min_dot_size(target: GPUTarget):
    # Other architectures will only support 16,16,16
    return lambda lhsType, rhsType: (4, 4, 4)


@dataclass(frozen=True)
class CPUOptions:
    # GPU-specific options are used in several places.
    # For now, we just provide dummy values.
    backend_name: str = "cpu"
    num_warps: int = 0
    num_stages: int = 0
    num_ctas: int = 0
    cluster_dims: tuple = (1, 1, 1)
    extern_libs: dict = None
    debug: bool = False
    allowed_dot_input_precisions: Tuple[str] = ("ieee", "tf32", "tf32x3")
    allow_fp8e4nv: bool = True
    allow_fp8e4b15: bool = True
    enable_fp_fusion: bool = True
    max_num_imprecise_acc_default: int = 0
    enable_fast_math: bool = True

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
        self.binary_ext = "so"
        self.cpu_arch = llvm.get_cpu_tripple().split("-")[0]
        self.cpu_name = llvm.get_cpu_name()
        self.cpu_features = llvm.get_cpu_features()

    def parse_options(self, opts) -> Any:
        args = {k: opts[k] for k in CPUOptions.__dataclass_fields__.keys() if k in opts}
        if not "enable_fast_math" in args:
            args["enable_fast_math"] = os.getenv("TRITON_CPU_FAST_MATH", "1") != "0"
        return CPUOptions(**args)

    def pack_metadata(self, metadata):
        return metadata

    def get_codegen_implementation(self):
        codegen_fns = {"min_dot_size": min_dot_size(self.target)}
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
        cpu.passes.ttcpuir.add_convert_memory_ops(pm)
        cpu.passes.ttcpuir.add_convert_ptr_ops(pm)
        cpu.passes.ttcpuir.add_convert_elementwise_ops(pm)
        cpu.passes.ttcpuir.add_convert_elem_manip_ops(pm)
        cpu.passes.ttcpuir.add_convert_dot_op(pm)
        cpu.passes.ttcpuir.add_convert_histogram_op(pm)
        cpu.passes.ttcpuir.add_convert_reduction_op(pm)
        cpu.passes.ttcpuir.add_convert_scan_op(pm)
        cpu.passes.ttcpuir.add_convert_cf_ops(pm)
        cpu.passes.ttcpuir.add_convert_atomic_ops(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        passes.common.add_canonicalizer(pm)
        pm.run(mod)
        metadata["cluster_dims"] = (opt.cluster_dims[0], opt.cluster_dims[1], opt.cluster_dims[2])
        return mod

    def make_tttcir(self, mod, metadata, opt):
        # TTCIR -> Target TTCIR
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        cpu.passes.ttcpuir.add_optimize_masks(pm)
        promote_bf16_to_fp32 = self.cpu_arch == "x86_64" and "avx512bf16" not in self.cpu_features
        # We don't have any lowering for mixed precision matmuls, so always use casts for now
        convert_mixed_precision_matmul = True
        # We don't have math lib functions for FP8, FP16, BF16. Promote such operations to FP32.
        promote_lib_math_to_fp32 = True
        cpu.passes.ttcpuir.add_convert_unsupported_ops(pm, promote_bf16_to_fp32, convert_mixed_precision_matmul,
                                                       promote_lib_math_to_fp32)
        decompose_bf16_conv = self.cpu_arch == "x86_64" and "avx512bf16" not in self.cpu_features
        decompose_fp8_conv = True
        cpu.passes.ttcpuir.add_decompose_fp_conversions(pm, decompose_bf16_conv, decompose_fp8_conv)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        passes.common.add_canonicalizer(pm)
        pm.run(mod)
        return mod

    def make_llir(self, src, metadata, options):
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
        cpu.passes.ttcpuir.add_func_op_to_llvmir(pm)
        cpu.passes.ttcpuir.add_program_id_to_llvmir(pm)
        cpu.passes.ttcpuir.add_memory_op_to_llvmir(pm)
        cpu.passes.ttcpuir.add_atomic_ops_to_llvmir(pm)
        cpu.passes.ttcpuir.add_debug_ops_to_llvmir(pm)
        use_sleef = os.environ.get("TRITON_CPU_USE_SLEEF", "0") != "0"
        use_vec_math = os.environ.get("TRITON_CPU_USE_LIBMVEC", "1") != "0"
        if (use_sleef or use_vec_math) and self.cpu_arch == "x86_64" and "avx512f" in self.cpu_features:
            cpu.passes.ttcpuir.add_math_to_libmvec(pm, use_sleef)
        passes.convert.add_math_to_llvmir(pm)
        cpu.passes.ttcpuir.add_math_to_libm(pm)
        cpu.passes.ttcpuir.add_vector_to_llvmir(pm, options.enable_fast_math)
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
        if llvm_mod is None:
            raise RuntimeError("Failed to convert to LLVM IR")
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
        return llvm.translate_to_host_asm(src, options.enable_fp_fusion, options.enable_fast_math)

    @staticmethod
    def make_so(src, metadata, options):
        with tempfile.TemporaryDirectory() as tmpdir:
            asm_path = os.path.join(tmpdir, "kernel.s")
            Path(asm_path).write_text(src)
            lib_dirs = cpu_driver.library_dirs
            libs = ["gcc", "m", "TritonCPURuntime"]
            # TRITON_CPU_USE_SLEEF=1 - use system libsleef
            # TRITON_CPU_USE_SLEEF=path - use libsleef from the specified path
            use_sleef = os.environ.get("TRITON_CPU_USE_SLEEF", "0")
            if use_sleef != "0":
                if os.path.isdir(use_sleef):
                    lib_dirs.append(use_sleef)
                libs.append("sleef")
            so = _build("kernel", asm_path, tmpdir, lib_dirs, cpu_driver.include_dirs, libs)
            with open(so, "rb") as f:
                return f.read()

    def add_stages(self, stages, options):
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["ttcir"] = lambda src, metadata: self.make_ttcir(src, metadata, options)
        stages["tttcir"] = lambda src, metadata: self.make_tttcir(src, metadata, options)
        stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options)
        stages["asm"] = lambda src, metadata: self.make_asm(src, metadata, options)
        stages["so"] = lambda src, metadata: self.make_so(src, metadata, options)

    @functools.lru_cache()
    def hash(self):
        # TODO: Get more detailed CPU info like raw brand name with supported ISAs.
        # Right now it would only return a simple string like "x86_64" or "aarch64".
        import platform

        return f"{platform.machine()}"
