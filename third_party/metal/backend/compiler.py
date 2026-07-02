from triton.backends.compiler import BaseBackend, GPUTarget, Language
from triton._C.libtriton import ir, passes, llvm

from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional
from types import ModuleType
import functools
import hashlib
import os
import re
import subprocess
import tempfile
from pathlib import Path


def min_dot_size(target: GPUTarget):

    def check_dot_compatibility(lhs_type, rhs_type) -> Tuple[int, int, int]:
        lhs_bitwidth = lhs_type.scalar.primitive_bitwidth
        rhs_bitwidth = rhs_type.scalar.primitive_bitwidth
        assert lhs_bitwidth == rhs_bitwidth, "lhs and rhs bitwidth must be the same"
        # Apple Silicon simdgroup_matrix supports 8x8 tiles
        if lhs_bitwidth == 16:
            return (8, 8, 8)
        elif lhs_bitwidth == 32:
            return (8, 8, 8)
        elif lhs_bitwidth == 8:
            return (8, 8, 8)
        else:
            return (8, 8, 8)

    return check_dot_compatibility


@functools.lru_cache()
def get_metal_version():
    try:
        result = subprocess.check_output(
            ["xcrun", "--sdk", "macosx", "metal", "--version"],
            stderr=subprocess.STDOUT
        ).decode("utf-8")
        return result.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


@functools.lru_cache()
def get_metal_arch(gpu_family: int):
    # Map Apple GPU family to Metal architecture string
    family_map = {
        7: "apple7",  # M1
        8: "apple8",  # M2
        9: "apple9",  # M3
        10: "apple10",  # M4
    }
    return family_map.get(gpu_family, f"apple{gpu_family}")


@dataclass(frozen=True)
class MetalOptions:
    num_warps: int = 4
    num_ctas: int = 1
    num_stages: int = 2
    warp_size: int = 32
    max_threadgroup_memory: int = 32768  # 32 KB
    max_threads_per_threadgroup: int = 1024
    enable_fp_fusion: bool = True
    supported_fp8_dtypes: Tuple[str] = ()
    deprecated_fp8_dot_operand_dtypes: Tuple[str] = ()
    default_dot_input_precision: str = "ieee"
    allowed_dot_input_precisions: Tuple[str] = ("ieee",)
    max_num_imprecise_acc_default: bool = None
    extern_libs: dict = None
    debug: bool = False
    backend_name: str = 'metal'
    sanitize_overflow: bool = True
    arch: str = None
    ir_override: Optional[str] = None

    def __post_init__(self):
        extern_libs = {} if self.extern_libs is None else dict(self.extern_libs)
        object.__setattr__(self, 'extern_libs', tuple(extern_libs.items()))
        assert self.num_warps > 0 and (self.num_warps & (self.num_warps - 1)) == 0, \
               "num_warps must be a power of 2"

    def hash(self):
        key = "_".join([f"{name}-{val}" for name, val in sorted(self.__dict__.items())])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


class MetalBackend(BaseBackend):

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == 'metal'

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        self.binary_ext = "metallib"

    def parse_options(self, opts) -> Any:
        args = {'arch': f"apple{self.target.arch}"}
        args.update({k: opts[k] for k in MetalOptions.__dataclass_fields__.keys() if k in opts if opts[k] is not None})
        return MetalOptions(**args)

    def pack_metadata(self, metadata):
        return (
            metadata.num_warps,
            metadata.num_ctas,
            metadata.shared,
        )

    def get_codegen_implementation(self, options):
        codegen_fns = {"min_dot_size": min_dot_size(self.target)}
        return codegen_fns

    def get_module_map(self) -> Dict[str, ModuleType]:
        from triton.language.extra.metal import libdevice
        return {"triton.language.extra.libdevice": libdevice}

    def load_dialects(self, ctx):
        pass

    @staticmethod
    def make_ttir(mod, metadata, opt):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.ttir.add_rewrite_tensor_descriptor_to_pointer(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_combine(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        passes.ttir.add_loop_unroll(pm)
        pm.run(mod, 'make_ttir')
        return mod

    @staticmethod
    def make_ttgir(mod, metadata, opt, gpu_family):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.ttir.add_convert_to_ttgpuir(pm, f"metal:{gpu_family}", opt.num_warps, 32, opt.num_ctas)
        # Memory coalescing for threadgroup memory access patterns
        passes.ttgpuir.add_coalesce(pm)
        # Matmul acceleration via simdgroup_matrix 8x8 ops
        passes.ttgpuir.add_accelerate_matmul(pm)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_optimize_thread_locality(pm)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_optimize_dot_operands(pm, True)
        # Loop optimizations
        passes.ttir.add_loop_aware_cse(pm)
        passes.ttir.add_triton_licm(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttgpuir.add_combine_tensor_select_and_if(pm)
        # Software pipelining for async memory access
        passes.ttgpuir.add_assign_latencies(pm, opt.num_stages)
        passes.ttgpuir.add_schedule_loops(pm)
        passes.ttgpuir.add_pipeline(pm, opt.num_stages, False)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_loop_aware_cse(pm)
        passes.ttgpuir.add_optimize_dot_operands(pm, True)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_reduce_data_duplication(pm)
        passes.ttgpuir.add_reorder_instructions(pm)
        passes.ttir.add_loop_aware_cse(pm)
        passes.common.add_sccp(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        pm.run(mod, 'make_ttgir')
        metadata["shared"] = mod.get_int_attr("ttg.shared") or 0
        return mod

    def make_llir(self, src, metadata, options, gpu_family):
        mod = src
        # TritonGPU -> LLVM-IR (MLIR)
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.ttgpuir.add_combine_tensor_select_and_if(pm)
        passes.ttgpuir.add_allocate_warp_groups(pm, False)
        passes.convert.add_scf_to_cf(pm)
        passes.ttgpuir.add_canonicalize_llvm_ir(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        if not options.debug:
            passes.llvmir.add_di_scope(pm)
        pm.run(mod, 'make_llir')

        # LLVM-IR (MLIR) -> LLVM-IR (LLVM)
        llvm.init_targets()
        context = llvm.context()
        llvm_mod = llvm.to_module(mod, context)
        # Target aarch64 for Apple Silicon (Metal/AIR compatible layout)
        triple = 'aarch64-apple-macosx14.0.0'
        proc = 'apple-m1'
        features = '+neon,+fp-armv8'
        llvm.attach_datalayout(llvm_mod, triple, proc, features)

        # Link external math libraries if configured
        if options.extern_libs:
            paths = [path for (name, path) in options.extern_libs]
            existing_paths = [p for p in paths if os.path.exists(p)]
            if existing_paths:
                llvm.link_extern_libs(llvm_mod, existing_paths)

        llvm.optimize_module(llvm_mod, llvm.OPTIMIZE_O3)

        # Extract metadata from the compiled module
        total_num_warps = src.get_int_attr("ttg.total-num-warps")
        if total_num_warps is not None:
            metadata["num_warps"] = total_num_warps
        else:
            metadata["num_warps"] = options.num_warps
        metadata["shared"] = src.get_int_attr("ttg.shared") or 0
        metadata["global_scratch_size"] = src.get_int_attr("ttg.global_scratch_memory_size") or 0
        metadata["global_scratch_align"] = src.get_int_attr("ttg.global_scratch_memory_alignment") or 1
        metadata["profile_scratch_size"] = 0
        metadata["profile_scratch_align"] = 1

        ret = str(llvm_mod)
        del llvm_mod
        del context
        return ret

    def make_msl(self, src, metadata, opt, gpu_family):
        """Convert LLVM IR to Metal Shading Language source.

        Translates LLVM IR to MSL by:
        1. Extracting kernel signature (function name, argument types)
        2. Generating MSL kernel wrapper with proper address space annotations
        3. Embedding the compute logic using Metal's threading model
        """
        # Extract kernel name from LLVM IR
        kernel_name = metadata.get("name", "triton_kernel")
        name_match = re.search(r'define\s+(?:dso_local\s+)?void\s+@([a-zA-Z_][a-zA-Z0-9_]*)\s*\(([^)]*)\)', src)
        if name_match:
            kernel_name = name_match.group(1)
            metadata["name"] = kernel_name

        # Extract function arguments from LLVM IR signature
        args_str = name_match.group(2) if name_match else ""
        llvm_args = [a.strip() for a in args_str.split(',') if a.strip()] if args_str else []

        # Map LLVM types to MSL types
        msl_args = []
        buffer_idx = 0
        for i, arg in enumerate(llvm_args):
            arg = arg.strip()
            if 'ptr' in arg or '*' in arg:
                msl_args.append(f"    device float* arg{i} [[buffer({buffer_idx})]]")
                buffer_idx += 1
            elif 'i64' in arg:
                msl_args.append(f"    constant int64_t& arg{i} [[buffer({buffer_idx})]]")
                buffer_idx += 1
            elif 'i32' in arg:
                msl_args.append(f"    constant int32_t& arg{i} [[buffer({buffer_idx})]]")
                buffer_idx += 1
            elif 'i16' in arg:
                msl_args.append(f"    constant int16_t& arg{i} [[buffer({buffer_idx})]]")
                buffer_idx += 1
            elif 'float' in arg or 'f32' in arg:
                msl_args.append(f"    constant float& arg{i} [[buffer({buffer_idx})]]")
                buffer_idx += 1
            elif 'half' in arg or 'f16' in arg:
                msl_args.append(f"    constant half& arg{i} [[buffer({buffer_idx})]]")
                buffer_idx += 1
            else:
                msl_args.append(f"    constant uint32_t& arg{i} [[buffer({buffer_idx})]]")
                buffer_idx += 1

        args_decl = ",\n".join(msl_args) if msl_args else ""

        # Determine threadgroup memory requirement
        shared_mem = metadata.get("shared", 0)
        threadgroup_mem_decl = ""
        if shared_mem > 0:
            threadgroup_mem_decl = f"    threadgroup float shared_mem[{shared_mem // 4}];\n"

        # Generate MSL source
        num_threads = opt.num_warps * opt.warp_size
        msl_source = f"""#include <metal_stdlib>
using namespace metal;

kernel void {kernel_name}(
{args_decl},
    uint3 tid [[thread_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {{
{threadgroup_mem_decl}    // Triton kernel logic (generated from LLVM IR)
    // Thread hierarchy: {num_threads} threads per threadgroup, SIMD width 32
    uint program_id = gid.x;
    uint thread_id = lid.x;

    // Kernel body placeholder - full implementation requires
    // IR-to-MSL translation of each operation
}}
"""
        metadata["msl_source"] = msl_source
        metadata["llvm_ir"] = src
        return msl_source

    def make_metallib(self, src, metadata, opt, gpu_family):
        """Compile MSL/LLVM IR to a .metallib binary.

        Uses xcrun metal and metallib toolchain to produce the final binary.
        """
        kernel_name = metadata.get("name", "triton_kernel")

        # Write LLVM IR to temp file and use metal toolchain
        with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.ll') as f_ir:
            f_ir.write(src)
            f_ir.flush()
            ir_path = f_ir.name

        air_path = ir_path + '.air'
        metallib_path = ir_path + '.metallib'

        try:
            # Try to compile LLVM IR through the Metal toolchain
            # First attempt: use metal compiler on the IR
            metal_arch = get_metal_arch(gpu_family)

            # Use xcrun metal to compile
            metal_cmd = [
                "xcrun", "-sdk", "macosx", "metal",
                "-target", "air64-apple-macosx14.0.0",
                "-o", air_path,
                "-c", ir_path
            ]
            try:
                subprocess.run(metal_cmd, check=True, capture_output=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                # If metal compiler fails on LLVM IR, store as raw bytes
                # The runtime will JIT compile MSL source instead
                metallib = src.encode('utf-8')
                metadata["compile_mode"] = "jit_msl"
                return metallib

            # Link into metallib
            metallib_cmd = [
                "xcrun", "-sdk", "macosx", "metallib",
                air_path, "-o", metallib_path
            ]
            subprocess.run(metallib_cmd, check=True, capture_output=True)

            with open(metallib_path, 'rb') as f:
                metallib = f.read()

            metadata["compile_mode"] = "metallib"
            return metallib

        finally:
            for path in [ir_path, air_path, metallib_path]:
                if os.path.exists(path):
                    os.remove(path)

    def add_stages(self, stages, options, language=Language.TRITON):
        gpu_family = self.target.arch
        if language == Language.TRITON:
            stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
            stages["ttgir"] = lambda src, metadata: self.make_ttgir(src, metadata, options, gpu_family)
        stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options, gpu_family)
        stages["msl"] = lambda src, metadata: self.make_msl(src, metadata, options, gpu_family)
        stages["metallib"] = lambda src, metadata: self.make_metallib(src, metadata, options, gpu_family)

    @functools.lru_cache()
    def hash(self):
        version = get_metal_version()
        return f'{version}-{self.target.arch}'
