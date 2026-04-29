import functools
import hashlib
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from triton import knobs
from triton._C.libtriton import ir, llvm, metal, passes
from triton.backends.compiler import BaseBackend, GPUTarget, Language

from .air_utils import convert_opaque_ptrs_to_typed


def get_min_dot_size(target: GPUTarget):
    # TODO copied from AMD, modify if needed
    return lambda lhs_type, rhs_type: (1, 1, 1)


@dataclass(frozen=True)
class MetalOptions:
    # TODO add more metal-specific options as needed
    backend_name: str = "metal"
    num_warps: int = 4
    num_stages: int = 1
    num_ctas: int = 1
    default_dot_input_precision: str = "ieee"
    allowed_dot_input_precisions: Tuple[str] = ("ieee",)
    warp_size: int = 32  # SIMD group size
    sanitize_overflow: bool = True  # TODO copied from AMD, modify if needed
    debug: bool = False
    extern_libs: Optional[dict] = None
    arch: Optional[str] = None
    instrumentation_mode: str = ""
    enable_fp_fusion: bool = True

    supported_fp8_dtypes: Tuple = ()  # TODO I believe mac does not support fp8
    launch_cooperative_grid: bool = False

    def __post_init__(self):
        # TODO verify this
        default_libdir = Path(__file__).parent / "lib"
        extern_libs = {} if self.extern_libs is None else dict(self.extern_libs)
        object.__setattr__(self, "extern_libs", tuple(extern_libs.items()))

    def hash(self):
        # TODO copied from AMD, modify if needed
        key = "_".join([f"{name}-{val}" for name, val in self.__dict__.items()])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


class MetalBackend(BaseBackend):
    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == "metal"

    def __init__(self, target: tuple) -> None:
        super().__init__(target)
        self.binary_ext = "metallib"

    @staticmethod
    def make_ttir(mod, metadata, opt):
        # TODO copied from triton-cpu/python/triton/backends/cpu/compiler.py, need to verify
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.ttir.add_rewrite_tensor_descriptor_to_pointer(pm)
        passes.ttir.add_combine(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.common.add_licm(pm)
        passes.common.add_symbol_dce(pm)
        pm.run(mod, "make_ttir")
        return mod

    @staticmethod
    def make_ttgir(mod, metadata, options):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        # TODO what to put for architecture label
        passes.ttir.add_convert_to_ttgpuir(pm, "metal", options.num_warps, options.warp_size, options.num_ctas)
        pm.run(mod, "make_ttgir_early")
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()

        # metal.passes.ttgpuir.add_accelerate_matmul(pm)
        metal.passes.ttgpuir.add_inject_tensor_stride_args(pm)
        metal.passes.ttgpuir.add_allocate_smem_for_simdgroup_matmul(pm)
        pm.run(mod, "make_ttgir")
        metadata["tensordesc_meta"] = mod.get_tensordesc_metadata()
        return mod

    @staticmethod
    def make_air(src, metadata, options):
        mod = src
        # TritonGPU -> LLVM-IR (MLIR)
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()

        passes.convert.add_scf_to_cf(pm)
        passes.ttgpuir.add_allocate_shared_memory(pm)

        metal.passes.ttgpuir.add_to_llvmir(pm, str(options.arch))
        passes.common.add_canonicalizer(pm)

        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)

        pm.run(mod, "make_llir")

        # LLVM-IR (MLIR) -> LLVM-IR (LLVM)
        llvm.init_targets()
        context = llvm.context()
        llvm_mod = llvm.to_module(mod, context)

        # TODO don't hardcode
        llvm_mod.set_target_triple("air64-apple-macosx15.0.0")
        llvm_mod.set_data_layout(
            "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-"
            "f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-"
            "v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-"
            "v512:512:512-v1024:1024:1024-n8:16:32"
        )

        # add module flags
        llvm_mod.add_flag(llvm.MODULE_FLAG_BEHAVIOR_ERROR, "wchar_size", 4)
        llvm_mod.add_flag(llvm.MODULE_FLAG_BEHAVIOR_MAX, "frame-pointer", 2)
        llvm_mod.add_flag(llvm.MODULE_FLAG_BEHAVIOR_MAX, "air.max_device_buffers", 31)
        llvm_mod.add_flag(llvm.MODULE_FLAG_BEHAVIOR_MAX, "air.max_constant_buffers", 31)
        llvm_mod.add_flag(llvm.MODULE_FLAG_BEHAVIOR_MAX, "air.max_threadgroup_buffers", 31)
        llvm_mod.add_flag(llvm.MODULE_FLAG_BEHAVIOR_MAX, "air.max_textures", 128)
        llvm_mod.add_flag(llvm.MODULE_FLAG_BEHAVIOR_MAX, "air.max_read_write_textures", 8)
        llvm_mod.add_flag(llvm.MODULE_FLAG_BEHAVIOR_MAX, "air.max_samplers", 16)

        # add metadata
        metal.add_kernel_metadata(llvm_mod)

        ret = str(llvm_mod)

        # replace llvm attributes with compatible versions
        llvm_attributes_replacements = {
            "captures(none)": "nocapture",
            "memory(none)": "readnone",
            "memory(read)": "readonly",
            "memory(write)": "writeonly",
            "memory(readwrite)": "",
        }
        for orig, new_attr in llvm_attributes_replacements.items():
            ret = ret.replace(orig, new_attr)

        # strip memory(...) attrs (e.g. memory(inaccessiblemem: write))
        # that older LLVM can't parse
        ret = re.sub(r"\bmemory\([^)]*:[^)]*\)", "", ret)

        # convert LLVM 17+ splat syntax to old vector constant format
        # xcrun metal uses old LLVM: use <type val> instead of splat(type val)
        ret = re.sub(r"splat\s*\(([^)]+)\)", r"<\1>", ret)

        # remove newer attributes unknown to old LLVM
        ret = re.sub(r"\bnocreateundeforpoison\b", "", ret)

        # convert opaque ptrs to typed ptrs for metal using regex
        # newer llvm that triton uses does not support typed ptrs
        # xcrun metal (older llvm) with -opaque-pointers does not work with metal jit can't compile
        ret = convert_opaque_ptrs_to_typed(ret)

        # find kernel name
        names = re.findall(r"define void @([a-zA-Z_][a-zA-Z0-9_]*)", ret)
        assert len(names) == 1
        metadata["name"] = names[0]

        # get more metadata
        # TODO need to handle this after adding allocate shared mem
        metadata["shared"] = src.get_int_attr("ttg.shared")

        del llvm_mod
        del context
        return ret

    @staticmethod
    def make_metallib(src: str, metadata, opt) -> bytes:
        # TODO check what is in metadata and opt
        with tempfile.TemporaryDirectory() as tmpdir:
            air_path = os.path.join(tmpdir, "kernel.ll")
            lib_path = os.path.join(tmpdir, "kernel.metallib")

            with open(air_path, "w") as f:
                f.write(src)

            # ir -> metallib
            result = subprocess.run(
                ["xcrun", "metal", "-x", "ir", air_path, "-o", lib_path], capture_output=True, text=True
            )
            if result.returncode != 0:
                raise RuntimeError(f"metal compiler failed: {result.stderr}")
            with open(lib_path, "rb") as f:
                return f.read()

    def add_stages(self, stages: dict, options: object, language: Language) -> None:
        assert language == Language.TRITON
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["ttgir"] = lambda src, metadata: self.make_ttgir(src, metadata, options)
        stages["air"] = lambda src, metadata: self.make_air(src, metadata, options)
        stages["metallib"] = lambda src, metadata: self.make_metallib(src, metadata, options)

    @functools.lru_cache()
    def hash(self):
        # TODO modify if needed, currently just return the target
        return f"{self.target}"

    def parse_options(self, opts: dict) -> MetalOptions:
        # Enable debug mode for ConSan, so device-side assertions are not optimized out
        if any(mode in opts.get("instrumentation_mode", "") for mode in ["consan"]):
            opts["debug"] = True
            opts["sanitize_overflow"] = False

        args = {"arch": knobs.runtime.override_arch or self.target.arch}
        args.update({k: opts[k] for k in MetalOptions.__dataclass_fields__.keys() if k in opts if opts[k] is not None})

        if args.get("num_ctas", 1) > 1 and not metal.supports_multi_cta_launch(self.target.arch):
            raise ValueError(f"num_ctas > 1 not supported on {self.target.arch}")

        if "supported_fp8_dtypes" not in args:
            args["supported_fp8_dtypes"] = tuple(sorted(MetalOptions.supported_fp8_dtypes))

        if "enable_fp_fusion" not in args:
            args["enable_fp_fusion"] = knobs.language.default_fp_fusion

        return MetalOptions(**args)

    def pack_metadata(self, metadata):
        # TODO modify if needed
        return metadata

    def get_codegen_implementation(self, options):
        # TODO copied from AMD backend, modify if needed
        return {"min_dot_size": get_min_dot_size(self.target)}

    def load_dialects(self, context):
        # TODO no dialects for now, add if needed
        pass

    def get_module_map(self) -> dict:
        # TODO no additional modules for now, add if needed
        return {}
