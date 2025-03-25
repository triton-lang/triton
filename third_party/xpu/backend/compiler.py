from triton.backends.compiler import BaseBackend, GPUTarget
from triton._C.libtriton import ir, passes, xpu, llvm
from triton.runtime.cache import get_cache_manager
import subprocess
import tempfile
import re
import warnings

import os

from dataclasses import dataclass
import functools
from typing import Any, Tuple, Optional
import hashlib
from pathlib import Path


@functools.lru_cache(None)
def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# @dataclass   create __dataclass_fields__ specical attribute
# frozen=True can't modift entry's attribute once it have been created
# [raise FrozenInstanceError]
@dataclass(frozen=True)
class XPUOptions:
    arch: int = int(os.environ.get("TRITON_XPU_ARCH", "3"))
    assert arch in [2, 3, 4], "Invalid XPU ARCH"
    cluster_num: int = 12 if arch == 3 else 8
    core_num: int = 64
    buffer_size_limit: int = 512
    extern_libs: dict = None
    debug: bool = False
    backend_name: str = "xpu"
    cluster_dims: tuple = (1, 1, 1)  # TODO: find mapping relationship

    isOpenCmpNan: bool = False
    isCloseOffsetAnalysis: bool = False
    isCloseCoreTiling: bool = False
    isCloseUnrollControl: bool = False
    isCLOSE_TTXPU_O_ATOMIC_SIM: bool = False

    enable_fp_fusion: bool = False
    allow_fp8e4nv: bool = False
    allow_fp8e4b15: bool = False
    default_dot_input_precision: str = "ieee"
    allowed_dot_input_precisions: Tuple[str] = ("ieee", )

    num_warps: int = (-1)  # TODO: invalid value, just to keep num_warps function signature
    num_ctas: int = -1  # TODO: invalid value, just to keep num_ctas function signature
    num_stages: int = 1

    def __post_init__(self):
        default_libdir = Path(__file__).parent / f"xpu{self.arch}"
        extern_libs = {} if self.extern_libs is None else dict(self.extern_libs)
        if not extern_libs.get("libdevice", None):
            extern_libs["libdevice"] = os.getenv(
                "TRITON_LIBDEVICE_PATH",
                str(default_libdir / "lib" / f"libdevice-xpu{self.arch}.bc"),
            )
            if not os.path.exists(extern_libs["libdevice"]):
                warnings.warn(f'libdevice not found: {extern_libs["libdevice"]}', UserWarning)
                del extern_libs["libdevice"]

        object.__setattr__(self, "extern_libs", tuple(extern_libs.items()))

        invalid_params = []
        if self.num_warps != -1:
            invalid_params.append(f"num_warps={self.num_warps}")
        if self.num_ctas != -1:
            invalid_params.append(f"num_ctas={self.num_ctas}")
        if len(invalid_params) > 0:
            warnings.warn(f"Invalid {', '.join(invalid_params)} in xpu arch", UserWarning)

    def hash(self):
        hash_dict = dict(self.__dict__)
        hash_dict["extern_libs"] = tuple((k, file_hash(v)) for k, v in sorted(hash_dict["extern_libs"]))
        key = "_".join([f"{name}-{val}" for name, val in sorted(hash_dict.items())])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


class XPUBackend(BaseBackend):

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        assert isinstance(target.arch, int)
        self.binary_ext = "xpubin"
        self.buffer_len = 128

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == "xpu"

    @staticmethod
    def path_to_xpu_compile_tool(opt):
        # Check env path for clang
        if "TRITON_XPU_CLANG_PATH" in os.environ:
            clang_path = os.getenv("TRITON_XPU_CLANG_PATH")
            return clang_path
        return os.path.join(Path(__file__).parent, f"xpu{opt.arch}", "bin")

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
    def make_ttxir(mod, metadata, opt):
        metadata["xpu_arch"] = opt.arch
        metadata["shared"] = (-1)  # TODO: invalid value, just to keep CompiledKernel _init_handles() success

        max_buffer_size = int(os.environ.get("TRITONXPU_BUFFER_SIZE", metadata["buffer_size_limit"]))
        max_buffer_size = metadata["buffer_size_limit"]
        XPUBackend.buffer_len = xpu.get_buffer_len(mod, max_buffer_size)
        # print(f"XPUBackend.buffer_len = {XPUBackend.buffer_len}")
        core_num = metadata["core_num"]

        # F/O Prefix For Function/Optimization Macro
        TTXPU_F_OHTER_VALUE_SIM = int(os.environ.get("TRITONXPU_OTHER_SIM", 0))
        TTXPU_F_STORE_MASK_SIM = int(os.environ.get("TRITONXPU_STORE_MASK_SIM", 0))
        TTXPU_F_DTYPE_CONVERT = int(os.environ.get("TRITONXPU_DTYPE_CONVERT", 1))
        TTXPU_O_ATOMIC_SIM = 0 if metadata["isCLOSE_TTXPU_O_ATOMIC_SIM"] else int(
            os.environ.get("TRITONXPU_ATOMIC_SIM", 1))
        TTXPU_O_CLOSE_OPT = int(os.environ.get("TRITONXPU_CLOSE_OPTIMIZE", 0))

        pm = ir.pass_manager(mod.context)
        pm.enable_debug()

        xpu.passes.ttxpuir.add_convert_triton_to_tritonxpu_pass(pm, opt.arch, XPUBackend.buffer_len, core_num)
        xpu.passes.ttxpuir.add_tritonxpu_gm2lm_pass(pm, opt.arch, TTXPU_O_ATOMIC_SIM)
        passes.common.add_canonicalizer(pm)
        if TTXPU_F_DTYPE_CONVERT:
            xpu.passes.ttxpuir.add_tritonxpu_dtype_convert_pass(pm, opt.arch)
        if not metadata["isCloseCoreTiling"]:
            xpu.passes.ttxpuir.add_tritonxpu_core_tiling_pass(
                pm, 0, XPUBackend.buffer_len) if not TTXPU_O_CLOSE_OPT else None  # dumpFlag=0
        # xpu.passes.ttxpuir.add_tritonxpu_lm_to_sm_pass(pm)
        if not metadata["isCloseOffsetAnalysis"]:
            xpu.passes.ttxpuir.add_tritonxpu_offset_state_pass(pm, 0) if not TTXPU_O_CLOSE_OPT else None  # dumpFlag=0
        passes.common.add_canonicalizer(pm)
        xpu.passes.ttxpuir.add_tritonxpu_legalize_pass(pm, XPUBackend.buffer_len, core_num)
        if not TTXPU_F_OHTER_VALUE_SIM:
            xpu.passes.ttxpuir.add_tritonxpu_mask_pass(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        passes.common.add_licm(pm)
        passes.common.add_symbol_dce(pm)
        xpu.passes.ttxpuir.add_tritonxpu_interleave_pass(pm) if not TTXPU_O_CLOSE_OPT else None
        # xpu.passes.ttxpuir.add_tritonxpu_interleave_mask_pass(pm)
        passes.common.add_canonicalizer(pm)
        xpu.passes.ttxpuir.add_tritonxpu_vectorize_pass(pm, 0) if not TTXPU_O_CLOSE_OPT else None  # dumpFlag=0
        xpu.passes.ttxpuir.add_tritonxpu_alloca_pass(pm)
        if not TTXPU_F_OHTER_VALUE_SIM:
            xpu.passes.ttxpuir.add_tritonxpu_other_sim_pass(pm)
        xpu.passes.ttxpuir.add_tritonxpu_memory_async_pass(pm, 0) if not TTXPU_O_CLOSE_OPT else None  # dumpFlag=0
        if not metadata["isCloseUnrollControl"]:
            xpu.passes.ttxpuir.add_tritonxpu_unroll_control_pass(pm) if not TTXPU_O_CLOSE_OPT else None
        xpu.passes.ttxpuir.add_tritonxpu_store_control_pass(pm) if not TTXPU_O_CLOSE_OPT else None
        xpu.passes.ttxpuir.add_tritonxpu_loop_grid_pass(pm)
        passes.common.add_cse(pm)
        passes.common.add_licm(pm)
        passes.common.add_symbol_dce(pm)

        pm.run(mod)
        return mod

    @staticmethod
    def make_llir(mod, metadata, opt):
        # TritonXPU -> LLVM-IR (MLIR)
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        # xpu.passes.ttxpuir.add_decompose_unsupported_conversions(pm, opt.arch)
        passes.convert.add_scf_to_cf(pm)  # cf->llvm exist  choose scf->cf->llvm
        # passes.convert.add_index_to_llvmir(pm) // TODO[dyq]: necessary?

        passes.ttgpuir.add_allocate_shared_memory(pm)
        xpu.passes.ttxpuir.add_convert_tritonxpu_to_llvm_pass(pm, opt.arch, XPUBackend.buffer_len)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)

        # passes.convert.add_cf_to_llvmir(pm)
        # passes.convert.add_arith_to_llvmir(pm)
        # passes.common.add_canonicalizer(pm)
        # passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        if os.environ.get("TRITON_DISABLE_LINE_INFO", "0") == "0":
            passes.llvmir.add_di_scope(pm)

        pm.run(mod)

        # LLVM-IR (MLIR) -> LLVM-IR (LLVM)
        llvm.init_targets()
        context = llvm.context()
        llvm_mod = llvm.to_module(mod, context)

        if opt.extern_libs:
            paths = [path for (name, path) in opt.extern_libs if xpu.llvm.need_extern_lib(mod)]
            assert (len(paths) <= 1), f"Expected 0/1 extern_lib path, but found {len(paths)}"
            llvm.link_extern_libs(llvm_mod, paths)

        llvm.optimize_module(llvm_mod, llvm.OPTIMIZE_O3, f"xpu{opt.arch}")
        xpu.llvm.amend_func(llvm_mod, mod, context, opt.arch)

        del context
        return llvm_mod

    @staticmethod
    def make_elf(mod, metadata, opt):
        # Find kernel names (there should only be one)
        # We get the name at the last possible step to accomodate `triton.compile`
        # on user-provided LLVM
        metadata["name"] = xpu.llvm.get_kernel_name(mod)

        # llvm -> elf/asm
        triple = f"xpu{opt.arch}-baidu-none-gnu"
        proc = f"xpu{opt.arch}"
        flags = ["xpu-cmp-nan"] if metadata["isOpenCmpNan"] else []
        ret_asm = xpu.llvm.translate_to_asm(mod, triple, proc, "", flags, False, False)
        fn_cache_manager = get_cache_manager(metadata["hash"])
        fn_cache_manager.put(ret_asm, f"{metadata['name']}.asm")
        ret_elf = xpu.llvm.translate_to_asm(mod, triple, proc, "", [], False, True)

        del mod
        return ret_elf

    @staticmethod
    def make_xpubin(mod, metadata, opt):
        with tempfile.TemporaryDirectory() as tmpdir:
            clang_path = XPUBackend.path_to_xpu_compile_tool(opt)
            elfconv = os.path.join(Path(__file__).parent, f"xpu{opt.arch}-elfconv")
            objfile = os.path.join(tmpdir, "kernel.o")
            binfile = os.path.join(tmpdir, "kernel.bin")
            with open(objfile, "wb") as f:
                f.write(mod)
            cmd = ["bash", elfconv, objfile, binfile, clang_path]
            out = subprocess.run(cmd, check=True, capture_output=True)
            printf_buf_offset_res = re.search(rb"0x[0-9a-fA-F]+", out.stdout)
            if printf_buf_offset_res:
                printf_buf_offset_hex = printf_buf_offset_res.group(0)
                printf_buf_offset_hex_str = printf_buf_offset_hex.decode("utf-8")
                printf_buf_offset = int(printf_buf_offset_hex_str, 16)
            else:
                printf_buf_offset = 0
            metadata["printf_buf_offset"] = printf_buf_offset
            with open(binfile, "rb") as f:
                return f.read()

    @staticmethod
    def is_elf_stack_size_oob(mod) -> bool:
        stack_size_oob = llvm.is_elf_stack_size_oob(mod)
        return stack_size_oob

    def hash(self) -> str:
        """Returns a unique identifier for this backend"""
        # TODO:
        return f"1"

    def parse_options(self, options: dict) -> object:
        args = {"arch": self.target.arch}
        args.update({k: options[k] for k in XPUOptions.__dataclass_fields__.keys() if k in options})
        return XPUOptions(**args)

    def add_stages(self, stages: dict, options: object) -> None:
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["ttxir"] = lambda src, metadata: self.make_ttxir(src, metadata, options)
        stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options)
        stages["elf"] = lambda src, metadata: self.make_elf(src, metadata, options)
        stages["xpubin"] = lambda src, metadata: self.make_xpubin(src, metadata, options)

    def pack_metadata(self, metadata):
        return (
            metadata.cluster_dims[0],
            metadata.cluster_dims[1],
            metadata.cluster_dims[2],
        )

    def get_codegen_implementation(self):
        codegen_fns = dict()
        return codegen_fns

    def load_dialects(self, context):
        xpu.load_dialects(context)
