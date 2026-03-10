"""
Apple MPS Triton backend.

Pipeline:
  Triton Python (@triton.jit kernel)
    → make_ttir   (shared Triton passes)
    → make_ttgir  (Apple MMA tiling via AccelerateAppleMatmul)
    → make_llir   (LLVM IR with simdgroup intrinsics)
    → make_metallib (xcrun metal-as + metallib OR MetalASM in-process)
    → dispatch via MTLComputeCommandEncoder

Status:
  [x] LinearLayout verified + implemented
  [x] AppleMmaEncodingAttr defined
  [x] AccelerateAppleMatmul pass written
  [x] DotOpToLLVM skeleton
  [x] Python bindings skeleton
  [x] Build + fix compile errors
  [x] Metal IR emission (simdgroup intrinsic → .ll)
  [x] metal-as + metallib integration
  [x] MetalASM integration for in-process metallib
  [x] driver.py (MTLDevice dispatch)
"""

from dataclasses import dataclass
from typing import Any
import ctypes
import functools
import hashlib
import subprocess
import tempfile
import os

from triton.backends.compiler import BaseBackend, GPUTarget
from triton._C.libtriton import ir, passes, llvm


def _load_metalasm():
    """Load MetalASMBridge dylib and return a compile function, or None."""
    dylib = os.path.expanduser(
        '~/projects/oss/llvm/.build/release/libMetalASMBridge.dylib')
    if not os.path.exists(dylib):
        return None
    try:
        lib = ctypes.CDLL(dylib)
        lib.metalasm_compile.restype  = ctypes.c_void_p
        lib.metalasm_compile.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.metalasm_free.argtypes = [ctypes.c_void_p]

        def compile_ir(llvm_ir: str) -> bytes:
            out_len = ctypes.c_uint64(0)
            errbuf  = ctypes.create_string_buffer(512)
            ptr = lib.metalasm_compile(
                llvm_ir.encode(), ctypes.byref(out_len), errbuf, 512)
            if ptr:
                data = bytes((ctypes.c_ubyte * out_len.value).from_address(ptr))
                lib.metalasm_free(ptr)
                return data
            raise RuntimeError(f"MetalASM: {errbuf.value.decode()}")

        return compile_ir
    except Exception:
        return None


_metalasm_compile = _load_metalasm()


@dataclass(frozen=True)
class MPSOptions:
    num_warps: int = 4
    num_stages: int = 2
    num_ctas: int = 1
    arch: str = "apple_m1"
    backend_name: str = "mps"

    # simdgroup tile — fixed by hardware
    simdgroup_m: int = 8
    simdgroup_n: int = 8
    simdgroup_k: int = 8

    # Standard Triton options (accepted but largely unused on MPS)
    debug: bool = False
    enable_fp_fusion: bool = True
    launch_cooperative_grid: bool = False
    instrumentation_mode: str = "none"
    sanitize_overflow: bool = False
    allowed_dot_input_precisions: tuple = ("ieee",)

    def hash(self):
        return hashlib.md5(
            str(self.__dict__).encode()
        ).hexdigest()


class MPSBackend(BaseBackend):

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == "mps"

    def __init__(self, target: GPUTarget):
        super().__init__(target)
        self.target = target
        self.binary_ext = "metallib"
        # The apple backend is a submodule of libtriton, not a standalone .so
        try:
            from triton._C.libtriton import apple as _apple
            self._apple = _apple
        except ImportError:
            self._apple = None

    def parse_options(self, opts) -> MPSOptions:
        args = {k: opts[k] for k in MPSOptions.__dataclass_fields__ if k in opts}
        return MPSOptions(**args)

    def pack_metadata(self, metadata):
        return metadata

    def get_codegen_implementation(self, options):
        def min_dot_size(lhs_type, rhs_type):
            # Apple simdgroup tile is 8×8; minimum dot operand = (1, 1, 8)
            return (1, 1, 8)
        return {"min_dot_size": min_dot_size}

    def get_module_map(self):
        return {}

    def load_dialects(self, ctx):
        ir.load_dialects(ctx)
        if self._apple:
            self._apple.dialect.register_dialect(ctx)

    def hash(self):
        return "mps-v0.1"

    # ── Stage 1: Triton IR optimization (shared) ───────────────────────────
    def make_ttir(self, mod, metadata, options):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.ttir.add_combine(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.common.add_licm(pm)
        passes.common.add_symbol_dce(pm)
        pm.run(mod, 'make_ttir')
        return mod

    # ── Stage 2: GPU tiling — THE make-or-break ────────────────────────────
    def make_ttgir(self, mod, metadata, options):
        if not self._apple:
            raise RuntimeError("triton_apple not built — run: pip install -e . from triton root")

        pm = ir.pass_manager(mod.context)
        pm.enable_debug()

        # Convert generic TritonIR → TritonGPU IR (shared pass)
        passes.ttir.add_convert_to_ttgpuir(
            pm, f"mps:{options.arch}", options.num_warps,
            32,  # warp_size = 32 (simdgroup size)
            options.num_ctas)

        # Shared layout optimization passes
        passes.ttgpuir.add_coalesce(pm)
        passes.ttgpuir.add_remove_layout_conversions(pm)

        # THE Apple-specific pass: BlockedEncoding → AppleMmaEncoding
        self._apple.passes.ttgpuir.add_accelerate_matmul(pm)

        # Clean up redundant layout conversions introduced by the rewrite
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_optimize_dot_operands(pm, True)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)

        pm.run(mod, 'make_ttgir')
        metadata["shared"] = mod.get_int_attr("ttg.shared") or 0
        return mod

    # ── Stage 3: LLVM IR with simdgroup intrinsics ─────────────────────────
    def make_llir(self, mod, metadata, options):
        if not self._apple:
            raise RuntimeError("triton_apple not built")

        pm = ir.pass_manager(mod.context)
        pm.enable_debug()

        # Standard TritonGPU → LLVM lowering prerequisites
        passes.convert.add_scf_to_cf(pm)
        passes.ttgpuir.add_allocate_shared_memory(pm)
        passes.convert.add_index_to_llvmir(pm)

        # Apple-specific: AppleMmaEncoding → simdgroup_multiply_accumulate calls
        self._apple.passes.ttgpuir.add_to_llvmir(pm)
        # Lower remaining gpu.thread_id/block_dim → air intrinsics/constants
        self._apple.passes.ttgpuir.add_lower_gpu_to_air(pm)
        self._apple.passes.ttgpuir.add_reconcile_unrealized_casts(pm)

        # Shared cleanup
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        pm.run(mod, 'make_llir')

        # Convert MLIR LLVM dialect → LLVM module
        llvm.init_targets()
        context = llvm.context()
        llvm_mod = llvm.to_module(mod, context)
        if os.environ.get('TRITON_MPS_DEBUG'):
            open('/tmp/raw_pre.ll', 'w').write(str(llvm_mod))
        return llvm_mod

    # ── Stage 4: LLVM IR → metallib ───────────────────────────────────────
    def make_metallib(self, llvm_mod, metadata, options):
        # Emit LLVM IR text
        llvm_ir = str(llvm_mod)

        # Extract kernel name (first defined void function = kernel entry)
        for line in llvm_ir.splitlines():
            if line.startswith('define void @'):
                metadata["name"] = line[len('define void @'):].split('(')[0]
                break

        debug = os.environ.get('TRITON_MPS_DEBUG')

        if debug:
            kname = metadata["name"]
            open(f'/tmp/dot_kernel_{kname}.ll', 'w').write(llvm_ir)

        # MetalASM: handles all AIR transforms (MMA ptrs, TG GEPs, barrier rename,
        # system-value lowering, !air.kernel metadata) in-process.
        if _metalasm_compile:
            result = _metalasm_compile(llvm_ir)
            if debug:
                open(f'/tmp/dot_kernel_{kname}.metallib', 'wb').write(result)
            return result

        # Fallback: xcrun metal-as + metallib (requires sequential SSA numbering)
        with tempfile.TemporaryDirectory() as tmp:
            ll_path  = os.path.join(tmp, "kernel.ll")
            air_path = os.path.join(tmp, "kernel.air")
            lib_path = os.path.join(tmp, "kernel.metallib")

            with open(ll_path, "w") as f:
                f.write(llvm_ir)

            subprocess.run(
                ["xcrun", "-sdk", "macosx", "metal-as", ll_path, "-o", air_path],
                check=True)
            subprocess.run(
                ["xcrun", "-sdk", "macosx", "metallib", air_path, "-o", lib_path],
                check=True)

            with open(lib_path, "rb") as f:
                return f.read()
    def add_stages(self, stages, options, language):
        stages["ttir"]     = lambda src, meta: self.make_ttir(src, meta, options)
        stages["ttgir"]    = lambda src, meta: self.make_ttgir(src, meta, options)
        stages["llir"]     = lambda src, meta: self.make_llir(src, meta, options)
        stages["metallib"] = lambda src, meta: self.make_metallib(src, meta, options)
