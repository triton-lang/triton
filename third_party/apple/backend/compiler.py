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
  [ ] Build + fix compile errors
  [ ] Metal IR emission (simdgroup intrinsic → .ll)
  [ ] MetalASM integration for in-process metallib
  [ ] driver.py (MTLDevice dispatch)
"""

from dataclasses import dataclass
from typing import Any
import functools
import hashlib
import subprocess
import tempfile
import os

from triton.backends.compiler import BaseBackend, GPUTarget
from triton._C.libtriton import ir, passes, llvm


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
        # lazy import — only available after build
        try:
            import triton_apple as _apple
            self._apple = _apple
        except ImportError:
            self._apple = None

    def parse_options(self, opts) -> MPSOptions:
        args = {k: opts[k] for k in MPSOptions.__dataclass_fields__ if k in opts}
        return MPSOptions(**args)

    def pack_metadata(self, metadata):
        return metadata

    def get_codegen_implementation(self):
        raise NotImplementedError

    def get_module_map(self):
        return {}

    def load_dialects(self, ctx):
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
        pm.run(mod)
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
        passes.ttgpuir.add_optimize_dot_operands(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)

        pm.run(mod)
        return mod

    # ── Stage 3: LLVM IR with simdgroup intrinsics ─────────────────────────
    def make_llir(self, mod, metadata, options):
        if not self._apple:
            raise RuntimeError("triton_apple not built")

        pm = ir.pass_manager(mod.context)
        pm.enable_debug()

        # Apple-specific: AppleMmaEncoding → simdgroup_multiply_accumulate calls
        self._apple.passes.ttgpuir.add_to_llvmir(pm)

        # Shared cleanup
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        pm.run(mod)

        # Convert MLIR LLVM dialect → LLVM module
        llvm.init_targets()
        context = llvm.context()
        llvm_mod = llvm.to_module(mod, context)
        return llvm_mod

    # ── Stage 4: LLVM IR → metallib ───────────────────────────────────────
    def make_metallib(self, llvm_mod, metadata, options):
        # Emit LLVM IR text
        llvm_ir = str(llvm_mod)

        # Try MetalASM in-process first (fast, no subprocess)
        if self._apple:
            try:
                return self._apple.metal.compile_metal_ir(llvm_ir, options.arch)
            except RuntimeError:
                pass  # fall through to xcrun

        # Fallback: xcrun metal-as + metallib subprocess
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

    def add_stages(self, stages, options):
        stages["ttir"]     = lambda src, meta: self.make_ttir(src, meta, options)
        stages["ttgir"]    = lambda src, meta: self.make_ttgir(src, meta, options)
        stages["llir"]     = lambda src, meta: self.make_llir(src, meta, options)
        stages["metallib"] = lambda src, meta: self.make_metallib(src, meta, options)
