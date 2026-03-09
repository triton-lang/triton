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
        open('/tmp/raw_pre.ll', 'w').write(str(llvm_mod))
        return llvm_mod

    @staticmethod
    def _add_air_metadata(llvm_ir: str, arch: str = "apple_m1") -> str:
        """Inject required AIR/Metal module metadata into LLVM IR text.

        metal-as requires thread_position_in_grid as explicit kernel args (not
        intrinsic calls). Strategy: use NAMED SSA args (%tid_x etc.) appended
        to the signature — named args coexist with numbered body SSAs, so no
        renumbering needed.

        Steps:
          1. Set target datalayout + triple
          2. Add %tid_x/y/z (and %simdlane if needed) as named args
          3. Strip air.thread_position_in_grid call+extractvalue → replace uses with %tid_x
          4. Strip air.thread_index_in_simdgroup call → replace uses with %simdlane
          5. Strip the intrinsic declarations
          6. Emit !air.kernel metadata for all args
        """
        import re as _re

        air_header = (
            'target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32'
            '-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32'
            '-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256'
            '-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"\n'
            'target triple = "air64-apple-macosx26.0.0"\n\n'
        )

        # Strip existing target lines and air intrinsic declarations
        lines = llvm_ir.split('\n')
        lines = [l for l in lines
                 if not l.startswith('target datalayout')
                 and not l.startswith('target triple')
                 and 'declare' not in l or (
                     'air.thread_position_in_grid' not in l
                     and 'air.threadgroup_position_in_grid' not in l
                     and 'air.thread_index_in_simdgroup' not in l)]
        body = '\n'.join(lines)

        # Find the kernel function definition
        fn_match = _re.search(r'define void @(\w+)\(([^{]*)\)\s*\{', body)
        if not fn_match:
            return air_header + body + '\n!air.version = !{!1001}\n!1001 = !{i32 2, i32 8, i32 0}\n'

        fn_name = fn_match.group(1)
        orig_args_str = fn_match.group(2).strip()

        def _split_args(s):
            depth, current, result = 0, [], []
            for ch in s:
                if ch == '(': depth += 1
                elif ch == ')': depth -= 1
                if ch == ',' and depth == 0:
                    result.append(''.join(current).strip())
                    current = []
                else:
                    current.append(ch)
            if current:
                result.append(''.join(current).strip())
            return [a for a in result if a]

        raw_args = _split_args(orig_args_str)

        def _arg_type(a):
            return _re.sub(r'\s+%\w+$', '', a).strip()

        orig_types = [_arg_type(a) for a in raw_args]
        n_orig = len(orig_types)
        has_simdlane = 'air.thread_index_in_simdgroup' in llvm_ir
        has_pid = 'air.threadgroup_position_in_grid' in llvm_ir

        # Named pid/tid/simdlane args appended — no collision with numbered body SSAs
        pid_args = ['i32 %pid_x', 'i32 %pid_y', 'i32 %pid_z'] if has_pid else []
        tid_args = ['i32 %tid_x', 'i32 %tid_y', 'i32 %tid_z']
        if has_simdlane:
            tid_args.append('i32 %simdlane')

        new_args_str = (orig_args_str.rstrip()
                        + (', ' if orig_args_str.strip() else '')
                        + ', '.join(pid_args + tid_args))
        old_def = fn_match.group(0)
        new_def = f'define void @{fn_name}({new_args_str}) {{'
        body = body.replace(old_def, new_def, 1)

        # Strip air.threadgroup_position_in_grid calls → replace with %pid_x/y/z
        pid_struct_ssas = set()
        if has_pid:
            def _strip_pid_call(m):
                pid_struct_ssas.add(m.group(1))
                return ''
            body = _re.sub(
                r'[ \t]*%(\w+)\s*=\s*(?:tail\s+)?call\s+\[3 x i32\]\s+@air\.threadgroup_position_in_grid\s*\(\s*\)[^\n]*\n?',
                _strip_pid_call, body)

        # Strip air.thread_position_in_grid call lines; collect struct SSA name
        tid_struct_ssas = set()
        def _strip_tid_call(m):
            tid_struct_ssas.add(m.group(1))
            return ''
        body = _re.sub(
            r'[ \t]*%(\w+)\s*=\s*(?:tail\s+)?call\s+\[3 x i32\]\s+@air\.thread_position_in_grid\s*\(\s*\)[^\n]*\n?',
            _strip_tid_call, body)

        # Strip extractvalue [3 x i32] %struct, N → replace with %pid_x/y/z or %tid_x/y/z
        pid_names = ['%pid_x', '%pid_y', '%pid_z']
        tid_names = ['%tid_x', '%tid_y', '%tid_z']
        def _strip_ev(m):
            dest, struct, axis = m.group(1), m.group(2), int(m.group(3))
            if struct in pid_struct_ssas:
                _ev_map[dest] = pid_names[axis]
            elif struct in tid_struct_ssas:
                _ev_map[dest] = tid_names[axis]
            return ''
        _ev_map = {}
        body = _re.sub(
            r'[ \t]*%(\w+)\s*=\s*extractvalue\s+\[3 x i32\]\s+%(\w+),\s*(\d)[^\n]*\n?',
            _strip_ev, body)
        for dest, replacement in _ev_map.items():
            body = _re.sub(r'%' + dest + r'\b', replacement, body)

        # Strip air.thread_index_in_simdgroup call → replace with %simdlane
        if has_simdlane:
            def _strip_simdlane(m):
                dest = m.group(1)
                _sl_map[dest] = '%simdlane'
                return ''
            _sl_map = {}
            body = _re.sub(
                r'[ \t]*%(\w+)\s*=\s*(?:tail\s+)?call\s+(?:i32|noundef\s+i32)\s+'
                r'@air\.thread_index_in_simdgroup\s*\([^)]*\)[^\n]*\n?',
                _strip_simdlane, body)
            for dest, replacement in _sl_map.items():
                body = _re.sub(r'%' + dest + r'\b', replacement, body)

        # Scalar args (i32, i64, etc.) must be passed as addrspace(2)* (constant buffer pointer)
        # and explicitly loaded in the function body. The Metal runtime uses setBytes for
        # scalars, which passes a pointer to a 4-byte constant buffer.

        # Identify scalar args: numbered SSAs with non-pointer types
        scalar_arg_ssas = {}  # '%N' → elem_type ('i32', etc.)
        for a in raw_args:
            parts = a.split()
            if len(parts) < 2: continue
            ssa = parts[-1]
            if not (ssa.startswith('%') and ssa[1:].isdigit()): continue
            ty = ' '.join(parts[:-1]).strip()
            if 'addrspace' not in ty and '*' not in ty and ty in ('i32', 'i64', 'i16', 'i8', 'i1'):
                scalar_arg_ssas[ssa] = ty

        if scalar_arg_ssas:
            # Rewrite define signature: `ty %N` → `ty addrspace(2)* %N`
            for ssa, ty in scalar_arg_ssas.items():
                body = body.replace(f'{ty} {ssa}', f'{ty} addrspace(2)* {ssa}', 1)
            # Update orig_types for metadata generation below
            for i, a in enumerate(raw_args):
                parts = a.split()
                if len(parts) < 2: continue
                ssa = parts[-1]
                if ssa in scalar_arg_ssas:
                    orig_types[i] = f'{scalar_arg_ssas[ssa]} addrspace(2)*'

        # Split body into sig_part + fn_body for SSA renumbering
        fn_body_start = body.index(f'define void @{fn_name}(')
        fn_body_start = body.index('{', fn_body_start) + 1
        sig_part = body[:fn_body_start]
        fn_body = body[fn_body_start:]

        if scalar_arg_ssas:
            # First: replace all uses of scalar SSAs in the original fn_body
            # (before adding load lines) with placeholder named SSAs.
            # This avoids load-line self-referencing issues.
            for ssa, ty in scalar_arg_ssas.items():
                val_ssa = f'%__scval_{ssa[1:]}__'
                # Replace %N as operand (word-boundary match):
                fn_body = _re.sub(
                    r'(?<!\w)' + _re.escape(ssa) + r'(?!\w)',
                    val_ssa, fn_body)
            # Then: prepend load instructions at start of fn_body that load from
            # the original arg pointer (%N, still intact in the define signature).
            load_lines = []
            for ssa, ty in scalar_arg_ssas.items():
                val_ssa = f'%__scval_{ssa[1:]}__'
                load_lines.append(f'  {val_ssa} = load {ty}, {ty} addrspace(2)* {ssa}, align 4')
            first_nl = fn_body.index('\n') + 1
            fn_body = fn_body[:first_nl] + '\n'.join(load_lines) + '\n' + fn_body[first_nl:]

        # Renumber body SSAs so they are contiguous starting at n_orig.
        # After stripping call+extractvalue lines, body SSAs have gaps.
        # metal-as requires sequential numbering; named args don't count.
        # Two-pass: old → placeholder → new (avoids cascading substitution).

        defined_ssas = sorted({int(m) for m in _re.findall(r'%(\d+)\s*=', fn_body)})
        if defined_ssas:
            # Named args don't consume numbered slots; implicit entry BB label uses %n_orig
            new_start = n_orig + 1
            ssa_map = {old: new_start + i for i, old in enumerate(defined_ssas)}
            for old in defined_ssas:
                fn_body = _re.sub(r'%' + str(old) + r'\b', f'%__p{old}__', fn_body)
            for old, new in ssa_map.items():
                fn_body = fn_body.replace(f'%__p{old}__', f'%{new}')
        body = sig_part + fn_body

        # Build full arg type list for metadata (ptr addrspace(N) → i8 addrspace(N)* for metadata)
        def _meta_ty(t):
            return _re.sub(r'\bptr\s+addrspace\((\d+)\)', r'i8 addrspace(\1)*', t)
        n_pid_start = n_orig
        n_tid_start = n_orig + (3 if has_pid else 0)
        all_types = [_meta_ty(t) for t in orig_types]
        if has_pid:
            all_types += ['i32', 'i32', 'i32']
        all_types += ['i32', 'i32', 'i32']
        if has_simdlane:
            all_types.append('i32')
        types_str = ', '.join(all_types)

        # Per-arg metadata
        arg_nodes = []
        arg_meta_ids = []
        for idx, atype in enumerate(all_types):
            node_id = 2000 + idx
            arg_meta_ids.append(node_id)
            if has_simdlane and idx == len(all_types) - 1:
                arg_nodes.append(
                    f'!{node_id} = !{{i32 {idx}, !"air.thread_index_in_simdgroup", '
                    f'!"air.arg_type_name", !"uint", !"air.arg_name", !"simdlane"}}'
                )
            elif idx >= n_tid_start and idx < n_tid_start + 3:
                dim = ['x', 'y', 'z'][idx - n_tid_start]
                arg_nodes.append(
                    f'!{node_id} = !{{i32 {idx}, !"air.thread_position_in_grid", '
                    f'!"air.arg_type_name", !"uint", !"air.arg_name", !"tid_{dim}"}}'
                )
            elif has_pid and idx >= n_pid_start and idx < n_pid_start + 3:
                dim = ['x', 'y', 'z'][idx - n_pid_start]
                arg_nodes.append(
                    f'!{node_id} = !{{i32 {idx}, !"air.threadgroup_position_in_grid", '
                    f'!"air.arg_type_name", !"uint", !"air.arg_name", !"pid_{dim}"}}'
                )
            elif 'addrspace(1)' in atype or ('ptr' in atype and 'addrspace' not in atype):
                arg_nodes.append(
                    f'!{node_id} = !{{i32 {idx}, !"air.buffer", '
                    f'!"air.location_index", i32 {idx}, i32 1, '
                    f'!"air.read_write", !"air.address_space", i32 1, '
                    f'!"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, '
                    f'!"air.arg_type_name", !"float", !"air.arg_name", !"arg{idx}"}}'
                )
            else:
                # Plain scalar (i32, i64, etc.) — passed as constant buffer (address_space=2)
                arg_nodes.append(
                    f'!{node_id} = !{{i32 {idx}, !"air.buffer", '
                    f'!"air.buffer_size", i32 4, '
                    f'!"air.location_index", i32 {idx}, i32 1, '
                    f'!"air.read", !"air.address_space", i32 2, '
                    f'!"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, '
                    f'!"air.arg_type_name", !"uint", !"air.arg_name", !"arg{idx}"}}'
                )

        ids_str = ', '.join(f'!{i}' for i in arg_meta_ids)
        air_footer = (
            f'\n!air.kernel = !{{!1999}}\n'
            f'!air.version = !{{!1001}}\n'
            f'!air.language_version = !{{!1002}}\n'
            f'!1001 = !{{i32 2, i32 8, i32 0}}\n'
            f'!1002 = !{{!"Metal", i32 3, i32 2, i32 0}}\n'
            f'!1999 = !{{void ({types_str})* @{fn_name}, !1998, !1997}}\n'
            f'!1998 = !{{}}\n'
            f'!1997 = !{{{ids_str}}}\n'
            + '\n'.join(arg_nodes) + '\n'
        )

        return air_header + body + air_footer

    # ── Stage 4: LLVM IR → metallib ───────────────────────────────────────
    def make_metallib(self, llvm_mod, metadata, options):
        import re as _re

        # Emit LLVM IR text
        llvm_ir_raw = str(llvm_mod)

        # Extract kernel name from IR (first void function = kernel entry point)
        names = _re.findall(r'define void @([a-zA-Z_][a-zA-Z0-9_]*)', llvm_ir_raw)
        if names:
            metadata["name"] = names[0]

        # Strip LLVM 17+ syntax unsupported by metal-as
        llvm_ir_raw = llvm_ir_raw.replace('or disjoint', 'or')
        # Expand `<N x T> splat (T val)` → `<N x T> <T val, ...>` (N times)
        def _expand_splat(m):
            n, ty, val = int(m.group(1)), m.group(2), m.group(3)
            return f'<{n} x {ty}> <' + ', '.join([f'{ty} {val}'] * n) + '>'
        llvm_ir_raw = _re.sub(r'<(\d+) x (\w+)> splat \(\w+ (\w+)\)', _expand_splat, llvm_ir_raw)

        # ── Simdgroup MMA: typed pointers + fast + convergent (before global ptr rewrite) ──
        llvm_ir_raw = llvm_ir_raw.replace(
            'declare <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3),',
            'declare <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(float addrspace(3)* nocapture readonly,')
        llvm_ir_raw = llvm_ir_raw.replace(
            'declare void @air.simdgroup_matrix_8x8_store.v64f32.p3f32(<64 x float>, ptr addrspace(3),',
            'declare void @air.simdgroup_matrix_8x8_store.v64f32.p3f32(<64 x float>, float addrspace(3)* nocapture writeonly,')
        llvm_ir_raw = _re.sub(
            r'call (<64 x float>) @(air\.simdgroup_matrix_8x8_load\.v64f32\.p3f32)\(ptr addrspace\(3\) ',
            r'call fast \1 @\2(float addrspace(3)* ', llvm_ir_raw)
        llvm_ir_raw = _re.sub(
            r'call void @(air\.simdgroup_matrix_8x8_store\.v64f32\.p3f32)\(<64 x float> (.*?), ptr addrspace\(3\) ',
            r'call void @\1(<64 x float> \2, float addrspace(3)* ', llvm_ir_raw)
        llvm_ir_raw = llvm_ir_raw.replace(
            'call <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate',
            'call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate')

        # ── TG globals: fix GEPs and inject base SSAs (before global ptr rewrite) ──
        tg_global_sizes = {}
        for gm in _re.finditer(
                r'(@\w+)\s*=\s*internal addrspace\(3\) global \[(\d+) x float\]', llvm_ir_raw):
            tg_global_sizes[gm.group(1)] = int(gm.group(2))
        if tg_global_sizes:
            def _fix_tg_gep(m):
                gname, idx = m.group(1), m.group(2)
                n = tg_global_sizes.get(gname)
                if n is None: return m.group(0)
                return f'getelementptr [{n} x float], [{n} x float] addrspace(3)* {gname}, i64 0, i64 {idx}'
            llvm_ir_raw = _re.sub(
                r'getelementptr float, ptr addrspace\(3\) (@\w+), i64 (%\w+|\d+)',
                _fix_tg_gep, llvm_ir_raw)
            base_ssas, preamble_lines = {}, []
            for gname, n in tg_global_sizes.items():
                ssa = f'%__base_{gname.lstrip("@")}'
                base_ssas[gname] = ssa
                preamble_lines.append(
                    f'  {ssa} = getelementptr [{n} x float], [{n} x float] addrspace(3)* {gname}, i64 0, i64 0')
            preamble = '\n'.join(preamble_lines) + '\n'
            llvm_ir_raw = _re.sub(r'(define [^\n]+\{)\n',
                lambda m: m.group(0) + preamble, llvm_ir_raw, count=1)
            for gname, ssa in base_ssas.items():
                llvm_ir_raw = llvm_ir_raw.replace(f'float addrspace(3)* {gname}', f'float addrspace(3)* {ssa}')

        # ── air.threadgroup.barrier → air.wg.barrier ──
        llvm_ir_raw = llvm_ir_raw.replace(
            'declare void @air.threadgroup.barrier(i32, i32)',
            'declare void @air.wg.barrier(i32, i32)')
        llvm_ir_raw = llvm_ir_raw.replace(
            'call void @air.threadgroup.barrier(i32 1, i32 4)',
            'call void @air.wg.barrier(i32 2, i32 1)')

        # ── Global opaque ptr → typed pointer (metal-as uses LLVM 14 syntax) ──
        # Build a map: SSA name → element type, from GEP/load/store uses.
        # Then rewrite all ptr addrspace(N) uses consistently.
        ptr_elem_types = {}  # '%arg' → 'float', 'i8', etc.
        for m in _re.finditer(
                r'\b(?:getelementptr|load)\s+(\S+),\s+ptr\s+addrspace\(\d+\)\s+(%\w+)', llvm_ir_raw):
            ptr_elem_types[m.group(2)] = m.group(1)
        for m in _re.finditer(
                r'\bstore\s+(\S+)\s+[^,]+,\s+ptr\s+addrspace\(\d+\)\s+(%\w+)', llvm_ir_raw):
            ptr_elem_types[m.group(2)] = m.group(1)

        def _ptr_ty(ssa, addrspace):
            elem = ptr_elem_types.get(ssa, 'i8')
            return f'{elem} addrspace({addrspace})*'

        # Rewrite GEP/load/store to use typed pointer matching element type
        llvm_ir_raw = _re.sub(
            r'\bgetelementptr\s+(\S+),\s+ptr\s+addrspace\((\d+)\)\s+(%\w+)',
            lambda m: f'getelementptr {m.group(1)}, {_ptr_ty(m.group(3), m.group(2))} {m.group(3)}',
            llvm_ir_raw)
        llvm_ir_raw = _re.sub(
            r'\bload\s+(\S+),\s+ptr\s+addrspace\((\d+)\)\s+(%\w+)',
            lambda m: f'load {m.group(1)}, {_ptr_ty(m.group(3), m.group(2))} {m.group(3)}',
            llvm_ir_raw)
        llvm_ir_raw = _re.sub(
            r'\bstore\s+(\S+)\s+([^,]+),\s+ptr\s+addrspace\((\d+)\)\s+(%\w+)',
            lambda m: f'store {m.group(1)} {m.group(2)}, {_ptr_ty(m.group(4), m.group(3))} {m.group(4)}',
            llvm_ir_raw)
        # Rewrite function arg declarations: `ptr addrspace(N) %arg` → `<elem_ty> addrspace(N)* %arg`
        def _rewrite_arg(m):
            addrspace, ssa = m.group(1), m.group(2)
            elem = ptr_elem_types.get(ssa, 'i8')
            return f'{elem} addrspace({addrspace})* {ssa}'
        llvm_ir_raw = _re.sub(r'\bptr\s+addrspace\((\d+)\)\s+(%\w+)', _rewrite_arg, llvm_ir_raw)
        # Remaining bare ptr addrspace(N) (no following SSA) → i8 addrspace(N)*
        llvm_ir_raw = _re.sub(r'\bptr\s+addrspace\((\d+)\)', r'i8 addrspace(\1)*', llvm_ir_raw)
        llvm_ir_raw = _re.sub(r'\bptr\b(?!\s+addrspace)', 'i8*', llvm_ir_raw)

        # Strip dead i8* args: pointer args that were unknown type (→ i8*) and
        # are never referenced in the function body. These are Triton-internal
        # mask/other pointer args that our lowering already inlined as selects.
        def _strip_dead_i8_args(ir):
            fn_m = _re.search(r'(define void @\w+)\(([^{]*)\)\s*\{', ir)
            if not fn_m:
                return ir
            args_str = fn_m.group(2)
            # Parse args
            def _split_args(s):
                depth, cur, res = 0, [], []
                for ch in s:
                    if ch in '([': depth += 1
                    elif ch in ')]': depth -= 1
                    if ch == ',' and depth == 0:
                        res.append(''.join(cur).strip()); cur = []
                    else:
                        cur.append(ch)
                if cur: res.append(''.join(cur).strip())
                return [a for a in res if a]
            args = _split_args(args_str)
            # Find the function body (after the opening {)
            body_start = ir.index('{', fn_m.start()) + 1
            body = ir[body_start:]
            kept = []
            for arg in args:
                # Extract SSA name (last token)
                ssa = arg.split()[-1]  # e.g. '%3'
                if not ssa.startswith('%'):
                    kept.append(arg); continue
                # Keep if it's a named arg (tid_x etc.) or used in body
                if not ssa[1:].isdigit():
                    kept.append(arg); continue
                # Check if %N appears in the body (as operand, not definition like "%N =")
                # Use negative lookbehind for "=" to exclude the definition itself
                if _re.search(_re.escape(ssa) + r'(?!\s*=)(?=[\s,)\]])', body):
                    kept.append(arg)
                # else: dead arg — drop it
            new_args = ', '.join(kept)
            old_def = fn_m.group(0)
            new_def = f'{fn_m.group(1)}({new_args}) {{'
            return ir.replace(old_def, new_def, 1)
        # Store IR-slot positions for scalar (non-ptr) args so the launcher can
        # build correct arg_casts. IR slot = position in define() arglist before tid_*.
        # We parse the IR signature before _add_air_metadata renumbers things.
        _fn_sig_m = _re.search(r'define void @\w+\(([^{]*)\)\s*\{', llvm_ir_raw)
        if _fn_sig_m:
            _raw = _fn_sig_m.group(1)
            _ir_args = [a.strip() for a in _raw.split(',') if a.strip() and 'tid_' not in a and 'pid_' not in a and 'simdlane' not in a]
            _scalar_ir_slots = {i for i, a in enumerate(_ir_args) if not any(x in a for x in ('addrspace', '*'))}
            metadata['scalar_ir_slots'] = sorted(_scalar_ir_slots)

        llvm_ir = self._add_air_metadata(llvm_ir_raw, options.arch)
        open('/tmp/dot_kernel_final.ll', 'w').write(llvm_ir)

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
    def add_stages(self, stages, options, language):
        stages["ttir"]     = lambda src, meta: self.make_ttir(src, meta, options)
        stages["ttgir"]    = lambda src, meta: self.make_ttgir(src, meta, options)
        stages["llir"]     = lambda src, meta: self.make_llir(src, meta, options)
        stages["metallib"] = lambda src, meta: self.make_metallib(src, meta, options)
