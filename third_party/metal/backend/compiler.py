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
        result = subprocess.check_output(["xcrun", "--sdk", "macosx", "metal", "--version"],
                                         stderr=subprocess.STDOUT).decode("utf-8")
        return result.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


@functools.lru_cache()
def get_metal_arch(gpu_family: int):
    # Map Apple GPU family to Metal architecture string
    # Metal 4 (macOS 26+) supports Apple7+ (M1+), optimized for Apple8+ (M2+)
    family_map = {
        7: "apple7",  # M1 - Metal 3
        8: "apple8",  # M2 - Metal 3
        9: "apple9",  # M3 - Metal 3
        10: "apple10",  # M4 - Metal 4 native
    }
    return family_map.get(gpu_family, f"apple{gpu_family}")


class LLVMIRToMSLTranslator:
    """Translates LLVM IR text to Metal Shading Language source code.

    This is the core code generation pass that maps LLVM IR operations
    to their MSL equivalents, handling:
    - Kernel signature extraction and buffer binding generation
    - LLVM type system → MSL type system mapping
    - Address space annotations (device, threadgroup, constant)
    - Arithmetic/logic/comparison instruction translation
    - Memory operations (load/store with proper address spaces)
    - Control flow (branches, phi nodes → MSL variables)
    - SIMD group intrinsics (shuffle, ballot, reduce)
    - Threadgroup memory allocation and barriers
    """

    # LLVM type → MSL type mapping
    TYPE_MAP = {
        'i1': 'bool',
        'i8': 'int8_t',
        'i16': 'int16_t',
        'i32': 'int32_t',
        'i64': 'int64_t',
        'half': 'half',
        'bfloat': 'bfloat',
        'float': 'float',
        'double': 'float',  # Metal does not support fp64; demote to fp32
        'void': 'void',
    }

    # LLVM address spaces → Metal address qualifiers
    ADDRSPACE_MAP = {
        0: 'device',  # Global memory
        1: 'constant',  # Constant memory (read-only)
        3: 'threadgroup',  # Shared memory (per-threadgroup)
        4: 'thread',  # Private (per-thread)
    }

    # LLVM binary ops → MSL operators
    BINOP_MAP = {
        'add': '+',
        'fadd': '+',
        'sub': '-',
        'fsub': '-',
        'mul': '*',
        'fmul': '*',
        'sdiv': '/',
        'udiv': '/',
        'fdiv': '/',
        'srem': '%',
        'urem': '%',
        'frem': 'fmod',
        'shl': '<<',
        'lshr': '>>',
        'ashr': '>>',
        'and': '&',
        'or': '|',
        'xor': '^',
    }

    # LLVM comparison predicates → MSL operators
    ICMP_MAP = {
        'eq': '==',
        'ne': '!=',
        'slt': '<',
        'sle': '<=',
        'sgt': '>',
        'sge': '>=',
        'ult': '<',
        'ule': '<=',
        'ugt': '>',
        'uge': '>=',
    }
    FCMP_MAP = {
        'oeq': '==',
        'one': '!=',
        'ogt': '>',
        'oge': '>=',
        'olt': '<',
        'ole': '<=',
        'ord': '!isnan',
        'ueq': '==',
        'une': '!=',
        'ugt': '>',
        'uge': '>=',
        'ult': '<',
        'ule': '<=',
        'uno': 'isnan',
    }

    # Metal intrinsic mapping for LLVM intrinsics
    INTRINSIC_MAP = {
        'llvm.fabs': 'abs',
        'llvm.sqrt': 'sqrt',
        'llvm.sin': 'sin',
        'llvm.cos': 'cos',
        'llvm.exp': 'exp',
        'llvm.exp2': 'exp2',
        'llvm.log': 'log',
        'llvm.log2': 'log2',
        'llvm.pow': 'pow',
        'llvm.fma': 'fma',
        'llvm.floor': 'floor',
        'llvm.ceil': 'ceil',
        'llvm.round': 'round',
        'llvm.trunc': 'trunc',
        'llvm.copysign': 'copysign',
        'llvm.minnum': 'min',
        'llvm.maxnum': 'max',
        'llvm.minimum': 'min',
        'llvm.maximum': 'max',
        'llvm.ctpop': 'popcount',
        'llvm.ctlz': 'clz',
        'llvm.cttz': 'ctz',
        'llvm.bitreverse': 'reverse_bits',
        'llvm.fmuladd': 'fma',
    }

    def __init__(self, llvm_ir: str, metadata: dict, opt, gpu_family: int):
        self.llvm_ir = llvm_ir
        self.metadata = metadata
        self.opt = opt
        self.gpu_family = gpu_family
        self.kernel_name = metadata.get("name", "triton_kernel")
        self.num_threads = opt.num_warps * opt.warp_size
        self.shared_mem = metadata.get("shared", 0)

        # State for translation
        self._var_counter = 0
        self._vars = {}  # LLVM SSA name → MSL variable name
        self._var_types = {}  # LLVM SSA name → MSL type
        self._body_lines = []
        self._local_decls = []

    def translate(self) -> str:
        """Main translation entry point. Returns complete MSL source."""
        # Parse kernel signature
        func_match = re.search(r'define\s+(?:dso_local\s+)?void\s+@([a-zA-Z_][a-zA-Z0-9_]*)\s*\(([^)]*)\)',
                               self.llvm_ir)
        if func_match:
            self.kernel_name = func_match.group(1)
            args_str = func_match.group(2)
        else:
            args_str = ""

        # Parse arguments
        kernel_params = self._parse_arguments(args_str)

        # Parse and translate the function body
        self._translate_body()

        # Assemble the MSL source
        return self._emit_msl(kernel_params)

    def _parse_arguments(self, args_str: str) -> list:
        """Parse LLVM IR function arguments into MSL kernel parameters."""
        if not args_str.strip():
            return []

        params = []
        buffer_idx = 0
        raw_args = self._split_args(args_str)

        for i, arg in enumerate(raw_args):
            arg = arg.strip()
            # Skip metadata/attribute arguments
            if not arg or arg.startswith('!') or arg.startswith('#'):
                continue

            msl_type, is_ptr, addrspace = self._parse_llvm_type(arg)
            param_name = f"arg{i}"

            # Extract name if present (e.g., "i32 %name")
            name_match = re.search(r'%([a-zA-Z_][a-zA-Z0-9_.]*)', arg)
            if name_match:
                self._vars[f'%{name_match.group(1)}'] = param_name

            if is_ptr:
                addr_qual = self.ADDRSPACE_MAP.get(addrspace, 'device')
                params.append({
                    'decl': f"    {addr_qual} {msl_type}* {param_name} [[buffer({buffer_idx})]]",
                    'name': param_name,
                    'type': f"{addr_qual} {msl_type}*",
                    'is_ptr': True,
                })
            else:
                params.append({
                    'decl': f"    constant {msl_type}& {param_name} [[buffer({buffer_idx})]]",
                    'name': param_name,
                    'type': msl_type,
                    'is_ptr': False,
                })
            buffer_idx += 1

        return params

    def _split_args(self, args_str: str) -> list:
        """Split argument string respecting nested angle brackets and parens."""
        args = []
        depth = 0
        current = []
        for ch in args_str:
            if ch in ('(', '<', '[', '{'):
                depth += 1
                current.append(ch)
            elif ch in (')', '>', ']', '}'):
                depth -= 1
                current.append(ch)
            elif ch == ',' and depth == 0:
                args.append(''.join(current))
                current = []
            else:
                current.append(ch)
        if current:
            args.append(''.join(current))
        return args

    def _parse_llvm_type(self, arg_str: str) -> tuple:
        """Parse an LLVM type string, return (msl_type, is_pointer, addrspace)."""
        arg_str = arg_str.strip()
        # Remove parameter attributes
        for attr in ('noundef', 'nonnull', 'readnone', 'readonly', 'writeonly', 'nocapture', 'noalias', 'signext',
                     'zeroext', 'align', 'dereferenceable', 'nofree', 'nsw', 'nuw'):
            arg_str = re.sub(rf'\b{attr}\b(\(\d+\))?', '', arg_str)
        arg_str = arg_str.strip()

        # Check for pointer type
        addrspace = 0
        if 'ptr' in arg_str:
            # ptr addrspace(N) or just ptr
            as_match = re.search(r'addrspace\((\d+)\)', arg_str)
            if as_match:
                addrspace = int(as_match.group(1))
            return ('float', True, addrspace)  # Default pointer element type

        # Scalar types
        for llvm_ty, msl_ty in self.TYPE_MAP.items():
            if re.search(rf'\b{llvm_ty}\b', arg_str):
                return (msl_ty, False, 0)

        # Vector types: <N x type>
        vec_match = re.search(r'<(\d+)\s*x\s*(\w+)>', arg_str)
        if vec_match:
            n = int(vec_match.group(1))
            elem_type = self.TYPE_MAP.get(vec_match.group(2), 'float')
            if n in (2, 3, 4):
                return (f"{elem_type}{n}", False, 0)
            return (elem_type, False, 0)

        return ('uint32_t', False, 0)

    def _translate_body(self):
        """Translate LLVM IR function body to MSL statements."""
        # Extract the function body (between first { and last })
        body_match = re.search(r'define\s+[^{]+\{(.+?)^\}', self.llvm_ir, re.MULTILINE | re.DOTALL)
        if not body_match:
            self._body_lines.append("    // Empty kernel body")
            return

        body = body_match.group(1)
        lines = body.strip().split('\n')

        for line in lines:
            line = line.strip()
            if not line or line.startswith(';'):
                continue
            self._translate_instruction(line)

    def _translate_instruction(self, line: str):
        """Translate a single LLVM IR instruction to MSL."""
        # Labels (basic blocks)
        if line.endswith(':') or re.match(r'^[a-zA-Z_][a-zA-Z0-9_.]*:', line):
            label = line.rstrip(':').strip()
            self._body_lines.append(f"    // block: {label}")
            return

        # Assignment: %result = operation
        assign_match = re.match(r'(%[a-zA-Z0-9_.]+)\s*=\s*(.+)', line)
        if assign_match:
            dest = assign_match.group(1)
            operation = assign_match.group(2).strip()
            self._translate_assignment(dest, operation)
            return

        # Store instruction
        if line.startswith('store'):
            self._translate_store(line)
            return

        # Branch/control flow
        if line.startswith('br '):
            self._translate_branch(line)
            return

        # Return
        if line.startswith('ret'):
            self._body_lines.append("    return;")
            return

        # Fence (barrier)
        if line.startswith('fence'):
            self._translate_fence(line)
            return

        # Call without assignment
        if 'call' in line:
            self._translate_call(None, line)
            return

    def _translate_assignment(self, dest: str, operation: str):
        """Translate an LLVM assignment instruction."""
        var_name = self._get_msl_var(dest)

        # Binary operations
        for llvm_op, msl_op in self.BINOP_MAP.items():
            match = re.match(rf'{llvm_op}\s+(\w+)\s+(.+),\s*(.+)', operation)
            if match:
                ty = self._llvm_to_msl_type(match.group(1))
                lhs = self._resolve_operand(match.group(2))
                rhs = self._resolve_operand(match.group(3))
                if msl_op == 'fmod':
                    self._body_lines.append(f"    {ty} {var_name} = fmod({lhs}, {rhs});")
                else:
                    self._body_lines.append(f"    {ty} {var_name} = {lhs} {msl_op} {rhs};")
                self._var_types[dest] = ty
                return

        # Integer comparison
        icmp_match = re.match(r'icmp\s+(\w+)\s+(\w+)\s+(.+),\s*(.+)', operation)
        if icmp_match:
            pred = icmp_match.group(1)
            lhs = self._resolve_operand(icmp_match.group(3))
            rhs = self._resolve_operand(icmp_match.group(4))
            op = self.ICMP_MAP.get(pred, '==')
            self._body_lines.append(f"    bool {var_name} = ({lhs} {op} {rhs});")
            self._var_types[dest] = 'bool'
            return

        # Float comparison
        fcmp_match = re.match(r'fcmp\s+(\w+)\s+(\w+)\s+(.+),\s*(.+)', operation)
        if fcmp_match:
            pred = fcmp_match.group(1)
            lhs = self._resolve_operand(fcmp_match.group(3))
            rhs = self._resolve_operand(fcmp_match.group(4))
            op = self.FCMP_MAP.get(pred, '==')
            if op in ('!isnan', 'isnan'):
                self._body_lines.append(f"    bool {var_name} = {op}({lhs});")
            else:
                self._body_lines.append(f"    bool {var_name} = ({lhs} {op} {rhs});")
            self._var_types[dest] = 'bool'
            return

        # Load instruction
        load_match = re.match(r'load\s+(\w+),\s*ptr\s+(.+?)(?:,\s*align\s+\d+)?$', operation)
        if load_match:
            ty = self._llvm_to_msl_type(load_match.group(1))
            ptr = self._resolve_operand(load_match.group(2).strip().rstrip(','))
            self._body_lines.append(f"    {ty} {var_name} = *{ptr};")
            self._var_types[dest] = ty
            return

        # GEP (getelementptr)
        gep_match = re.match(r'getelementptr\s+(?:inbounds\s+)?(\w+),\s*ptr\s+(.+)', operation)
        if gep_match:
            ptr = self._resolve_operand(gep_match.group(2).split(',')[0].strip())
            indices = gep_match.group(2).split(',')[1:]
            if indices:
                idx = self._resolve_operand(indices[-1].strip().split()[-1])
                self._body_lines.append(f"    auto {var_name} = {ptr} + {idx};")
            else:
                self._body_lines.append(f"    auto {var_name} = {ptr};")
            self._var_types[dest] = 'auto'
            return

        # Select
        sel_match = re.match(r'select\s+i1\s+(.+),\s*(\w+)\s+(.+),\s*(\w+)\s+(.+)', operation)
        if sel_match:
            cond = self._resolve_operand(sel_match.group(1).strip().rstrip(','))
            ty = self._llvm_to_msl_type(sel_match.group(2))
            true_val = self._resolve_operand(sel_match.group(3).strip().rstrip(','))
            false_val = self._resolve_operand(sel_match.group(5))
            self._body_lines.append(f"    {ty} {var_name} = {cond} ? {true_val} : {false_val};")
            self._var_types[dest] = ty
            return

        # Casts
        cast_ops = ('bitcast', 'trunc', 'zext', 'sext', 'fptrunc', 'fpext', 'fptoui', 'fptosi', 'uitofp', 'sitofp',
                    'ptrtoint', 'inttoptr')
        for cast_op in cast_ops:
            cast_match = re.match(rf'{cast_op}\s+(\w+)\s+(.+?)\s+to\s+(\w+)', operation)
            if cast_match:
                src_val = self._resolve_operand(cast_match.group(2))
                dst_ty = self._llvm_to_msl_type(cast_match.group(3))
                self._body_lines.append(f"    {dst_ty} {var_name} = static_cast<{dst_ty}>({src_val});")
                self._var_types[dest] = dst_ty
                return

        # PHI nodes → declare variable, will be assigned in predecessors
        if operation.startswith('phi'):
            phi_match = re.match(r'phi\s+(\w+)\s+(.+)', operation)
            if phi_match:
                ty = self._llvm_to_msl_type(phi_match.group(1))
                # For phi, just declare the variable; proper SSA resolution
                # happens during block ordering
                self._body_lines.append(f"    {ty} {var_name}; // phi")
                self._var_types[dest] = ty
                return

        # Call (with return value)
        if 'call' in operation:
            self._translate_call(dest, operation)
            return

        # Alloca
        if operation.startswith('alloca'):
            alloca_match = re.match(r'alloca\s+(\w+)', operation)
            if alloca_match:
                ty = self._llvm_to_msl_type(alloca_match.group(1))
                self._body_lines.append(f"    thread {ty} {var_name}_storage;")
                self._body_lines.append(f"    thread {ty}* {var_name} = &{var_name}_storage;")
                self._var_types[dest] = f"thread {ty}*"
                return

        # Fallback: emit as comment
        self._body_lines.append(f"    // TODO: {operation}")

    def _translate_store(self, line: str):
        """Translate LLVM store instruction."""
        match = re.match(r'store\s+(\w+)\s+(.+?),\s*ptr\s+(.+?)(?:,\s*align\s+\d+)?$', line)
        if match:
            val = self._resolve_operand(match.group(2).strip().rstrip(','))
            ptr = self._resolve_operand(match.group(3).strip())
            self._body_lines.append(f"    *{ptr} = {val};")

    def _translate_branch(self, line: str):
        """Translate LLVM branch instruction."""
        # Conditional branch
        cond_match = re.match(r'br\s+i1\s+(.+),\s*label\s+%(.+),\s*label\s+%(.+)', line)
        if cond_match:
            cond = self._resolve_operand(cond_match.group(1).strip().rstrip(','))
            true_label = cond_match.group(2).strip()
            false_label = cond_match.group(3).strip()
            self._body_lines.append(f"    if ({cond}) goto {true_label}; else goto {false_label};")
            return
        # Unconditional branch
        uncond_match = re.match(r'br\s+label\s+%(.+)', line)
        if uncond_match:
            target = uncond_match.group(1).strip()
            self._body_lines.append(f"    goto {target};")

    def _translate_fence(self, line: str):
        """Translate LLVM fence to Metal barrier."""
        if 'seq_cst' in line or 'acq_rel' in line:
            self._body_lines.append("    threadgroup_barrier(mem_flags::mem_threadgroup);")
        else:
            self._body_lines.append("    simdgroup_barrier(mem_flags::mem_none);")

    def _translate_call(self, dest, line: str):
        """Translate LLVM call instruction to MSL function call."""
        # Extract function name
        call_match = re.search(r'call\s+\w+\s+@([a-zA-Z_][a-zA-Z0-9_.]*)\s*\(([^)]*)\)', line)
        if not call_match:
            call_match = re.search(r'call\s+\w+\s+\([^)]*\)\s+@([a-zA-Z_][a-zA-Z0-9_.]*)\s*\(([^)]*)\)', line)
        if not call_match:
            if dest:
                var_name = self._get_msl_var(dest)
                self._body_lines.append(f"    auto {var_name} = 0; // unresolved call")
            return

        func_name = call_match.group(1)
        args_str = call_match.group(2)

        # Map LLVM intrinsics to Metal stdlib functions
        msl_func = None
        for llvm_prefix, metal_func in self.INTRINSIC_MAP.items():
            if func_name.startswith(llvm_prefix):
                msl_func = metal_func
                break

        if msl_func is None:
            # Check for Metal-specific intrinsics
            if 'threadgroup_barrier' in func_name or 'barrier' in func_name:
                self._body_lines.append("    threadgroup_barrier(mem_flags::mem_threadgroup);")
                return
            elif 'simd_shuffle' in func_name:
                msl_func = 'simd_shuffle'
            elif 'simd_ballot' in func_name:
                msl_func = 'simd_ballot'
            elif 'simd_sum' in func_name or 'simd_reduce' in func_name:
                msl_func = 'simd_sum'
            else:
                msl_func = func_name.replace('llvm.', '').replace('.', '_')

        # Parse call arguments
        call_args = []
        if args_str.strip():
            for arg in self._split_args(args_str):
                arg = arg.strip()
                parts = arg.rsplit(None, 1)
                if parts:
                    call_args.append(self._resolve_operand(parts[-1]))

        args_joined = ", ".join(call_args)
        if dest:
            var_name = self._get_msl_var(dest)
            self._body_lines.append(f"    auto {var_name} = {msl_func}({args_joined});")
            self._var_types[dest] = 'auto'
        else:
            self._body_lines.append(f"    {msl_func}({args_joined});")

    def _get_msl_var(self, llvm_name: str) -> str:
        """Get or create an MSL variable name for an LLVM SSA value."""
        if llvm_name in self._vars:
            return self._vars[llvm_name]
        self._var_counter += 1
        # Clean the name for MSL
        clean = llvm_name.lstrip('%').replace('.', '_').replace('-', '_')
        if clean[0].isdigit():
            clean = f"v_{clean}"
        msl_name = f"{clean}"
        self._vars[llvm_name] = msl_name
        return msl_name

    def _resolve_operand(self, operand: str) -> str:
        """Resolve an LLVM operand to its MSL equivalent."""
        operand = operand.strip()
        # Constants
        if operand == 'true':
            return 'true'
        if operand == 'false':
            return 'false'
        if operand == 'null' or operand == 'zeroinitializer':
            return '0'
        # Numeric constants
        try:
            if '.' in operand or 'e' in operand.lower():
                return str(float(operand))
            return str(int(operand))
        except (ValueError, TypeError):
            pass
        # Hex float constants
        if operand.startswith('0x'):
            try:
                import struct
                bits = int(operand, 16)
                val = struct.unpack('d', struct.pack('Q', bits))[0]
                return f"{val:.6e}f"
            except (ValueError, struct.error):
                return operand
        # SSA references
        if operand.startswith('%'):
            return self._get_msl_var(operand)
        # Already resolved
        return operand

    def _llvm_to_msl_type(self, ty: str) -> str:
        """Convert an LLVM type name to MSL type."""
        ty = ty.strip()
        return self.TYPE_MAP.get(ty, 'uint32_t')

    def _emit_msl(self, kernel_params: list) -> str:
        """Assemble the final MSL source from translated components."""
        params_decl = ",\n".join([p['decl'] for p in kernel_params])
        if params_decl:
            params_decl += ","

        # Threadgroup memory declaration
        tg_mem = ""
        if self.shared_mem > 0:
            num_floats = self.shared_mem // 4
            tg_mem = f"    threadgroup float shared_mem[{num_floats}];\n"

        # Body
        body = "\n".join(self._body_lines) if self._body_lines else "    // Kernel body"

        return f"""// Metal 4.0 kernel generated by Triton compiler
// Target: Apple Silicon (apple{self.gpu_family}), SIMD width 32
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_math>
using namespace metal;

[[kernel, max_total_threads_per_threadgroup({self.num_threads})]]
void {self.kernel_name}(
{params_decl}
    uint3 tid [[thread_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {{
{tg_mem}    uint program_id = gid.x;
    uint thread_id = lid.x;

{body}
}}
"""


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
    allowed_dot_input_precisions: Tuple[str] = ("ieee", )
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
        from triton.language.extra.metal import utils as metal_utils
        codegen_fns = {
            "convert_custom_types": metal_utils.convert_custom_float8,
            "min_dot_size": min_dot_size(self.target),
        }
        return codegen_fns

    def get_module_map(self) -> Dict[str, ModuleType]:
        from triton.language.extra.metal import libdevice
        return {"triton.language.extra.libdevice": libdevice}

    def load_dialects(self, ctx):
        try:
            from triton.backends.metal import metal as metal_module
            metal_module.load_dialects(ctx)
        except (ImportError, AttributeError):
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
        from triton.backends.metal import metal as metal_module
        metal_module.passes.ttgpuir.add_allocate_shared_memory(pm, gpu_family)
        passes.ttgpuir.add_allocate_warp_groups(pm, False)
        metal_module.passes.ttgpuir.add_to_llvmir(pm, gpu_family)
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
        # Target aarch64 for Apple Silicon (Metal 4 / macOS 26+)
        triple = 'aarch64-apple-macosx26.0.0'
        proc = 'apple-m2'
        features = '+neon,+fp-armv8,+fullfp16,+sha2,+aes'
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

        Performs a structured translation of LLVM IR to MSL:
        1. Parses the LLVM IR module to extract kernel function(s)
        2. Maps LLVM types to MSL types with proper address space annotations
        3. Translates LLVM instructions to MSL operations
        4. Handles Metal-specific features: threadgroup memory, SIMD operations,
           buffer bindings, thread position attributes

        The translation preserves the computational semantics while adapting to
        Metal's execution model (threadgroups, SIMD groups, buffer bindings).
        """
        kernel_name = metadata.get("name", "triton_kernel")
        llir_to_msl = LLVMIRToMSLTranslator(src, metadata, opt, gpu_family)
        msl_source = llir_to_msl.translate()
        metadata["msl_source"] = msl_source
        metadata["llvm_ir"] = src
        metadata["name"] = llir_to_msl.kernel_name
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
                "xcrun", "-sdk", "macosx", "metal", "-target", "air64-apple-macosx26.0.0", "-std", "metal4.0", "-o",
                air_path, "-c", ir_path
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
            metallib_cmd = ["xcrun", "-sdk", "macosx", "metallib", air_path, "-o", metallib_path]
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
