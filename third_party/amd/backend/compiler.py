from triton.backends.compiler import BaseBackend, GPUTarget, Language
from triton._C.libtriton import ir, passes, llvm, amd
from triton import knobs
from dataclasses import dataclass
from typing import Any, Dict, Tuple
from types import ModuleType
import hashlib
import tempfile
import re
import functools
import warnings
from pathlib import Path


def get_min_dot_size(target: GPUTarget):
    # We fallback to use FMA and cast arguments if certain configurations is
    # not supported natively by matrix core units.
    return lambda lhs_type, rhs_type: (1, 1, 1)


def is_pingpong_schedule_enabled(arch, use_async_copy):
    return (arch == "gfx942" or (arch == "gfx950" and use_async_copy is True)
            ) if knobs.amd.use_block_pingpong is None else knobs.amd.use_block_pingpong


def is_in_thread_transpose_enabled(arch):
    return (arch == "gfx942") if knobs.amd.use_in_thread_transpose is None else knobs.amd.use_in_thread_transpose


@dataclass(frozen=True)
class HIPOptions:
    num_warps: int = 4
    waves_per_eu: int = 1
    num_stages: int = 2
    num_ctas: int = 1
    extern_libs: dict = None
    cluster_dims: tuple = (1, 1, 1)
    debug: bool = False
    sanitize_overflow: bool = True
    arch: str = None
    # We have native support for OCP fp8 variants since CDNA4/RDNA4. For earlier generations,
    # we software emulate the support for them.
    # UZ fp8 variants (fp8e4b8 and fp8e5b16) are natively supported for CDNA3. For other
    # architectures they are software emulated.
    supported_fp8_dtypes: Tuple[str] = ("fp8e4nv", "fp8e5", "fp8e5b16", "fp8e4b8")
    deprecated_fp8_dot_operand_dtypes: Tuple[str] = ()
    default_dot_input_precision: str = "ieee"
    allowed_dot_input_precisions: Tuple[str] = ("ieee", )
    enable_fp_fusion: bool = True
    launch_cooperative_grid: bool = False
    matrix_instr_nonkdim: int = 0
    kpack: int = 1
    allow_flush_denorm: bool = False
    max_num_imprecise_acc_default: int = 0
    backend_name: str = 'hip'
    instrumentation_mode: str = ""

    # The following option provides hints to the AMDGPU backend regarding instruction scheduling
    # for all `tt.dot` operations in a kernel. The "none" variant preserves the default
    # instruction scheduling of the AMDGPU backend which aims at maximizing occupancy.
    # The option is experimental and may change at any time regarding its semantics and/or may
    # be gone entirely anytime.
    #
    # Current experimental scheduling variants:
    #
    # attention: enables a bunch of optimizations for attention kernels, including:
    #            - iglp 2 and sched.barrier around it
    #            - sink-insts-to-avoid-spills flag to avoid register spills
    schedule_hint: str = 'none'

    def __post_init__(self):
        gfx_major = int(self.arch[3:-2])  # Drop "gfx" prefix and minor/patch number
        warp_size = 32 if gfx_major >= 10 else 64
        object.__setattr__(self, 'warp_size', warp_size)
        assert self.num_warps > 0 and (self.num_warps & (self.num_warps - 1)) == 0, \
               "num_warps must be a power of 2"

        if (self.arch == 'gfx950') and (self.kpack != 1):
            warnings.warn(
                f"kpack is deprecated starting from gfx950 and will be removed in later releases. So for now kpack = {self.kpack} will be overwritten to 1 to make transitioning easier."
            )
            object.__setattr__(self, 'kpack', 1)

        default_libdir = Path(__file__).parent / 'lib'
        extern_libs = {} if self.extern_libs is None else dict(self.extern_libs)
        for lib in ["ocml", "ockl"]:
            extern_libs[lib] = str(default_libdir / f'{lib}.bc')
        object.__setattr__(self, 'extern_libs', tuple(extern_libs.items()))

    def hash(self):
        key = '_'.join([f'{name}-{val}' for name, val in self.__dict__.items()])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


class HIPBackend(BaseBackend):
    instrumentation = None

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == 'hip'

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        assert isinstance(target.arch, str)
        self.binary_ext = "hsaco"

    def get_target_name(self, options) -> str:
        return f"hip:{options.arch}"

    def parse_options(self, opts) -> Any:
        args = {'arch': knobs.runtime.override_arch or self.target.arch}

        if opts.get("num_ctas", 1) > 1:
            raise ValueError("num_ctas > 1 not supported for AMD GPUs")

        # Enable XF32 (TF32) for CDNA3 GPUs
        if self.target.arch == 'gfx942':
            allowed_dot_input_precisions = set(HIPOptions.allowed_dot_input_precisions)
            allowed_dot_input_precisions.update({'tf32'})
            args["allowed_dot_input_precisions"] = tuple(sorted(allowed_dot_input_precisions))

        if "supported_fp8_dtypes" not in opts:
            args["supported_fp8_dtypes"] = tuple(sorted(HIPOptions.supported_fp8_dtypes))

        if self.target.arch == 'gfx950':
            deprecated_fp8_dot_operand_dtypes = set(HIPOptions.deprecated_fp8_dot_operand_dtypes)
            deprecated_fp8_dot_operand_dtypes.update({"fp8e5b16", "fp8e4b8"})
            args["deprecated_fp8_dot_operand_dtypes"] = tuple(sorted(deprecated_fp8_dot_operand_dtypes))

        if "enable_fp_fusion" not in opts:
            args["enable_fp_fusion"] = knobs.language.default_fp_fusion
        args.update({k: opts[k] for k in HIPOptions.__dataclass_fields__.keys() \
                     if k in opts and opts[k] is not None})
        return HIPOptions(**args)

    def pack_metadata(self, metadata):
        return (
            metadata.num_warps,
            metadata.num_ctas,
            metadata.shared,
            metadata.cluster_dims[0],
            metadata.cluster_dims[1],
            metadata.cluster_dims[2],
        )

    def get_codegen_implementation(self, options):
        return {"min_dot_size": get_min_dot_size(self.target)}

    def get_module_map(self) -> Dict[str, ModuleType]:
        from triton.language.extra.hip import libdevice

        return {"triton.language.extra.libdevice": libdevice}

    def load_dialects(self, ctx):
        amd.load_dialects(ctx)
        if HIPBackend.instrumentation:
            HIPBackend.instrumentation.load_dialects(ctx)

    @staticmethod
    def is_within_2gb(arg):
        import torch

        MAX_INT_32 = 2**31 - 1
        if hasattr(arg, "ptr_range"):
            return arg.ptr_range() <= MAX_INT_32
        if isinstance(arg, torch.Tensor) and hasattr(arg, "untyped_storage"):
            return arg.untyped_storage().size() <= MAX_INT_32
        return False

    @staticmethod
    def parse_attr(desc):
        ret = BaseBackend.parse_attr(desc)
        if "S" in desc:
            ret += [["tt.pointer_range", 32]]
        return ret

    @staticmethod
    def get_arg_specialization(arg, ty, **kwargs):
        ret = BaseBackend.get_arg_specialization(arg, ty, **kwargs)
        # Only attempt to do buffer ops specialization if buffer ops are enabled.
        # Otherwise the is_within_2gb check is unnecessary overhead.
        if knobs.amd.use_buffer_ops and ty == "tensor" and HIPBackend.is_within_2gb(arg):
            ret += "S"
        return ret

    @staticmethod
    def make_ttir(mod, metadata, options):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.ttir.add_rewrite_tensor_pointer(pm)
        passes.ttir.add_rewrite_tensor_descriptor_to_pointer(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_combine(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.ttir.add_triton_licm(pm)
        passes.common.add_symbol_dce(pm)
        passes.ttir.add_loop_unroll(pm)
        pm.run(mod)
        return mod

    @staticmethod
    def make_ttgir(mod, metadata, options):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.ttir.add_convert_to_ttgpuir(pm, f"hip:{options.arch}", options.num_warps, options.warp_size,
                                           options.num_ctas)
        pm.run(mod)
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.ttgpuir.add_coalesce(pm)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_optimize_thread_locality(pm)
        amd.passes.ttgpuir.add_accelerate_matmul(pm, options.arch, options.matrix_instr_nonkdim, options.kpack)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        amd.passes.ttgpuir.add_optimize_epilogue(pm)
        passes.ttgpuir.add_optimize_dot_operands(pm, True)
        amd.passes.ttgpuir.add_hoist_layout_conversions(pm)

        passes.ttgpuir.add_fuse_nested_loops(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_triton_licm(pm)
        passes.common.add_canonicalizer(pm)

        global_prefetch = knobs.amd.global_prefetch
        local_prefetch = knobs.amd.local_prefetch
        use_async_copy = knobs.amd.use_async_copy
        use_block_pingpong = is_pingpong_schedule_enabled(options.arch, use_async_copy)

        amd.passes.ttgpuir.add_stream_pipeline(pm, options.num_stages, global_prefetch, local_prefetch, use_async_copy,
                                               use_block_pingpong)
        if use_async_copy:
            amd.passes.ttgpuir.add_coalesce_async_copy(pm, options.arch)
        passes.common.add_canonicalizer(pm)
        if options.schedule_hint.lower() != "none":
            amd.passes.ttgpuir.insert_instruction_sched_hints(pm, options.schedule_hint)
        passes.ttgpuir.add_optimize_dot_operands(pm, True)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_reduce_data_duplication(pm)
        if is_in_thread_transpose_enabled(options.arch):
            amd.passes.ttgpuir.add_in_thread_transpose(pm)
            passes.ttgpuir.add_remove_layout_conversions(pm)
        amd.passes.ttgpuir.add_reorder_instructions(pm)
        if use_block_pingpong and options.num_stages > 1:
            amd.passes.ttgpuir.add_block_pingpong(pm, options.num_stages)

        if knobs.amd.use_buffer_ops:
            amd.passes.ttgpuir.add_canonicalize_pointers(pm)
            passes.common.add_canonicalizer(pm)
            amd.passes.ttgpuir.add_convert_to_buffer_ops(pm, options.arch, knobs.amd.use_buffer_atomics)

        amd.passes.ttgpuir.add_fold_true_cmpi(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        if use_async_copy:
            amd.passes.ttgpuir.add_update_async_wait_count(pm, options.arch)
        pm.run(mod)
        return mod

    @staticmethod
    def gluon_to_ttgir(src, metadata, options):
        mod = src
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()

        passes.gluon.add_inliner(pm)
        passes.gluon.add_resolve_auto_encodings(pm)
        passes.common.add_sccp(pm)
        passes.ttir.add_loop_aware_cse(pm)
        passes.gluon.add_canonicalizer(pm)
        passes.ttgpuir.add_combine_tensor_select_and_if(pm)

        pm.run(mod)
        return mod

    @staticmethod
    def make_llir(src, metadata, options):
        mod = src
        # TritonGPU -> LLVM-IR (MLIR)
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        # custom_lds_size is an experimental parameter that defines amount of LDS available
        # for one thread block. Measured in bytes.
        #
        # If custom_lds_size = 0, pass will consider all LDS is available for one threads block,
        # LDS size is determined by provided arch name.
        custom_lds_size = 0
        amd.passes.ttgpuir.add_optimize_lds_usage(pm, options.arch, custom_lds_size)
        passes.convert.add_scf_to_cf(pm)
        passes.convert.add_index_to_llvmir(pm)

        amd.passes.ttgpuir.add_allocate_shared_memory(pm)
        # instrumentation point here so we can override IRs above (e.g., ttir and ttgir)
        if HIPBackend.instrumentation:
            HIPBackend.instrumentation.patch("ttgpuir_to_llvmir", pm, mod.context)
        ## __HIP_FTZ is used to control the denorm flushing behavior of exp2 op as follows:
        ## 1. If __HIP_FTZ = 1, exp2 flushes denorms in input and output regardless
        ##    of the value of kernel arg `allow_flush_denorm`.
        ## 2. If __HIP_FTZ = 0, whether exp2 flushes denorms in input and output
        ##    depends on the value of kernel arg `allow_flush_denorm`.
        ## 3. __HIP_FTZ is default to 1 and not exposed as a kernel argument.
        ##    For now it is used as a controller for developers only.
        __HIP_FTZ = True
        amd.passes.ttgpuir.add_to_llvmir(pm, options.arch, __HIP_FTZ)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)

        passes.convert.add_cf_to_llvmir(pm)
        passes.convert.add_arith_to_llvmir(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)

        if options.schedule_hint.lower() != "none":
            amd.passes.ttgpuir.lower_instruction_sched_hints(pm, options.arch, options.num_stages)

        # This can not be moved below the di_scope pass
        if HIPBackend.instrumentation:
            HIPBackend.instrumentation.patch("llvmir_to_llvm", pm, mod.context)

        if not knobs.compilation.disable_line_info:
            passes.llvmir.add_di_scope(pm)

        amd.passes.ttgpuir.add_builtin_func_to_llvmir(pm, __HIP_FTZ)
        pm.run(mod)

        # LLVM-IR (MLIR) -> LLVM-IR (LLVM)
        llvm.init_targets()
        context = llvm.context()
        llvm_mod = llvm.to_module(mod, context)
        amd.attach_target_triple(llvm_mod)
        target_features = ''
        if knobs.compilation.enable_asan:
            target_features = '+xnack'
        llvm.attach_datalayout(llvm_mod, amd.TARGET_TRIPLE, options.arch, target_features)

        # Set various control constants on the LLVM module so that device
        # libraries can resolve references to them.
        amd.set_isa_version(llvm_mod, options.arch)
        amd.set_abi_version(llvm_mod, 500)
        amd.set_bool_control_constant(llvm_mod, "__oclc_finite_only_opt", False)
        amd.set_bool_control_constant(llvm_mod, "__oclc_correctly_rounded_sqrt32", True)
        amd.set_bool_control_constant(llvm_mod, "__oclc_unsafe_math_opt", False)
        amd.set_bool_control_constant(llvm_mod, "__oclc_wavefrontsize64", options.warp_size == 64)

        # Set kernel attributes first given this may affect later optimizations.
        fns = [fn for fn in llvm_mod.get_functions() if not fn.is_declaration()]
        # The public kernel should be kernel 0.
        fns[0].set_calling_conv(amd.CALLING_CONV_AMDGPU_KERNEL)
        fns[0].add_fn_attr("amdgpu-flat-work-group-size", f"1,{options.num_warps*options.warp_size}")
        # LLVM AMDGPU backend supports the attribute "amdgpu-waves-per-eu"="<min>[, <max>]".
        # This attribute may be attached to a kernel function definition and is an optimization hint.
        # <min> parameter specifies the requested minimum number of waves per EU, and optional <max> parameter
        # specifies the requested maximum number of waves per EU (must be greater than <min> if specified).
        # If <max> is omitted, then there is no restriction on the maximum number of waves per EU other than
        # the one dictated by the hardware for which the kernel is compiled. Passing 0, 0 as <min>, <max>
        # implies the default behavior (no limits).
        fns[0].add_fn_attr("amdgpu-waves-per-eu", f"{options.waves_per_eu}")
        denormal_mode = "preserve-sign" if options.allow_flush_denorm else "ieee"
        fns[0].add_fn_attr("denormal-fp-math-f32", denormal_mode)
        if knobs.compilation.enable_asan:
            fns[0].add_fn_target_feature("+xnack")
            fns[0].add_fn_asan_attr()

        # Hint the compiler that we'd like the firmware to set the kernel arguments
        # to user SGPRs so that the kernel does not need to s_load its arguments
        # from memory.
        amd.set_all_fn_arg_inreg(fns[0])

        if knobs.compilation.enable_asan:
            default_libdir = Path(__file__).parent / 'lib'
            paths = [
                str(default_libdir / 'asanrtl.bc'),
                str(default_libdir / "ocml.bc"),
                str(default_libdir / "ockl.bc")
            ]
            llvm.link_extern_libs(llvm_mod, paths)
        elif options.extern_libs:
            paths = [path for (name, path) in options.extern_libs if amd.need_extern_lib(llvm_mod, name)]
            if len(paths) > 0:
                llvm.link_extern_libs(llvm_mod, paths)

        llvm.optimize_module(llvm_mod, llvm.OPTIMIZE_O3, options.arch, '', [], options.enable_fp_fusion)

        # Architectures with architected SGPRs store the workgroup id in ttmp9 (X) and ttmp7 (Y[15:0], Z[31:16]).
        # These attributes are used to determine if Z should be masked out when loading Y. They are inferred during
        # optimize_module from calls to @llvm.amdgcn.workgroup.id.x/y/z(). We cannot rely on this because a
        # dispatch dimensions might be used even if there is no program_id() call for it.
        if amd.has_architected_sgprs(options.arch):
            fns[0].remove_fn_attr("amdgpu-no-workgroup-id-x")
            fns[0].remove_fn_attr("amdgpu-no-workgroup-id-y")
            fns[0].remove_fn_attr("amdgpu-no-workgroup-id-z")

        if knobs.amd.scalarize_packed_fops:
            amd.add_scalarize_packed_fops_llvm_pass(fns[0])

        # Get some metadata
        metadata["shared"] = src.get_int_attr("ttg.shared")
        metadata["profile_scratch_size"] = src.get_int_attr("ttg.profile_scratch_memory_size") or 0
        metadata["profile_scratch_align"] = src.get_int_attr("ttg.profile_scratch_memory_alignment") or 1

        amd.cleanup_bitcode_metadata(llvm_mod)
        # Disable inlining of print related functions,
        # because inlining of these function could slow down compilation significantly
        amd.disable_print_inline(llvm_mod)
        return str(llvm_mod)

    @staticmethod
    def make_amdgcn(src, metadata, options):
        # Find kernel names (there should only be one)
        # We get the name at the last possible step to accommodate `triton.compile`
        # on user-provided LLVM
        names = re.findall(r"define amdgpu_kernel void @([a-zA-Z_][a-zA-Z0-9_]*)", src)
        assert len(names) == 1
        metadata["name"] = names[0]
        # llvm -> hsaco
        flags = []
        # The sink-insts-to-avoid-spills flag asks LLVM backend to sink instructions
        # into loops to avoid register spills in the MachineSinking pass, while it
        # can also lead to regression in some cases. But from current observation,
        # the regression is not significant. It would be better to have some heuristics.
        if options.schedule_hint == 'attention':
            flags.append('sink-insts-to-avoid-spills')
        features = '-real-true16' if 'gfx11' in options.arch else ''
        amdgcn = llvm.translate_to_asm(src, amd.TARGET_TRIPLE, options.arch, features, flags, options.enable_fp_fusion,
                                       False)
        if knobs.amd.dump_amdgcn:
            print("// -----// AMDGCN Dump //----- //")
            print(amdgcn)
        return amdgcn

    @staticmethod
    def make_hsaco(src, metadata, options):
        target_features = ''
        if knobs.compilation.enable_asan:
            target_features = '+xnack'
        hsaco = amd.assemble_amdgcn(src, options.arch, target_features)
        with tempfile.NamedTemporaryFile() as tmp_out:
            with tempfile.NamedTemporaryFile() as tmp_in:
                with open(tmp_in.name, "wb") as fd_in:
                    fd_in.write(hsaco)
                amd.link_hsaco(tmp_in.name, tmp_out.name)
            with open(tmp_out.name, "rb") as fd_out:
                ret = fd_out.read()
        return ret

    def add_stages(self, stages, options, language):
        if language == Language.TRITON:
            stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
            stages["ttgir"] = lambda src, metadata: self.make_ttgir(src, metadata, options)
        elif language == Language.GLUON:
            stages["ttgir"] = lambda src, metadata: self.gluon_to_ttgir(src, metadata, options)
        stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options)
        stages["amdgcn"] = lambda src, metadata: self.make_amdgcn(src, metadata, options)
        stages["hsaco"] = lambda src, metadata: self.make_hsaco(src, metadata, options)

    @functools.lru_cache()
    def hash(self):
        return f'{self.target}'
