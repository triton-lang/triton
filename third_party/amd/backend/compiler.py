from triton.backends.compiler import BaseBackend, GPUTarget, AttrsDescriptor, register_descriptor
from triton._C.libtriton import ir, passes, llvm, amd
from dataclasses import dataclass
from typing import Any, Dict, Tuple
from types import ModuleType
import hashlib
import tempfile
import os
import re
import subprocess
import functools
from pathlib import Path


def min_dot_size(target: GPUTarget):
    arch_str = target.arch
    # CDNA 3.0 supports k==8 in all mfma variants except for int8
    # (where the smallest `k` supported is 16)
    if "gfx94" in arch_str:
        return lambda lhsType, rhsType: (16, 16, 16) if (lhsType.is_int8() or rhsType.is_int8()) else (16, 16, 8)
    # CDNA 2.0 always supports `k==8`
    if "gfx9" in arch_str:
        return lambda lhsType, rhsType: (16, 16, 8)
    # Other architectures will only support 16,16,16
    return lambda lhsType, rhsType: (16, 16, 16)


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
    supported_fp8_dtypes: Tuple[str] = ("fp8e5", )
    deprecated_fp8_dtypes: Tuple[str] = ()
    default_dot_input_precision: str = "ieee"
    allowed_dot_input_precisions: Tuple[str] = ("ieee", )
    enable_fp_fusion: bool = True
    matrix_instr_nonkdim: int = 0
    kpack: int = 1
    allow_flush_denorm: bool = False
    max_num_imprecise_acc_default: int = 0
    backend_name: str = 'hip'

    # The following option provides hints to the AMDGPU backend regarding instruction scheduling
    # for all `tt.dot` operations in a kernel. The "none" variant preserves the default
    # instruction scheduling of the AMDGPU backend which aims at maximizing occupancy.
    # The option is experimental and may change at any time regarding its semantics and/or may
    # be gone entirely anytime.
    instruction_sched_variant: str = 'none'

    def __post_init__(self):
        default_libdir = Path(__file__).parent / 'lib'
        extern_libs = {} if self.extern_libs is None else dict(self.extern_libs)
        # Ignore user-defined warp size for gfx9
        warp_size = 32 if 'gfx10' in self.arch or 'gfx11' in self.arch or 'gfx12' in self.arch else 64
        object.__setattr__(self, 'warp_size', warp_size)
        libs = ["ocml", "ockl"]
        for lib in libs:
            extern_libs[lib] = str(default_libdir / f'{lib}.bc')
        object.__setattr__(self, 'extern_libs', tuple(extern_libs.items()))
        assert self.num_warps > 0 and (self.num_warps & (self.num_warps - 1)) == 0, \
               "num_warps must be a power of 2"

    def hash(self):
        key = '_'.join([f'{name}-{val}' for name, val in self.__dict__.items()])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


@register_descriptor
class HIPAttrsDescriptor(AttrsDescriptor):
    # This property asserts if the underlying storage area of a given pointer
    # can be resepresented as a 32 bit integer. When this is true, we can be
    # sure that all indices into the tensor behind that pointer can use 32-bit
    # indexing. That opens the door for the AMD backend to use buffer load/store
    # instrinsics, which requires this property. Buffer load/store intrinsics
    # gives direct out-of-bound support and simplifies index calculation for
    # lower register pressure.
    __slots__ = ("pointer_range_32")

    def _add_backend_properties(self, params=None, values=None):
        self.property_values["tt.pointer_range"] = 32
        if params is None or values is None:
            return

        self.arg_properties["tt.pointer_range"] = [
            param.num for param, arg in zip(params, values) if HIPAttrsDescriptor.is_within2gb(arg)
            and not param.do_not_specialize and not param.do_not_specialize_on_alignment
        ]

    @staticmethod
    def is_within2gb(arg):
        if hasattr(arg, "ptr_range"):
            return arg.ptr_range() <= 2**31 - 1
        if "torch.Tensor" in str(type(arg)) and hasattr(arg, "untyped_storage"):
            # Please note that 2**31-1 is the max int32 positive limit
            return arg.untyped_storage().size() <= 2**31 - 1
        return False

    @staticmethod
    def get_property_key(val, align):
        generic_key = AttrsDescriptor.get_property_key(val, align)
        hip_key = "S" if HIPAttrsDescriptor.is_within2gb(val) else "N"
        key = (generic_key + hip_key).replace("N", "")
        return key if key else "N"


class HIPBackend(BaseBackend):

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == 'hip'

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        assert isinstance(target.arch, str)
        self.binary_ext = "hsaco"

    def parse_options(self, opts) -> Any:
        args = {'arch': self.target.arch}

        if "supported_fp8_dtypes" not in opts:
            supported_fp8_dtypes = set(HIPOptions.supported_fp8_dtypes)
            if self.target.arch in ('gfx940', 'gfx941', 'gfx942'):
                supported_fp8_dtypes.update({'fp8e4nv', 'fp8e4b8', 'fp8e5b16'})
            args["supported_fp8_dtypes"] = tuple(sorted(supported_fp8_dtypes))

        if "enable_fp_fusion" not in opts:
            args["enable_fp_fusion"] = os.getenv("TRITON_DEFAULT_FP_FUSION", "1") == "1"
        args.update({k: opts[k] for k in HIPOptions.__dataclass_fields__.keys() if k in opts})
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

    def get_codegen_implementation(self):
        codegen_fns = {"min_dot_size": min_dot_size(self.target)}
        return codegen_fns

    def get_module_map(self) -> Dict[str, ModuleType]:
        from triton.language.extra.hip import libdevice
        return {"triton.language.extra.libdevice": libdevice}

    def load_dialects(self, ctx):
        amd.load_dialects(ctx)

    def get_attrs_descriptor(self, params, args):
        return HIPAttrsDescriptor(params, args)

    @staticmethod
    def compute_spec_key(arg, align):
        return HIPAttrsDescriptor.get_property_key(arg, align)

    @staticmethod
    def path_to_rocm_lld():
        # Check env path for ld.lld
        lld_env_path = os.getenv("TRITON_HIP_LLD_PATH")
        if lld_env_path is not None:
            lld = Path(lld_env_path)
            if lld.is_file():
                return lld
        # Check backend for ld.lld (used for pytorch wheels)
        lld = Path(__file__).parent / "llvm/bin/ld.lld"
        if lld.is_file():
            return lld
        lld = Path("/opt/rocm/llvm/bin/ld.lld")
        if lld.is_file():
            return lld
        lld = Path("/usr/bin/ld.lld")
        if lld.is_file():
            return lld
        raise Exception("ROCm linker /opt/rocm/llvm/bin/ld.lld not found. Set 'TRITON_HIP_LLD_PATH' to its path.")

    @staticmethod
    def make_ttir(mod, metadata, options):
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
        if amd.has_matrix_core_feature(options.arch):
            assert options.num_stages != 0, ("Triton AMD backend pipeliner has been updated. "
                                             "We used to trigger software pipelining with "
                                             "num_stages == 0. Now it will not happen anymore; "
                                             "please update to use num_stages == 2 for "
                                             "equivalent behavior in the past.")
            amd.passes.ttgpuir.add_stream_pipelinev2(pm, options.num_stages)
            passes.common.add_canonicalizer(pm)
        amd.passes.ttgpuir.insert_instruction_sched_hints(pm)
        passes.ttgpuir.add_optimize_dot_operands(pm, True)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_reduce_data_duplication(pm)
        if amd.has_matrix_core_feature(options.arch):
            amd.passes.ttgpuir.add_reorder_instructions(pm)
        if os.environ.get("AMDGCN_USE_BUFFER_OPS", "0") == "1":
            amd.passes.ttgpuir.add_canonicalize_pointers(pm)
            passes.common.add_canonicalizer(pm)
            amd.passes.ttgpuir.add_convert_to_buffer_ops(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        pm.run(mod)
        return mod

    @staticmethod
    def make_llir(src, metadata, options):
        mod = src
        # TritonGPU -> LLVM-IR (MLIR)
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        amd.passes.ttgpuir.add_decompose_unsupported_conversions(pm, options.arch)
        # custom_lds_size is an experimental parameter that defines amount of LDS available
        # for one thread block. Measured in bytes.
        #
        # If custom_lds_size = 0, pass will consider all LDS is available for one threads block,
        # LDS size is determined by provided arch name.
        custom_lds_size = 0
        amd.passes.ttgpuir.add_optimize_lds_usage(pm, options.arch, custom_lds_size)
        passes.convert.add_scf_to_cf(pm)
        passes.convert.add_index_to_llvmir(pm)

        passes.ttgpuir.add_allocate_shared_memory(pm)
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
        amd.passes.ttgpuir.lower_instruction_sched_hints(pm, options.num_stages, options.instruction_sched_variant)
        if os.environ.get("TRITON_DISABLE_LINE_INFO", "0") == "0":
            passes.llvmir.add_di_scope(pm)
        amd.passes.ttgpuir.add_builtin_func_to_llvmir(pm, __HIP_FTZ)
        pm.run(mod)

        # LLVM-IR (MLIR) -> LLVM-IR (LLVM)
        llvm.init_targets()
        context = llvm.context()
        llvm_mod = llvm.to_module(mod, context)
        amd.attach_target_triple(llvm_mod)
        llvm.attach_datalayout(llvm_mod, amd.TARGET_TRIPLE, options.arch, '')

        # Set various control constants on the LLVM module so that device
        # libraries can resolve references to them.
        amd.set_isa_version(llvm_mod, options.arch)
        amd.set_abi_version(llvm_mod, 400)
        amd.set_bool_control_constant(llvm_mod, "__oclc_finite_only_opt", False)
        amd.set_bool_control_constant(llvm_mod, "__oclc_correctly_rounded_sqrt32", True)
        amd.set_bool_control_constant(llvm_mod, "__oclc_unsafe_math_opt", False)
        amd.set_bool_control_constant(llvm_mod, "__oclc_wavefrontsize64", options.warp_size == 64)

        # Set kernel attributes first given this may affect later optimizations.
        fns = [fn for fn in llvm_mod.get_functions() if not fn.is_declaration()]
        # The public kernel should be kernel 0.
        fns[0].set_calling_conv(amd.CALLING_CONV_AMDGPU_KERNEL)
        fns[0].add_fn_attr("amdgpu-flat-work-group-size", f"1,{options.num_warps*options.warp_size}")
        fns[0].add_fn_attr("amdgpu-waves-per-eu", f"{options.waves_per_eu}")
        denormal_mode = "preserve-sign" if options.allow_flush_denorm else "ieee"
        fns[0].add_fn_attr("denormal-fp-math-f32", denormal_mode)

        # Hint the compiler that we'd like the firmware to set the kernel arguments
        # to user SGPRs so that the kernel does not need to s_load its arguments
        # from memory.
        amd.set_all_fn_arg_inreg(fns[0])

        if options.extern_libs:
            paths = [path for (name, path) in options.extern_libs if amd.need_extern_lib(llvm_mod, name)]
            llvm.link_extern_libs(llvm_mod, paths)

        llvm.optimize_module(llvm_mod, llvm.OPTIMIZE_O3, options.arch, '', [], options.enable_fp_fusion)

        # Get some metadata
        metadata["shared"] = src.get_int_attr("triton_gpu.shared")

        amd.cleanup_bitcode_metadata(llvm_mod)
        return str(llvm_mod)

    @staticmethod
    def make_amdgcn(src, metadata, options):
        # Find kernel names (there should only be one)
        # We get the name at the last possible step to accomodate `triton.compile`
        # on user-provided LLVM
        names = re.findall(r"define amdgpu_kernel void @([a-zA-Z_][a-zA-Z0-9_]*)", src)
        assert len(names) == 1
        metadata["name"] = names[0]
        # llvm -> hsaco
        amdgcn = llvm.translate_to_asm(src, amd.TARGET_TRIPLE, options.arch, '', [], options.enable_fp_fusion, False)
        if os.environ.get("AMDGCN_ENABLE_DUMP", "0") == "1":
            print("// -----// AMDGCN Dump //----- //")
            print(amdgcn)
        return amdgcn

    @staticmethod
    def make_hsaco(src, metadata, options):
        hsaco = amd.assemble_amdgcn(src, options.arch, '')

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
        stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options)
        stages["amdgcn"] = lambda src, metadata: self.make_amdgcn(src, metadata, options)
        stages["hsaco"] = lambda src, metadata: self.make_hsaco(src, metadata, options)

    @functools.lru_cache()
    def hash(self):
        version = subprocess.check_output([HIPBackend.path_to_rocm_lld(), "--version"], encoding='utf-8')
        return f'{version}-{self.target}'
