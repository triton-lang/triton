import sys
import triton._C.libtriton.triton as _triton

from triton.code_gen import (
    Binary,
    CodeGenerator,
    CompilationError,
    JITFunction,
    Kernel,
    OutOfResources,
)

class AOTKernel(Kernel):
    @staticmethod
    def _get_type_name(obj):
        if hasattr(obj, "tt_dtype"):
            return obj.tt_dtype
        if hasattr(obj, "data_ptr"):
            return Kernel._type_name(obj.dtype)

        return Kernel._type_name(obj.__class__)

    @staticmethod
    def _to_triton_ir(context, obj):
        type_map = {
            "I": _triton.ir.type.get_int32,
            "f": _triton.ir.type.get_fp32,
            "B": _triton.ir.type.get_int1,
            "f8": _triton.ir.type.get_fp8,
            "f16": _triton.ir.type.get_fp16,
            "bf16": _triton.ir.type.get_bf16,
            "f32": _triton.ir.type.get_fp32,
            "f64": _triton.ir.type.get_fp64,
            "i1": _triton.ir.type.get_int1,
            "i8": _triton.ir.type.get_int8,
            "i16": _triton.ir.type.get_int16,
            "i32": _triton.ir.type.get_int32,
            "i64": _triton.ir.type.get_int64,
        }
        # convert torch.Tensor to Triton IR pointers
        name = AOTKernel._get_type_name(obj)

        if hasattr(obj, "data_ptr"):
            elt_ty = type_map[name](context)
            return _triton.ir.type.make_ptr(elt_ty, 1)
        # default path returns triton.ir.type directly
        return type_map[name](context)

    def __init__(self, fn):
        super().__init__(fn)

        self._bin = None
        self._cache = {}

    def _compile(
        self,
        *wargs,
        device,
        attributes,
        constants,
        num_warps,
        num_stages,
        force_nc_cache,
        **meta,
    ):
        # explicitly set device
        # torch.cuda.set_device(device.index)
        # create IR module
        context = _triton.ir.context()
        # get just-in-time proto-type of kernel
        arg_types = [AOTKernel._to_triton_ir(context, arg) for arg in wargs]
        ret_type = _triton.ir.type.get_void(context)
        prototype = _triton.ir.type.make_function(ret_type, arg_types)
        # generate Triton-IR
        # export symbols visible from self.fn into code-generator object
        gscope = sys.modules[self.fn.module].__dict__
        generator = CodeGenerator(
            context,
            prototype,
            gscope=gscope,
            attributes=attributes,
            constants=constants,
            kwargs=meta,
        )
        try:
            generator.visit(self.fn.parse())
        except Exception as e:
            node = generator.last_node
            if node is None or isinstance(e, (NotImplementedError, CompilationError)):
                raise e
            raise CompilationError(self.fn.src, node, e)
        tt_device = _triton.driver.cu_device(device.index, False)
        # Compile to machine code
        mod, ker, shared_mem, ir_asm = _triton.code_gen.add_passes_to_emit_bin(
            generator.module, tt_device, num_warps, num_stages, force_nc_cache
        )
        if shared_mem > tt_device.max_shared_memory():
            raise OutOfResources(
                shared_mem, tt_device.max_shared_memory(), "shared memory"
            )
        return Binary(
            mod, ker, num_warps, num_stages, force_nc_cache, shared_mem, ir_asm
        )

    def aot_compile(
        self, *wargs, num_warps=4, num_stages=2, force_nc_cache=False, **meta
    ):

        tensor_idxs = [i for i, arg in enumerate(wargs) if hasattr(arg, "data_ptr")]
        if len(tensor_idxs) == 0:
            raise ValueError("No Tensor argument found.")
        device = wargs[tensor_idxs[0]].device

        attributes = {
            i: Kernel.pow2_divisor(a) for i, a in enumerate(wargs) if isinstance(a, int)
        }
        # transforms ints whose value is one into constants for just-in-time compilation
        constants = {
            i: arg for i, arg in enumerate(wargs) if isinstance(arg, int) and arg == 1
        }

        attr_sign = "".join([v.__repr__() for v in wargs if "," in v.__repr__()])

        bin_ =  self._compile(
            *wargs,
            device=device,
            attributes=attributes,
            num_warps=num_warps,
            num_stages=num_stages,
            force_nc_cache=force_nc_cache,
            constants=constants,
            **meta,
        )
        return attr_sign, bin_
