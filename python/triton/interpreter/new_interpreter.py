import inspect

import numpy as np

import triton
import triton.language as tl
from .._C.libtriton.triton import interpreter as _interpreter


# TODO: duplicate
def str_to_ty(name):
    language = tl
    if name[0] == "*":
        ty = str_to_ty(name[1:])
        return language.pointer_type(ty)
    tys = {
        "fp8e4nv": language.float8e4nv,
        "fp8e5": language.float8e5,
        "fp8e4b15": language.float8e4b15,
        "fp8e4b15x4": language.float8e4b15x4,
        "fp16": language.float16,
        "bf16": language.bfloat16,
        "fp32": language.float32,
        "fp64": language.float64,
        "i1": language.int1,
        "i8": language.int8,
        "i16": language.int16,
        "i32": language.int32,
        "i64": language.int64,
        "u8": language.uint8,
        "u16": language.uint16,
        "u32": language.uint32,
        "u64": language.uint64,
        "B": language.int1,
    }
    return tys[name]


class TensorHandle:

    def __init__(self, data, dtype):
        self.data = data
        self.dtype = dtype


def wrap_ret(compute_ret_ty):
    def wrapper(fn):
        def wrapped(*args, **kwargs):
            ret = fn(*args, **kwargs)
            return TensorHandle(ret.data, compute_ret_ty(*args, **kwargs))
        return wrapped
    return wrapper


class Builder:

    def np_dtype(self, tt_dtype):
        if isinstance(tt_dtype, tl.pointer_type):
            return np.dtype(np.uint64)
        np_types = {
            tl.float16: np.dtype(np.float16),
            tl.float32: np.dtype(np.float32),
            tl.float64: np.dtype(np.float64),
            tl.int8: np.dtype(np.int8),
            tl.uint8: np.dtype(np.uint8),
            tl.int16: np.dtype(np.int16),
            tl.uint16: np.dtype(np.uint16),
            tl.int32: np.dtype(np.int32),
            tl.uint32: np.dtype(np.uint32),
            tl.int64: np.dtype(np.int64),
            tl.uint64: np.dtype(np.uint64),
        }
        return np_types[tt_dtype]

    def __init__(self) -> None:
        pass

    # constants
    def get_int32(self, value):
        return TensorHandle(np.array([value], dtype=np.int32), tl.int32)

    # programming model
    def create_get_program_id(self, axis):
        pass

    def create_get_num_programs(self, axis):
        pass

    # memory ops
    def create_load(self, ptr, _0, _1, volatile):
        dtype_tt = ptr.dtype.element_ty
        dtype_np = self.np_dtype(dtype_tt)
        ret = _interpreter.load_ptrs(ptr.data, dtype_np)
        return TensorHandle(ret, dtype_tt)

    def create_store(self, ptr, val, _0, _1):
        return _interpreter.store_ptrs(ptr.data, val.data)

    # casting ops
    def cast_impl(self, src, dst_type):
        return TensorHandle(src.data.astype(self.np_dtype(dst_type)), dst_type)
    create_si_to_fp = lambda self, src, dst_type: self.cast_impl(src, dst_type)
    create_ui_to_fp = lambda self, src, dst_type: self.cast_impl(src, dst_type)
    create_fp_to_si = lambda self, src, dst_type: self.cast_impl(src, dst_type)
    create_fp_to_ui = lambda self, src, dst_type: self.cast_impl(src, dst_type)
    create_fp_ext = lambda self, src, dst_type: self.cast_impl(src, dst_type)
    create_fp_trunc = lambda self, src, dst_type: self.cast_impl(src, dst_type)
    create_int_cast = lambda self, src, dst_type, is_signed: self.cast_impl(src, dst_type)

    def create_fp_to_fp(self, src, dst_type):
        assert "float8 not NotImplemented yet"

    def create_bitcast(self, src, dst_type):
        return TensorHandle(src.data.view(self.np_dtype(dst_type)), dst_type)

    # binary operators
    def binary_op(self, lhs, rhs, op):
        return TensorHandle(op(lhs.data, rhs.data), lhs.dtype)

    create_fadd = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.add)
    create_fmul = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.multiply)
    create_fdiv = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.divide)
    create_frem = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.remainder)
    create_fsub = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.subtract)
    create_mul = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.multiply)
    create_sdiv = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.divide)
    create_udiv = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.divide)
    create_srem = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.remainder)
    create_urem = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.remainder)
    create_add = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.add)
    create_sub = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.subtract)
    create_shl = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.left_shift)
    create_lshr = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.right_shift)
    create_ashr = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.right_shift)
    create_minsi = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.minimum)
    create_minui = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.minimum)
    create_minf = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.minimum)
    create_maxsi = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.maximum)
    create_maxui = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.maximum)
    create_maxf = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.maximum)
    create_icmpSLE = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.less_equal)
    create_icmpSLT = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.less)
    create_icmpSGE = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.greater_equal)
    create_icmpSGT = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.greater)
    create_icmpULE = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.less_equal)
    create_icmpULT = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.less)
    create_icmpUGE = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.greater_equal)
    create_icmpUGT = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.greater)
    create_icmpEQ = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.equal)
    create_icmpNE = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.not_equal)
    create_fcmpOLT = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.less)
    create_fcmpOGT = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.greater)
    create_fcmpOLE = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.less_equal)
    create_fcmpOGE = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.greater_equal)
    create_fcmpOEQ = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.equal)
    create_fcmpONE = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.not_equal)
    create_fcmpULT = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.less)
    create_fcmpUGT = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.greater)
    create_fcmpULE = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.less_equal)
    create_fcmpUGE = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.greater_equal)
    create_fcmpUEQ = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.equal)
    create_fcmpUNE = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.not_equal)
    create_and = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.bitwise_and)
    create_xor = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.bitwise_xor)
    create_or = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.bitwise_or)

    # ternary functions
    def ternary_op(self, lhs, rhs, other, op):
        return TensorHandle(op(lhs.data, rhs.data, other.data), other.dtype)
    create_select = lambda self, cond, lhs, rhs: self.ternary_op(cond, lhs, rhs, np.where)

    # unary functions
    def unary_op(self, arg, op):
        return TensorHandle(op(arg.data), arg.dtype)
    create_exp = lambda self, arg: self.unary_op(arg, np.exp)
    create_cos = lambda self, arg: self.unary_op(arg, np.cos)
    create_sin = lambda self, arg: self.unary_op(arg, np.sin)
    create_log = lambda self, arg: self.unary_op(arg, np.log)
    create_sqrt = lambda self, arg: self.unary_op(arg, np.sqrt)
    create_fabs = lambda self, arg: self.unary_op(arg, np.abs)
    create_iabs = lambda self, arg: self.unary_op(arg, np.abs)

    # tensor operators
    create_dot = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.dot)
    create_view = lambda self, arg, shape: TensorHandle(arg.data.reshape(shape), arg.dtype)
    create_trans = lambda self, arg: self.unary_op(arg, np.transpose)

    def create_make_range(self, start, stop):
        return TensorHandle(np.arange(start, stop, dtype=np.int32), tl.int32)

    # pointer arithmetic

    def create_addptr(self, ptr, offset):
        dtype_tt = ptr.dtype.element_ty
        return TensorHandle(ptr.data + (dtype_tt.primitive_bitwidth // 8) * offset.data, ptr.dtype)

    # def create_tensor_pointer_load(self, ptr, boundaryCheck, paddingOption, cacheModifier, evictionPolicy, isVolatile):
    #     pass

    # def create_tensor_pointer_store(self, ptr, value, boundaryCheck, cacheModifier, evictionPolicy):
    #     pass

    # def create_masked_load(self, ptrs, mask, other, cacheModifier, evictionPolicy, isVolatile):
    #     pass

    # def create_masked_store(self, ptrs, value, mask, cacheModifier, evictionPolicy):
    #     pass

    def create_expand_dims(self, arg, axis):
        return TensorHandle(np.expand_dims(arg.data, axis), arg.dtype)

    def create_broadcast(self, arg, shape):
        return TensorHandle(np.broadcast_to(arg.data, shape), arg.dtype)

    # def create_cat(self, lhs, rhs):
    #     pass

    # def create_broadcast(self, arg, shape):
    #     pass

    def create_splat(self, arg, shape):
        return TensorHandle(np.full(shape, arg.data[0], dtype=self.np_dtype(arg.dtype)), arg.dtype)

    # def create_atomic_cas(self, ptr, cmp, val, sem):
    #     pass

    # def create_atomic_rmw(self, rmwOp, ptr, val, mask, sem):
    #     pass

    # def create_extern_elementwise(self, libName, libPath, symbol, argList, retType, isPure):
    #     pass

    # def create_reduce(self, operands, axis):
    #     pass

    # def create_reduce_ret(self, args):
    #     pass

    # def create_scan(self, operands, axis):
    #     pass

    # def create_scan_ret(self, args):
    #     pass

    # def create_ptr_to_int(self, val, type):
    #     pass

    # def create_int_to_ptr(self, val, type):
    #     pass

    # def create_inline_asm(self, inlineAsm, constraints, values, type, isPure, pack):
    #     pass

    # def create_print(self, prefix, values):
    #     pass

    # def create_assert(self, condition, message, fileName, funcName, lineNo):
    #     pass

    # def create_undef(self, type):
    #     pass

    # def create_barrier(self):
    #     pass

    # def create_make_block_ptr(self, base, shape, strides, offsets, tensorShape, order):
    #     pass

    # def create_advance(self, ptr, offsets):
    #     pass


class Interpreter:

    @staticmethod
    def patch_attr(obj, name, member, builder):
        new_member = lambda *args, member=member, **kwargs: (member(*args, **{k: v for k, v in kwargs.items() if k != '_builder'}, _builder=builder))
        setattr(obj, name, new_member)

    @staticmethod
    def _patch_lang_tensor(tensor, builder):
        for name, member in inspect.getmembers(tensor):
            if tl.core.is_builtin(member):
                Interpreter.patch_attr(tensor, name, member, builder)

    @staticmethod
    def _patch_lang_core(lang, builder):
        for name, member in inspect.getmembers(lang):
            if tl.core.is_builtin(member):
                Interpreter.patch_attr(lang, name, member, builder)

    @staticmethod
    def _patch_lang(fn):
        builder = Builder()
        lang = [value for _, value in fn.__globals__.items() if value is tl]
        assert len(lang) == 1, "triton.language must be visible from within jit'd function"
        Interpreter._patch_lang_tensor(getattr(lang[0], 'tensor'), builder)
        Interpreter._patch_lang_core(lang[0], builder)

    def __init__(self, fn, grid) -> None:
        self.grid = grid
        self.fn = fn
        Interpreter._patch_lang(fn)

    @staticmethod
    def _implicit_cvt(arg):
        if hasattr(arg, 'data_ptr'):
            ty = str_to_ty(triton.runtime.jit.JITFunction._type_of(triton.runtime.jit.JITFunction._key_of(arg)))
            handle = TensorHandle(np.array([arg.data_ptr()], dtype=np.uint64), ty)
            return tl.tensor(handle, ty)
        return arg

    def __call__(self, *args_dev, **kwargs):
        # we need to copy arguments to the host for the interpreter
        args_hst = [arg.cpu() if hasattr(arg, 'data_ptr') else arg for arg in args_dev]
        # implicitly convert tensor arguments to their base pointers
        wrapped_args = [self._implicit_cvt(arg) for arg in args_hst]
        # run function
        self.fn(*wrapped_args, **kwargs)
        # copy arguments back to propagate side-effects
        for arg_dev, arg_hst in zip(args_dev, args_hst):
            if hasattr(arg_dev, 'data_ptr'):
                arg_dev.copy_(arg_hst.to(arg_dev.device))


class InterpretedFunction:

    def __init__(self, fn) -> None:
        self.fn = fn

    def __getitem__(self, grid):
        return Interpreter(self.fn, grid)

    def __call__(self, *args, **kwargs):
        return Interpreter(self.fn, None)(*args, **kwargs)
