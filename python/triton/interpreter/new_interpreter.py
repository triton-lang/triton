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

    def create_load(self, ptr, _0, _1, volatile):
        dtype_tt = ptr.dtype.element_ty
        dtype_np = self.np_dtype(dtype_tt)
        ret = _interpreter.load_ptrs(ptr.data, dtype_np)
        return TensorHandle(ret, dtype_tt)

    def create_store(self, ptr, val, _0, _1):
        return _interpreter.store_ptrs(ptr.data, val.data)

    def create_fadd(self, lhs, rhs):
        # assert lhs.dtype is not tl.bfloat16
        return TensorHandle(lhs.data + rhs.data, lhs.dtype)

    # casting ops
    def cast_impl(self, src, dst_type):
        return TensorHandle(src.data.astype(self.np_dtype(dst_type)), dst_type)

    def create_fp_to_fp(self, src, dst_type):
        pass

    def create_si_to_fp(self, src, dst_type):
        return self.cast_impl(src, dst_type)

    def create_ui_to_fp(self, src, dst_type):
        pass

    def create_fp_to_si(self, src, dst_type):
        pass

    def create_fp_to_ui(self, src, dst_type):
        pass

    def create_fp_ext(self, src, dst_type):
        pass

    def create_fp_trunc(self, src, dst_type):
        pass

    def create_int_cast(self, src, dst_type, is_signed):
        pass

    def create_bitcast(self, src, dst_type):
        pass

    def create_to_index(self, input):
        pass

    def create_index_to_si(self, input):
        pass

    def create_fmul(self, lhs, rhs):
        pass

    def create_fdiv(self, lhs, rhs):
        pass

    def create_frem(self, lhs, rhs):
        pass

    def create_fsub(self, lhs, rhs):
        pass

    def create_mul(self, lhs, rhs):
        pass

    def create_sdiv(self, lhs, rhs):
        pass

    def create_udiv(self, lhs, rhs):
        pass

    def create_srem(self, lhs, rhs):
        pass

    def create_urem(self, lhs, rhs):
        pass

    def create_add(self, lhs, rhs):
        pass

    def create_sub(self, lhs, rhs):
        pass

    def create_shl(self, lhs, rhs):
        pass

    def create_lshr(self, lhs, rhs):
        pass

    def create_ashr(self, lhs, rhs):
        pass

    def create_minsi(self, lhs, rhs):
        pass

    def create_minui(self, lhs, rhs):
        pass

    def create_minf(self, lhs, rhs):
        pass

    def create_maxsi(self, lhs, rhs):
        pass

    def create_maxui(self, lhs, rhs):
        pass

    def create_maxf(self, lhs, rhs):
        pass

    def create_addptr(self, ptr, offset):
        pass

    def create_icmpSLE(self, lhs, rhs):
        pass

    def create_icmpSLT(self, lhs, rhs):
        pass

    def create_icmpSGE(self, lhs, rhs):
        pass

    def create_icmpSGT(self, lhs, rhs):
        pass

    def create_icmpULE(self, lhs, rhs):
        pass

    def create_icmpULT(self, lhs, rhs):
        pass

    def create_icmpUGE(self, lhs, rhs):
        pass

    def create_icmpUGT(self, lhs, rhs):
        pass

    def create_icmpEQ(self, lhs, rhs):
        pass

    def create_icmpNE(self, lhs, rhs):
        pass

    def create_fcmpOLT(self, lhs, rhs):
        pass

    def create_fcmpOGT(self, lhs, rhs):
        pass

    def create_fcmpOLE(self, lhs, rhs):
        pass

    def create_fcmpOGE(self, lhs, rhs):
        pass

    def create_fcmpOEQ(self, lhs, rhs):
        pass

    def create_fcmpONE(self, lhs, rhs):
        pass

    def create_fcmpULT(self, lhs, rhs):
        pass

    def create_fcmpUGT(self, lhs, rhs):
        pass

    def create_fcmpULE(self, lhs, rhs):
        pass

    def create_fcmpUGE(self, lhs, rhs):
        pass

    def create_fcmpUEQ(self, lhs, rhs):
        pass

    def create_fcmpUNE(self, lhs, rhs):
        pass

    def create_and(self, lhs, rhs):
        pass

    def create_xor(self, lhs, rhs):
        pass

    def create_or(self, lhs, rhs):
        pass

    def create_tensor_pointer_load(self, ptr, boundaryCheck, paddingOption, cacheModifier, evictionPolicy, isVolatile):
        pass

    def create_tensor_pointer_store(self, ptr, value, boundaryCheck, cacheModifier, evictionPolicy):
        pass

    def create_masked_load(self, ptrs, mask, other, cacheModifier, evictionPolicy, isVolatile):
        pass

    def create_masked_store(self, ptrs, value, mask, cacheModifier, evictionPolicy):
        pass

    def create_view(self, arg, shape):
        pass

    def create_expand_dims(self, arg, axis):
        pass

    def create_cat(self, lhs, rhs):
        pass

    def create_trans(self, arg):
        pass

    def create_broadcast(self, arg, shape):
        pass

    def create_splat(self, arg, shape):
        pass

    def create_atomic_cas(self, ptr, cmp, val, sem):
        pass

    def create_atomic_rmw(self, rmwOp, ptr, val, mask, sem):
        pass

    def create_extern_elementwise(self, libName, libPath, symbol, argList, retType, isPure):
        pass

    def create_get_program_id(self, axis):
        pass

    def create_get_num_programs(self, axis):
        pass

    def create_dot(self, a, b, c, allowTF32):
        pass

    def create_exp(self, val):
        pass

    def create_cos(self, val):
        pass

    def create_sin(self, val):
        pass

    def create_log(self, val):
        pass

    def create_sqrt(self, val):
        pass

    def create_fabs(self, val):
        pass

    def create_iabs(self, val):
        pass

    def create_reduce(self, operands, axis):
        pass

    def create_reduce_ret(self, args):
        pass

    def create_scan(self, operands, axis):
        pass

    def create_scan_ret(self, args):
        pass

    def create_ptr_to_int(self, val, type):
        pass

    def create_int_to_ptr(self, val, type):
        pass

    def create_select(self, condition, trueValue, falseValue):
        pass

    def create_inline_asm(self, inlineAsm, constraints, values, type, isPure, pack):
        pass

    def create_print(self, prefix, values):
        pass

    def create_assert(self, condition, message, fileName, funcName, lineNo):
        pass

    def create_undef(self, type):
        pass

    def create_barrier(self):
        pass

    def create_make_block_ptr(self, base, shape, strides, offsets, tensorShape, order):
        pass

    def create_advance(self, ptr, offsets):
        pass


class Interpreter:

    @staticmethod
    def patch_attr(obj, name, member, builder):
        new_member = lambda *args, member=member, **kwargs: (member(*args, **kwargs, _builder=builder))
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
        args_hst = [arg.cpu() for arg in args_dev]
        # implicitly convert tensor arguments to their base pointers
        wrapped_args = [self._implicit_cvt(arg) for arg in args_hst]
        # run function
        self.fn(*wrapped_args, **kwargs)
        # copy arguments back to propagate side-effects
        for arg_dev, arg_hst in zip(args_dev, args_hst):
            arg_dev.copy_(arg_hst.to(arg_dev.device))


class InterpretedFunction:

    def __init__(self, fn) -> None:
        self.fn = fn

    def __getitem__(self, grid):
        return Interpreter(self.fn, grid)

    def __call__(self, *args, **kwargs):
        return Interpreter(self.fn, None)(*args, **kwargs)
