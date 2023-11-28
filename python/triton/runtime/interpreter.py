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

    def __bool__(self):
        return bool(self.data.all())


class BlockPointerHandle:

    def __init__(self, base, shape, strides, offsets, tensor_shape, order):
        self.base = base
        self.shape = shape
        self.strides = strides
        self.offsets = offsets
        self.tensor_shape = tensor_shape
        self.order = order

    def materialize_pointers(self, boundary_check):
        dtype_tt = self.base.dtype.element_ty
        n_bytes = dtype_tt.primitive_bitwidth // 8
        tensor_shape = self.tensor_shape
        ptrs = np.broadcast_to(self.base.data, self.tensor_shape)
        masks = np.ones(self.tensor_shape, dtype=bool)
        for dim in range(len(tensor_shape)):
            bcast_dims = [1] * len(tensor_shape)
            bcast_dims[dim] = tensor_shape[dim]
            off = (self.offsets[dim].data + np.arange(tensor_shape[dim])).reshape(bcast_dims)
            ptrs = ptrs + (n_bytes * off * self.strides[dim].data).astype(np.uint64)
            if dim in boundary_check:
                masks = np.logical_and(masks, off < self.shape[dim].data)
        ptrs = TensorHandle(ptrs, self.base.dtype)
        return ptrs, masks


def wrap_ret(compute_ret_ty):

    def wrapper(fn):

        def wrapped(*args, **kwargs):
            ret = fn(*args, **kwargs)
            return TensorHandle(ret.data, compute_ret_ty(*args, **kwargs))

        return wrapped

    return wrapper


class Builder:

    def __init__(self) -> None:
        self.arch = None
        # pass

    def set_grid_idx(self, x, y, z):
        assert x < self.grid_dim[0]
        assert y < self.grid_dim[1]
        assert z < self.grid_dim[2]
        self.grid_idx = (x, y, z)

    def set_grid_dim(self, nx, ny, nz):
        self.grid_dim = (nx, ny, nz)

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

    # constants
    def get_half_ty(self):
        return tl.float16

    def get_float_ty(self):
        return tl.float32

    def get_int64_ty(self):
        return tl.int64

    def get_ptr_ty(self, elt_ty, addr_space):
        return tl.pointer_type(elt_ty, addr_space)

    def get_block_ty(self, dtype, shape):
        return tl.tensor(shape, dtype)

    def get_int32(self, value):
        return TensorHandle(np.array([value], dtype=np.int32), tl.int32)

    def get_int64(self, value):
        return TensorHandle(np.array([value], dtype=np.int64), tl.int64)

    def get_fp16(self, value):
        return TensorHandle(np.array([value], dtype=np.float16), tl.float16)

    def get_fp32(self, value):
        return TensorHandle(np.array([value], dtype=np.float32), tl.float32)

    def get_null_value(self, type):
        return TensorHandle(np.array([0], dtype=self.np_dtype(type)), type)

    # programming model
    def create_get_program_id(self, axis):
        assert self.grid_idx is not None
        return TensorHandle(np.array([self.grid_idx[axis]], dtype=np.int32), tl.int32)

    def create_get_num_programs(self, axis):
        return TensorHandle(np.array([self.grid_dim[axis]], dtype=np.int32), tl.int32)

    # memory ops
    def create_load(self, ptr, _0, _1, is_volatile):
        mask = TensorHandle(np.ones_like(ptr.data, dtype=bool), tl.int1)
        other = None
        return self.create_masked_load(ptr, mask, other, _0, _1, is_volatile)

    def create_store(self, ptr, val, _0, _1):
        mask = TensorHandle(np.ones_like(ptr.data, dtype=bool), tl.int1)
        return self.create_masked_store(ptr, val, mask, None, None)

    def create_masked_load(self, ptrs, mask, other, cache_modifier, eviction_policy, is_volatile):
        dtype_tt = ptrs.dtype.element_ty
        dtype_np = self.np_dtype(dtype_tt)
        if other is None:
            other = TensorHandle(np.ones_like(ptrs.data, dtype=dtype_np), dtype_tt)
        ret = _interpreter.load(ptrs.data, mask.data, other.data, dtype_np)
        return TensorHandle(ret, dtype_tt)

    def create_masked_store(self, ptrs, value, mask, cache_modifier, eviction_policy):
        return _interpreter.store(ptrs.data, value.data, mask.data)

    # casting ops
    def cast_impl(self, src, dst_type):
        if isinstance(dst_type, tl.tensor):
            dst_type = dst_type.dtype
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
    create_sdiv = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.floor_divide)
    create_udiv = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.floor_divide)
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
    create_reshape = lambda self, arg, shape, allowReorder: TensorHandle(arg.data.reshape(shape), arg.dtype)
    create_trans = lambda self, arg: self.unary_op(arg, np.transpose)

    def create_dot(self, a, b, d, allow_tf32, maxNumImpreciseAcc):
        return TensorHandle(np.dot(a.data, b.data) + d.data, a.dtype)

    def create_make_range(self, start, stop):
        return TensorHandle(np.arange(start, stop, dtype=np.int32), tl.int32)

    # pointer arithmetic

    def create_addptr(self, ptr, offset):
        dtype_tt = ptr.dtype.element_ty
        return TensorHandle(ptr.data + (dtype_tt.primitive_bitwidth // 8) * offset.data.astype(np.uint64), ptr.dtype)

    def create_tensor_pointer_load(self, ptr, boundary_check, padding_option, cache_modifier, eviction_policy,
                                   is_volatile):
        ptrs, masks = ptr.materialize_pointers(boundary_check)
        assert padding_option is None
        other = None
        return self.create_masked_load(ptrs, masks, other, cache_modifier, eviction_policy, is_volatile)

    def create_tensor_pointer_store(self, ptr, value, boundary_check, cache_modifier, eviction_policy):
        ptrs, masks = ptr.materialize_pointers(boundary_check)
        return self.create_masked_store(ptrs, value, masks, cache_modifier, eviction_policy)

    def create_expand_dims(self, arg, axis):
        return TensorHandle(np.expand_dims(arg.data, axis), arg.dtype)

    def create_broadcast(self, arg, shape):
        return TensorHandle(np.broadcast_to(arg.data, shape), arg.dtype)

    def create_int_to_ptr(self, val, dst_ty):
        return TensorHandle(val.data.astype(np.uint64), dst_ty)

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

    def create_make_block_ptr(self, base, shape, strides, offsets, tensor_shape, order):
        return BlockPointerHandle(base, shape, strides, np.array(offsets), tensor_shape, order)

    def create_advance(self, ptr, offsets):
        assert len(ptr.offsets) == len(offsets)
        ret = BlockPointerHandle(ptr.base, ptr.shape, ptr.strides, ptr.offsets, ptr.tensor_shape, ptr.order)
        for i in range(len(offsets)):
            ret.offsets[i].data += offsets[i].data
        return ret


def patch_attr(obj, name, member, builder):
    new_member = lambda *args, member=member, **kwargs: (member(*args, **
                                                                {k: v
                                                                 for k, v in kwargs.items()
                                                                 if k != "_builder"}, _builder=builder))
    setattr(obj, name, new_member)


def _patch_lang_tensor(tensor, builder):
    for name, member in inspect.getmembers(tensor):
        if tl.core.is_builtin(member):
            patch_attr(tensor, name, member, builder)
    tensor.__index__ = lambda self: int(self.handle.data)
    tensor.__bool__ = lambda self: True
    tensor.__str__ = lambda self: str(self.handle.data)
    tensor.__getitem__ = lambda self, slices: self.handle.data.__getitem__(slices)


def _patch_lang_core(lang, builder):
    for name, member in inspect.getmembers(lang):
        if tl.core.is_builtin(member):
            patch_attr(lang, name, member, builder)
    # reduce is better off with a separate patch due to how
    # the builder currently interfaces with custom functions

    def _new_reduce(input, axis, combine_fn):
        fn = combine_fn.fn.__name__
        mapping = {
            "maximum": np.max,
            "_sum_combine": np.sum,
        }
        ret = mapping[fn](input.handle.data, axis=axis)
        ret_type = tl.block_type(input.dtype, ret.shape)
        return tl.core.tensor(TensorHandle(ret, input.dtype), ret_type)

    lang.reduce = _new_reduce


def _patch_lang_math(lang, builder):
    math = lang.math
    mapping = {
        "abs": "abs",
        "acos": "arccos",
        "asin": "arcsin",
        "exp2": "exp2",
        "log2": "log2",
        "max": "maximum",
    }

    def make_numpy(name):

        def impl(*args, **kwargs):
            ret_type = args[0].type  # TODO: incorrect
            ret_dtype = args[0].dtype  # TODO: incorrect
            args = [arg.handle.data for arg in args]
            kwargs = {k: v.handle.data for k, v in kwargs.items()}
            ret = getattr(np, mapping[name])(*args, **kwargs)
            ret = tl.core.tensor(TensorHandle(ret, ret_dtype), ret_type)
            return ret

        return impl

    def make_fallback(name):

        def fallback(*args, **kwargs):
            raise NotImplementedError(f"""
{name} not supported in interpreter mode: no known numpy implementation.
If you think that {name} in fact does have a numpy implementation, please add it
to the mapping in python/triton/interpreter/new_interpreter.py:_patch_lang_math.
""")

        return fallback

    for name, member in inspect.getmembers(math):
        if name in mapping:
            setattr(math, name, make_numpy(name))
        else:
            setattr(math, name, make_fallback(name))


# TODO: wrap everything in triton tensors
def _implicit_cvt(arg):
    if isinstance(arg, int):
        ty = str_to_ty(triton.runtime.jit.JITFunction._type_of(triton.runtime.jit.JITFunction._key_of(arg)))
        handle = TensorHandle(np.array([arg], dtype=np.int32), ty)
        return tl.tensor(handle, ty)
    if hasattr(arg, "data_ptr"):
        ty = str_to_ty(triton.runtime.jit.JITFunction._type_of(triton.runtime.jit.JITFunction._key_of(arg)))
        handle = TensorHandle(np.array([arg.data_ptr()], dtype=np.uint64), ty)
        return tl.tensor(handle, ty)
    return arg


def _unwrap(tensor):
    if isinstance(tensor, triton.TensorWrapper):
        return tensor.base
    return tensor


builder = Builder()

RESERVED_KWS = ["num_warps", "num_stages", "num_ctas", "enable_warp_specialization", "enable_fp_fusion"]


class GridExecutor:

    def __init__(self, fn, arg_names, grid):
        from .jit import _normalize_ty  # TODO: modularize

        self.fn = fn
        self.arg_names = arg_names
        self.grid = grid
        __annotations__ = {name: _normalize_ty(ty) for name, ty in fn.__annotations__.items()}
        self.constexprs = [name for name in arg_names if __annotations__.get(name) == "constexpr"]

    def _patch_lang(self, builder):
        lang = [value for _, value in self.fn.__globals__.items() if value in [tl, tl.core]]
        assert len(lang) == 1, "triton.language must be visible from within jit'd function"
        _patch_lang_tensor(getattr(lang[0], "tensor"), builder)
        _patch_lang_core(lang[0], builder)
        _patch_lang_math(lang[0], builder)

    def __call__(self, *args_dev, **kwargs):
        args_hst = [_unwrap(arg).cpu() if hasattr(arg, "data_ptr") else arg for arg in args_dev]
        # removes reserved keywords from kwargs
        kwargs = {k: v for k, v in kwargs.items() if k not in RESERVED_KWS}
        # remaps core language functions to interpreted ones
        self._patch_lang(builder)
        # we need to copy arguments to the host for the interpreter
        # implicitly convert tensor arguments to their base pointers
        args = inspect.getcallargs(self.fn, *args_hst, **kwargs)
        args = {name: arg if name in self.constexprs else _implicit_cvt(arg) for name, arg in args.items()}
        # iterate through grid
        grid = self.grid(args) if callable(self.grid) else self.grid
        assert len(grid) <= 3
        grid = grid + (1, ) * (3 - len(grid))
        builder.set_grid_dim(*grid)
        for x in range(grid[0]):
            for y in range(grid[1]):
                for z in range(grid[2]):
                    builder.set_grid_idx(x, y, z)
                    self.fn(**args)
        # copy arguments back to propagate side-effects
        for arg_dev, arg_hst in zip(args_dev, args_hst):
            if hasattr(arg_dev, "data_ptr"):
                _unwrap(arg_dev).copy_(arg_hst.to(arg_dev.device))


class InterpretedFunction:

    def _patch_lang(self, builder):
        lang = [value for _, value in self.fn.__globals__.items() if value in [tl, tl.core]]
        assert len(lang) == 1, "triton.language must be visible from within jit'd function"
        _patch_lang_tensor(getattr(lang[0], "tensor"), builder)
        _patch_lang_core(lang[0], builder)

    def __init__(self, fn) -> None:
        self.fn = fn

        def run(*args, **kwargs):
            grid = kwargs["grid"]
            kwargs = {k: v for k, v in kwargs.items() if k not in RESERVED_KWS + ["grid"]}

            return GridExecutor(self.fn, self.arg_names, grid)(*args, **kwargs)

        self.run = run
        signature = inspect.signature(fn)
        self.arg_names = [v.name for v in signature.parameters.values()]

    def __getitem__(self, grid):
        return GridExecutor(self.fn, self.arg_names, grid)

    def __call__(self, *args, **kwargs):
        self._patch_lang(builder)
        return self.fn(*args, **kwargs)
