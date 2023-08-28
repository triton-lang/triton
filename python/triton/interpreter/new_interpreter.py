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


def to_numpy_ty(ty):
    typemap = {
        tl.float16: np.float16,
        tl.bfloat16: np.float16,
        tl.float32: np.float32,
        tl.float64: np.float64,
        tl.int8: np.int8,
        tl.uint8: np.uint8,
    }
    return typemap[ty]


class Builder:

    def __init__(self) -> None:
        pass

    def create_load(self, ptr, _0, _1, isVolatile):
        return _interpreter.load_ptrs(ptr, np.dtype(np.float32))

    def create_store(self, ptr, val, _0, _1):
        return _interpreter.store_ptrs(ptr, val)

    def create_fadd(self, lhs, rhs):
        return lhs + rhs


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
        lang = [value for key, value in fn.__globals__.items() if value is tl]
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
            return tl.tensor(np.array([arg.data_ptr()], dtype=np.uint64), ty)
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
