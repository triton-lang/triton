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


def make_handle(arg, ty):
    if ty.is_ptr():
        return np.array([arg.data_ptr()], dtype=np.uint64)
    assert False


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

    def _make_wrapper(self, arg):
        ty_str = triton.runtime.jit.JITFunction._type_of(triton.runtime.jit.JITFunction._key_of(arg))
        ty = str_to_ty(ty_str)
        handle = make_handle(arg, ty)
        return tl.tensor(handle, ty)

    def patch_member(sef, obj, name, member, builder):
        new_member = lambda *args, member=member, **kwargs: (member(*args, **kwargs, _builder=builder))
        setattr(obj, name, new_member)

    def _patch_triton_functions(self, fn):
        builder = Builder()
        for key, obj in fn.__globals__.items():
            if obj is not tl:
                continue
            for name, member in inspect.getmembers(getattr(obj, 'tensor')):
                if tl.core.is_builtin(member):
                    self.patch_member(getattr(obj, 'tensor'), name, member, builder)
            for name, member in inspect.getmembers(obj):
                if tl.core.is_builtin(member):
                    self.patch_member(obj, name, member, builder)
            fn.__globals__[key] = obj
        return fn

    def __init__(self, fn, grid) -> None:
        self.fn = self._patch_triton_functions(fn)
        self.grid = grid

    def __call__(self, *args, **kwargs):
        cpu_args = [arg.cpu() for arg in args]
        wrapped_args = [self._make_wrapper(arg) for arg in cpu_args]
        self.fn(*wrapped_args, **kwargs)
        for arg, new_arg in zip(args, cpu_args):
            arg.copy_(new_arg.to(arg.device))


class InterpretedFunction:

    def __init__(self, fn) -> None:
        self.fn = fn

    def __getitem__(self, grid):
        return Interpreter(self.fn, grid)

    def __call__(self, *args, **kwargs):
        return Interpreter(self.fn, None)(*args, **kwargs)
