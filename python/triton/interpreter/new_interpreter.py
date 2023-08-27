import inspect

import numpy as np

import triton
import triton.language as tl


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


def make_handle(arg, ty):
    if ty.is_ptr():
        return np.array([arg.data_ptr()], dtype=np.uint64)
    assert False


class Builder:

    def __init__(self) -> None:
        pass


class Interpreter:

    def _make_wrapper(self, arg):
        ty_str = triton.runtime.jit.JITFunction._type_of(triton.runtime.jit.JITFunction._key_of(arg))
        ty = str_to_ty(ty_str)
        handle = make_handle(arg, ty)
        return tl.tensor(handle, ty)

    def _patch_triton_functions(self, fn):
        builder = Builder()
        for key, obj in fn.__globals__.items():
            if obj is not tl:
                continue
            for name, member in inspect.getmembers(obj):
                if tl.core.is_builtin(member):
                    new_member = lambda *args, member=member, **kwargs: (member(*args, **kwargs, _builder=builder))
                    setattr(obj, name, new_member)
            fn.__globals__[key] = obj
        return fn

    def __init__(self, fn, grid) -> None:
        self.fn = self._patch_triton_functions(fn)
        self.grid = grid

    def __call__(self, *args, **kwargs):
        args = [self._make_wrapper(arg) for arg in args]
        self.fn(*args, **kwargs)


class InterpretedFunction:

    def __init__(self, fn) -> None:
        self.fn = fn

    def __getitem__(self, grid):
        return Interpreter(self.fn, grid)

    def __call__(self, *args, **kwargs):
        return Interpreter(self.fn, None)(*args, **kwargs)
