from __future__ import annotations
import triton.language.core as core

type_canonicalisation_dict = {
    "void": "void",
    "bool": "int1",
    "int8": "int8",
    "int16": "int16",
    "int32": "int32",
    "int64": "int64",
    "uint8": "uint8",
    "uint16": "uint16",
    "uint32": "uint32",
    "uint64": "uint64",
    "float8e5": "fp8e5",
    "float8e4nv": "fp8e4nv",
    "float8e4b15": "fp8e4b15",
    "float16": "fp16",
    "bfloat16": "bf16",
    "float32": "fp32",
    "float64": "fp64",
}

def normalize_ty(ty) -> str:
    if isinstance(ty, str):
        ty = ty.strip()
        if ty.startswith("const "):
            ty = ty.removeprefix("const")
            ty = normalize_ty(ty)
            assert ty.startswith("*")
            return "*k" + ty[1:]
        if ty.endswith("*"):
            return "*" + normalize_ty(ty[:-1])
        if ty.startswith("*"):
            return "*" + normalize_ty(ty[1:])
        if ty.startswith("tl."):
            return normalize_ty(ty.removeprefix("tl."))
    elif isinstance(ty, core.pointer_type):
        return f"*{normalize_ty(ty.element_ty)}"
    elif isinstance(ty, core.dtype):
        ty = ty.name
    elif isinstance(ty, type):
        ty = ty.__name__
    else:
        ty = str(ty)
    return type_canonicalisation_dict.get(ty.replace("_t", ""), ty)
