"""isort:skip_file"""
# Import order is significant here.

from . import math
from . import extra
from .standard import (
    argmax,
    argmin,
    cdiv,
    cumprod,
    cumsum,
    flip,
    interleave,
    max,
    min,
    ravel,
    sigmoid,
    softmax,
    sort,
    sum,
    swizzle2d,
    xor_sum,
    zeros,
    zeros_like,
)
from .core import (
    PropagateNan,
    TRITON_MAX_TENSOR_NUMEL,
    _experimental_descriptor_load,
    _experimental_descriptor_store,
    _experimental_make_tensor_descriptor,
    _experimental_reinterpret_tensor_descriptor,
    _experimental_tensor_descriptor,
    add,
    advance,
    arange,
    associative_scan,
    assume,
    atomic_add,
    atomic_and,
    atomic_cas,
    atomic_max,
    atomic_min,
    atomic_or,
    atomic_xchg,
    atomic_xor,
    bfloat16,
    block_type,
    broadcast,
    broadcast_to,
    cat,
    cast,
    clamp,
    const,
    constexpr,
    debug_barrier,
    device_assert,
    device_print,
    dot,
    dot_scaled,
    dtype,
    expand_dims,
    float16,
    float32,
    float64,
    float8e4b15,
    float8e4nv,
    float8e4b8,
    float8e5,
    float8e5b16,
    full,
    histogram,
    inline_asm_elementwise,
    int1,
    int16,
    int32,
    int64,
    int8,
    join,
    load,
    make_block_ptr,
    max_constancy,
    max_contiguous,
    maximum,
    minimum,
    multiple_of,
    num_programs,
    permute,
    pi32_t,
    pointer_type,
    nv_tma_desc_type,
    program_id,
    range,
    reduce,
    reshape,
    slice,
    split,
    static_assert,
    static_print,
    static_range,
    store,
    tensor,
    trans,
    tuple,
    tuple_type,
    uint16,
    uint32,
    uint64,
    uint8,
    view,
    void,
    where,
)
from .math import (umulhi, exp, exp2, fma, log, log2, cos, rsqrt, sin, sqrt, sqrt_rn, abs, fdiv, div_rn, erf, floor,
                   ceil)
from .random import (
    pair_uniform_to_normal,
    philox,
    philox_impl,
    rand,
    rand4x,
    randint,
    randint4x,
    randn,
    randn4x,
    uint_to_uniform_float,
)

__all__ = [
    "PropagateNan",
    "TRITON_MAX_TENSOR_NUMEL",
    "_experimental_descriptor_load",
    "_experimental_descriptor_store",
    "_experimental_make_tensor_descriptor",
    "_experimental_reinterpret_tensor_descriptor",
    "_experimental_tensor_descriptor",
    "abs",
    "add",
    "advance",
    "arange",
    "argmax",
    "argmin",
    "associative_scan",
    "assume",
    "atomic_add",
    "atomic_and",
    "atomic_cas",
    "atomic_max",
    "atomic_min",
    "atomic_or",
    "atomic_xchg",
    "atomic_xor",
    "bfloat16",
    "block_type",
    "broadcast",
    "broadcast_to",
    "cat",
    "cast",
    "cdiv",
    "ceil",
    "clamp",
    "const",
    "constexpr",
    "cos",
    "cumprod",
    "cumsum",
    "debug_barrier",
    "device_assert",
    "device_print",
    "div_rn",
    "dot",
    "dot_scaled",
    "dtype",
    "erf",
    "exp",
    "exp2",
    "expand_dims",
    "extra",
    "fdiv",
    "flip",
    "float16",
    "float32",
    "float64",
    "float8e4b15",
    "float8e4nv",
    "float8e4b8",
    "float8e5",
    "float8e5b16",
    "floor",
    "fma",
    "full",
    "histogram",
    "inline_asm_elementwise",
    "interleave",
    "int1",
    "int16",
    "int32",
    "int64",
    "int8",
    "ir",
    "join",
    "load",
    "log",
    "log2",
    "make_block_ptr",
    "math",
    "max",
    "max_constancy",
    "max_contiguous",
    "maximum",
    "min",
    "minimum",
    "multiple_of",
    "num_programs",
    "pair_uniform_to_normal",
    "permute",
    "philox",
    "philox_impl",
    "pi32_t",
    "pointer_type",
    "nv_tma_desc_type",
    "program_id",
    "rand",
    "rand4x",
    "randint",
    "randint4x",
    "randn",
    "randn4x",
    "range",
    "ravel",
    "reduce",
    "reshape",
    "rsqrt",
    "slice",
    "sigmoid",
    "sin",
    "softmax",
    "sort",
    "split",
    "sqrt",
    "sqrt_rn",
    "static_assert",
    "static_print",
    "static_range",
    "store",
    "sum",
    "swizzle2d",
    "tensor",
    "trans",
    "triton",
    "tuple",
    "uint16",
    "uint32",
    "uint64",
    "uint8",
    "uint_to_uniform_float",
    "umulhi",
    "view",
    "void",
    "where",
    "xor_sum",
    "zeros",
    "zeros_like",
]


def parse_list_string(s):
    s = s.strip()
    if s.startswith('[') and s.endswith(']'):
        s = s[1:-1]
    result = []
    current = ''
    depth = 0
    for c in s:
        if c == '[':
            depth += 1
            current += c
        elif c == ']':
            depth -= 1
            current += c
        elif c == ',' and depth == 0:
            result.append(current.strip())
            current = ''
        else:
            current += c
    if current.strip():
        result.append(current.strip())
    return result



def str_to_ty(name):
    if name == "none":
        return str_to_ty("*i8")

    if name[0] == "*":
        name = name[1:]
        const = False
        if name[0] == "k":
            name = name[1:]
            const = True
        ty = str_to_ty(name)
        return pointer_type(element_ty=ty, const=const)

    if name[0] == "[":
        names = parse_list_string(name)
        tys = [str_to_ty(x) for x in names]
        return tuple_type(types=tys)

    if name == "nvTmaDesc":
        return nv_tma_desc_type()
    
    if name == "constexpr":
        return constexpr

    tys = {
        "fp8e4nv": float8e4nv,
        "fp8e4b8": float8e4b8,
        "fp8e5": float8e5,
        "fp8e5b16": float8e5b16,
        "fp8e4b15": float8e4b15,
        "fp16": float16,
        "bf16": bfloat16,
        "fp32": float32,
        "fp64": float64,
        "i1": int1,
        "i8": int8,
        "i16": int16,
        "i32": int32,
        "i64": int64,
        "u1": int1,
        "u8": uint8,
        "u16": uint16,
        "u32": uint32,
        "u64": uint64,
        "B": int1,
    }
    return tys[name]
