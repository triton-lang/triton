from dataclasses import dataclass
import functools
import torch
import triton
import triton.language as tl

MAX_FINITE_FLOAT8E5 = 57344.0
TL_MAX_FINITE_FLOAT8E5 = tl.constexpr(MAX_FINITE_FLOAT8E5)
MAX_FINITE_FLOAT8E4NV = 448.0
TL_MAX_FINITE_FLOAT8E4NV = tl.constexpr(MAX_FINITE_FLOAT8E4NV)
MAX_FINITE_FLOAT8E4B8 = 240.0
TL_MAX_FINITE_FLOAT8E4B8 = tl.constexpr(MAX_FINITE_FLOAT8E4B8)
TL_MAX_FINITE_FLOAT8E4B15 = tl.constexpr(1.750)
TL_MAX_FINITE_FLOAT16 = tl.constexpr(65472.0)

TL_RCP_MAX_FINITE_FLOAT8E5 = tl.constexpr(0x37924925)  # 0x1.24924Ap-16
TL_RCP_MAX_FINITE_FLOAT8E4NV = tl.constexpr(0x3B124925)  # 0x1.24924Ap-9
TL_RCP_MAX_FINITE_FLOAT8E4B8 = tl.constexpr(0x3B888889)  # 0x1.111112p-8
TL_RCP_MAX_FINITE_FLOAT8E4B15 = tl.constexpr(0x3F124925)  # 0x1.24924Ap-1
TL_RCP_MAX_FINITE_FLOAT16 = tl.constexpr(0x37802008)  # 0x1.004010p-16

cached_capabilities = {}


def constexpr_function(f):
    """
    Wraps an arbitrary Python function so that it can be called at
    compile-time on constexpr arguments in a Triton function and
    returns a constexpr result.
    """

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        # de-constexpr arguments and discard the _builder keyword argument:
        args = [getattr(x, "value", x) for x in args]
        kwargs = {k: getattr(v, "value", v) for (k, v) in kwargs.items() if k != "_builder"}

        # call the raw Python function f:
        res = f(*args, **kwargs)

        # convert result back to a Triton constexpr:
        return tl.constexpr(res)

    # disguise the function as a Triton builtin to avoid raising an error
    # that we're calling a non-JIT function from within a Triton kernel:
    wrapper.__triton_builtin__ = True
    wrapper.__module__ = getattr(tl, "__name__", "triton.language")
    return wrapper


def inline_function(f):
    """
    Wraps an arbitrary Python function so that it can be inlined into a Triton function at compile-time.
    """

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)

    # disguise the function as a Triton builtin to avoid raising an error
    # that we're calling a non-JIT function from within a Triton kernel:
    wrapper.__triton_builtin__ = True
    wrapper.__module__ = getattr(tl, "__name__", "triton.language")
    return wrapper


@constexpr_function
def is_hip():
    if "is_hip" not in cached_capabilities:
        cached_capabilities["is_hip"] = torch.cuda.is_available() and torch.version.hip
    return cached_capabilities["is_hip"]


@constexpr_function
def cuda_capability_geq(major, minor=0):
    """
    Determines whether we have compute capability >= (major, minor) and
    returns this as a constexpr boolean. This can be used for guarding
    inline asm implementations that require a certain compute capability.
    """
    if is_hip():
        return False
    if "cuda" not in cached_capabilities:
        if torch.cuda.is_available():
            cached_capabilities["cuda"] = torch.cuda.get_device_capability()
        else:
            cached_capabilities["cuda"] = (0, 0)
    return cached_capabilities["cuda"] >= (major, minor)


@constexpr_function
def num_sms():
    return torch.cuda.get_device_properties(0).multi_processor_count


@constexpr_function
def threads_per_warp():
    if is_hip():
        return 64
    else:
        return 32


@constexpr_function
def get_scaled_dot_format_string(dtype: tl.dtype):
    mapping = {
        tl.float16: "fp16",
        tl.bfloat16: "bf16",
        tl.uint8: "e2m1",
        tl.float8e4nv: "e4m3",
        tl.float8e5: "e5m2",
    }
    return mapping[dtype]


@triton.jit
def max_finite(dtype):
    if dtype == tl.constexpr(tl.float8e5):
        return TL_MAX_FINITE_FLOAT8E5
    elif dtype == tl.constexpr(tl.float8e4nv):
        return TL_MAX_FINITE_FLOAT8E4NV
    elif dtype == tl.constexpr(tl.float8e4b8):
        return TL_MAX_FINITE_FLOAT8E4B8
    elif dtype == tl.constexpr(tl.float8e4b15):
        return TL_MAX_FINITE_FLOAT8E4B15
    elif dtype == tl.constexpr(tl.float16):
        return TL_MAX_FINITE_FLOAT16
    else:
        tl.static_assert(tl.constexpr(False), f"{dtype} not supported in flexpoint")


@triton.jit
def rcp_max_finite(dtype):
    if dtype == tl.constexpr(tl.float8e5):
        return TL_RCP_MAX_FINITE_FLOAT8E5
    elif dtype == tl.constexpr(tl.float8e4nv):
        return TL_RCP_MAX_FINITE_FLOAT8E4NV
    elif dtype == tl.constexpr(tl.float8e4b8):
        return TL_RCP_MAX_FINITE_FLOAT8E4B8
    elif dtype == tl.constexpr(tl.float8e4b15):
        return TL_RCP_MAX_FINITE_FLOAT8E4B15
    elif dtype == tl.constexpr(tl.float16):
        return TL_RCP_MAX_FINITE_FLOAT16
    else:
        tl.static_assert(tl.constexpr(False), f"{dtype} not supported in flexpoint")
