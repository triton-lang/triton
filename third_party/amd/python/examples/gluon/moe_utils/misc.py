import torch
from typing import TypeVar
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
from triton.runtime.jit import JITFunction
from triton.experimental.gluon._runtime import GluonJITFunction, jit
from triton_kernels.target_info import is_cuda, get_cdna_version, cuda_capability_geq
from triton_kernels.tensor import FP4
from triton_kernels.numerics import InFlexData
from triton_kernels.numerics_details.mxfp import downcast_to_mxfp
from triton_kernels.tensor import wrap_torch_tensor, convert_layout

T = TypeVar("T")


def _import_from_triton(fn: JITFunction[T]) -> GluonJITFunction[T]:
    # Wrap the function and preserve its original docstring
    gluon_fn = jit(fn.fn)
    gluon_fn.__doc__ = fn.__doc__
    return gluon_fn


# Adapted from https://github.com/triton-lang/triton/blob/53b0eafd76debe074965a5d751dd21c593097eb2/python/triton_kernels/bench/bench_mlp.py#L35
def quantize_weight(w, dtype, value_layout=None, scale_layout=None):
    if dtype == "bf16":
        wq = w.to(torch.bfloat16).transpose(-1, -2).contiguous().transpose(-1, -2)
        return wq, InFlexData(), None
    elif dtype == "fp8":
        fp8e4_dtype = torch.float8_e4m3fn if get_cdna_version() != 3 else torch.float8_e4m3fnuz
        wq = w.to(fp8e4_dtype)
        if is_cuda() and not cuda_capability_geq(10, 0):
            wq = wq.transpose(-1, -2).contiguous().transpose(-1, -2)
        return wq, InFlexData(dtype=wq.dtype, scale=w.abs().max().unsqueeze(0)), None
    else:
        assert dtype == "mx4", f"{dtype=}"
        w, w_scale = downcast_to_mxfp(w.to(torch.bfloat16), torch.uint8, axis=1)
        if value_layout is not None:
            w = convert_layout(wrap_torch_tensor(w, dtype=FP4), value_layout)
        if scale_layout is not None:
            w_scale = convert_layout(wrap_torch_tensor(w_scale), scale_layout)
        return w, InFlexData(), w_scale


@gluon.constexpr_function
def get_scaled_dot_format_string(dtype: gl.dtype):
    mapping = {
        gl.float16: "fp16",
        gl.bfloat16: "bf16",
        gl.uint8: "e2m1",
        gl.float8e4nv: "e4m3",
        gl.float8e5: "e5m2",
    }
    return mapping[dtype]


class DType:

    def __init__(self, dtype_str):
        self.has_global_scale = dtype_str.startswith("float8")
        self.has_mx_scale = dtype_str.startswith("mx")
        to_torch_dtype = lambda name: torch.uint8 if name == "float4_e2m1" else getattr(torch, name)
        self.torch_dtype = to_torch_dtype(dtype_str.strip("mx"))
        self.is_mxfloat4 = self.has_mx_scale and "float4" in dtype_str
