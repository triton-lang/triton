from copy import deepcopy
from dataclasses import dataclass

import triton_kernels
import triton_kernels.swiglu
from triton_kernels.matmul import PrecisionConfig, FnSpecs, FusedActivation
from triton_kernels.numerics_details.mxfp import downcast_to_mxfp
from triton_kernels.tensor import convert_layout, wrap_torch_tensor, FP4, Tensor
from triton_kernels.target_info import is_cuda, get_cdna_version, cuda_capability_geq, is_hip
from triton_kernels.tensor_details import layout
import torch


def _quantize_weight(w, dtype, **opt):
    if dtype == "bf16":
        wq = w.to(torch.bfloat16).transpose(-1, -2).contiguous().transpose(-1, -2)
        return wrap_torch_tensor(wq)
    elif dtype == "fp8":
        fp8e4_dtype = torch.float8_e4m3fn if get_cdna_version() != 3 else torch.float8_e4m3fnuz
        wq = w.to(fp8e4_dtype)
        if is_cuda() and not cuda_capability_geq(10, 0):
            wq = wq.transpose(-1, -2).contiguous().transpose(-1, -2)
        wq = wrap_torch_tensor(wq)
        wq.scale_global = w.abs().max().unsqueeze(0)
        return wq
    else:
        assert dtype == "mx4", f"{dtype=}"
        w, w_scale = downcast_to_mxfp(w.to(torch.bfloat16), torch.uint8, axis=1)
        if opt:
            w = convert_layout(wrap_torch_tensor(w, dtype=FP4), opt["value_layout"], **opt["value_layout_opts"])
            w_scale = convert_layout(wrap_torch_tensor(w_scale), opt["scale_layout"], **opt["scale_layout_opts"])
        w.scale_mx = w_scale
        return w


@dataclass
class MlpNumerics:
    wg: torch.Tensor | Tensor | None
    w1: torch.Tensor | Tensor | None
    w2: torch.Tensor | Tensor | None
    pcg: PrecisionConfig
    pc1: PrecisionConfig
    pc2: PrecisionConfig
    activation: FusedActivation


def _make_default_mlp_activation() -> FusedActivation:
    return FusedActivation(
        FnSpecs("swiglu", triton_kernels.swiglu.swiglu_fn, ("alpha", "limit"), reduction_n=2),
        (1.0, 1.0),
    )


def _make_mx4_quantization_opts(batch: int, w_dtype: str) -> dict:
    if w_dtype != "mx4" or is_hip():
        return {}
    num_warps = 4 if batch <= 512 and cuda_capability_geq(10, 0) else 8
    value_layout, value_layout_opts = layout.make_default_matmul_mxfp4_w_layout(mx_axis=1)
    scale_layout, scale_layout_opts = layout.make_default_matmul_mxfp4_w_scale_layout(mx_axis=1, num_warps=num_warps)
    return {
        "value_layout": value_layout,
        "value_layout_opts": value_layout_opts,
        "scale_layout": scale_layout,
        "scale_layout_opts": scale_layout_opts,
    }


def prepare_mlp_numerics(batch: int, w_dtype: str, wg, w1, w2) -> MlpNumerics:
    quantization_opts = _make_mx4_quantization_opts(batch, w_dtype)
    wg = _quantize_weight(wg, "bf16")
    w1 = _quantize_weight(w1, w_dtype, **deepcopy(quantization_opts))
    w2 = _quantize_weight(w2, w_dtype, **deepcopy(quantization_opts))
    activation = _make_default_mlp_activation()
    return MlpNumerics(
        wg=wg,
        w1=w1,
        w2=w2,
        pcg=PrecisionConfig(),
        pc1=PrecisionConfig(),
        pc2=PrecisionConfig(),
        activation=activation,
    )


def resolve_x_dtype(x_dtype: str) -> torch.dtype:
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp8": torch.float8_e4m3fn}
    dtype = dtype_map[x_dtype]
    if dtype == torch.float8_e4m3fn and get_cdna_version() == 3:
        return torch.float8_e4m3fnuz
    return dtype
