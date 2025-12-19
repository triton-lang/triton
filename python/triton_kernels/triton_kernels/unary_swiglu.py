import torch

from .swiglu import PrecisionConfig as SwiGLUPrecisionConfig
from .swiglu import swiglu_torch
from .unary_swiglu_details._unary_swiglu import _unary_swiglu_fn

unary_swiglu_fn = _unary_swiglu_fn


def unary_swiglu_torch(x: torch.Tensor, alpha: float) -> torch.Tensor:
    packed = torch.repeat_interleave(x, 2, dim=-1)
    return swiglu_torch(packed, alpha, SwiGLUPrecisionConfig(limit=None))
