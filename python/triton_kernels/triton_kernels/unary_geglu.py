import torch

from .unary_geglu_details._unary_geglu import _unary_geglu_fn

unary_geglu_fn = _unary_geglu_fn


def unary_geglu_torch(x: torch.Tensor, alpha: float) -> torch.Tensor:
    return x * torch.sigmoid(alpha * x) * (x + 1)
