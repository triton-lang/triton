import torch
import triton

from triton.language.target_info import (
    cuda_capability_geq,
    is_cuda,
    is_hip,
    is_hip_cdna3,
    is_hip_cdna4,
    is_hip_gfx1250,
    get_cdna_version,
    get_rdna_version,
)

__all__ = [
    "cuda_capability_geq",
    "has_tma_gather",
    "has_native_mxfp",
    "is_cuda",
    "is_hip",
    "is_hip_cdna3",
    "is_hip_cdna4",
    "is_hip_gfx1250",
    "get_cdna_version",
    "get_rdna_version",
    "num_sms",
]


@triton.constexpr_function
def has_tma_gather():
    return cuda_capability_geq(10, 0)


@triton.constexpr_function
def has_native_mxfp():
    return cuda_capability_geq(10, 0)


def num_sms():
    return torch.cuda.get_device_properties(0).multi_processor_count
