from __future__ import annotations

import torch

_CONSTRUCT_STORAGE_FROM_DATA_POINTER = getattr(torch._C, "_construct_storage_from_data_pointer", None)

SHADOW_SIZE_BYTES = 8
SHADOW_GRANULARITY_BYTES = 4


def uint8_cuda_tensor_from_ptr(data_ptr: int, numel: int, device_index: int) -> torch.Tensor:
    if _CONSTRUCT_STORAGE_FROM_DATA_POINTER is None:
        raise RuntimeError("torch._C._construct_storage_from_data_pointer is unavailable.")
    device = torch.device("cuda", device_index)
    storage = _CONSTRUCT_STORAGE_FROM_DATA_POINTER(int(data_ptr), device, int(numel))
    return torch.empty(0, dtype=torch.uint8, device=device).set_(storage, 0, (int(numel), ), (1, ))


def shadow_region(real_ptr: int, real_size_bytes: int, reserve_ptr: int, reserve_size: int) -> tuple[int, int]:
    real_base = reserve_ptr + reserve_size // 2
    word_offset = (real_ptr - real_base) // SHADOW_GRANULARITY_BYTES
    shadow_ptr = reserve_ptr + word_offset * SHADOW_SIZE_BYTES
    shadow_size = ((real_size_bytes + SHADOW_GRANULARITY_BYTES - 1) // SHADOW_GRANULARITY_BYTES) * SHADOW_SIZE_BYTES
    return shadow_ptr, shadow_size
