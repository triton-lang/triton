import torch


def unpack(data: torch.Tensor, dim: int, is_fp4: bool):
    if not is_fp4:
        return data
    data_lo = (data >> 0) & 0x0F
    data_hi = (data >> 4) & 0x0F
    return torch.cat([data_lo, data_hi], dim=dim)


def pack(data: torch.Tensor, dim: int, is_fp4: bool):
    if not is_fp4:
        return data
    data_lo, data_hi = torch.chunk(data, 2, dim=dim)
    data = (data_hi << 4) | data_lo
    return data
