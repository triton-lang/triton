import torch
import gc


def unpack(data: torch.Tensor, dim: int, is_fp4: bool):
    if not is_fp4:
        return data
    if data.shape[dim] == 1:
        return data
    data_lo = (data >> 0) & 0x0F
    data_hi = (data >> 4) & 0x0F
    return torch.cat([data_lo, data_hi], dim=dim)


def pack(data: torch.Tensor, dim: int, is_fp4: bool):
    if not is_fp4:
        return data
    if data.shape[dim] == 1:
        return data
    data_lo, data_hi = torch.chunk(data, 2, dim=dim)
    data = (data_hi << 4) | data_lo
    return data


def repack(data: torch.Tensor, old_dim: int, new_dim: int, is_fp4: bool):
    if not is_fp4:
        return data
    tmp = unpack(data, old_dim, is_fp4)
    ret = pack(tmp, new_dim, is_fp4)
    del tmp
    gc.collect()
    torch.cuda.empty_cache()
    return ret
