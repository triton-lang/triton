import torch


def unpack(data: torch.Tensor, dim: int, is_fp4: bool):
    if not is_fp4:
        return data
    if data.shape[dim] == 1:
        return data
    data_lo = data & 0x0F
    data_hi = data & 0xF0
    data_hi >>= 4
    return torch.cat([data_lo, data_hi], dim=dim)


def pack(data: torch.Tensor, dim: int, is_fp4: bool):
    if not is_fp4:
        return data
    if data.shape[dim] == 1:
        return data
    size = data.shape[dim] // 2
    idx_lo = [slice(None)] * data.ndim
    idx_hi = [slice(None)] * data.ndim
    idx_lo[dim] = slice(0, size)
    idx_hi[dim] = slice(size, 2 * size)
    return data[tuple(idx_lo)] | (data[tuple(idx_hi)] << 4)


def repack(data: torch.Tensor, old_dim: int, new_dim: int, is_fp4: bool):
    old_dim %= data.ndim
    new_dim %= data.ndim
    if not is_fp4 or old_dim == new_dim:
        return data
    tmp = unpack(data, old_dim, is_fp4)
    ret = pack(tmp, new_dim, is_fp4)
    return ret
