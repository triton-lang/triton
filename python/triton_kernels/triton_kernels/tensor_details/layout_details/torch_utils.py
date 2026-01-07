import torch


def unpack(data: torch.Tensor, dim: int, is_fp4: bool):
    if not is_fp4:
        return data
    if data.shape[dim] == 1:
        return data
    ret_shape = list(data.shape)
    ret_shape[dim] *= 2
    ret = torch.empty(ret_shape, dtype=data.dtype, device=data.device)
    idx_lo = [slice(None)] * data.ndim
    idx_hi = [slice(None)] * data.ndim
    idx_lo[dim] = slice(0, data.shape[dim])
    idx_hi[dim] = slice(data.shape[dim], 2 * data.shape[dim])
    ret[tuple(idx_lo)] = data & 0x0F
    ret[tuple(idx_hi)] = data & 0xF0
    ret[tuple(idx_hi)] >>= 4
    return ret


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
    out = (data[tuple(idx_hi)] << 4)
    out |= data[tuple(idx_lo)]
    return out


def repack(data: torch.Tensor, old_dim: int, new_dim: int, is_fp4: bool):
    old_dim %= data.ndim
    new_dim %= data.ndim
    if not is_fp4 or old_dim == new_dim:
        return data
    tmp = unpack(data, old_dim, is_fp4)
    ret = pack(tmp, new_dim, is_fp4)
    return ret
