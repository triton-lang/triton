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
    size = data.shape[dim] // 2
    idx_lo = [slice(None)] * data.ndim
    idx_hi = [slice(None)] * data.ndim
    idx_lo[dim] = slice(0, size)
    idx_hi[dim] = slice(size, 2 * size)
    return data[tuple(idx_lo)] | (data[tuple(idx_hi)] << 4)


def repack(data: torch.Tensor, old_dim: int, new_dim: int, is_fp4: bool):
    if not is_fp4:
        return data
    if data.ndim == 3:
        shape = list(data.shape)
        shape[old_dim] = shape[old_dim] * 2
        shape[new_dim] = shape[new_dim] // 2
        ret = torch.empty(shape, dtype=data.dtype, device=data.device)
        assert old_dim % data.ndim != 0
        assert new_dim % data.ndim != 0
        for i in range(data.shape[0]):
            tmp = unpack(data[i], (old_dim % data.ndim) - 1, is_fp4)
            ret[i] = pack(tmp, (new_dim % data.ndim) - 1, is_fp4)
    else:
        tmp = unpack(data, old_dim, is_fp4)
        ret = pack(tmp, new_dim, is_fp4)
        del tmp
        gc.collect()
        torch.cuda.empty_cache()
    return ret


# def repack(data: torch.Tensor, old_dim: int, new_dim: int, is_fp4: bool):
#     if not is_fp4:
#         return data
#     if data.shape[old_dim] == 1 or data.shape[new_dim] == 1:
#         return data
#     # unpack data
#     data_lo = (data >> 0) & 0x0F
#     data_hi = (data >> 4) & 0x0F
#     tmp = torch.cat([data_lo, data_hi], dim=old_dim)
#     # free memory
#     del data_lo, data_hi
#     gc.collect()
#     torch.cuda.empty_cache()
#     # repack data
#     size = tmp.shape[new_dim] // 2
#     idx_lo = [slice(None)] * data.ndim
#     idx_hi = [slice(None)] * data.ndim
#     idx_lo[new_dim] = slice(0, size)
#     idx_hi[new_dim] = slice(size, 2 * size)
#     tmp[tuple(idx_lo)] |= (tmp[tuple(idx_hi)] << 4)
#     ret = tmp[tuple(idx_lo)].clone()
#     del tmp
#     gc.collect()
#     torch.cuda.empty_cache()
#     return ret
