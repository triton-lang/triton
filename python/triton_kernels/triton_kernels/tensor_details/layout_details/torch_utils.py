import torch

# def unpack(data: torch.Tensor, dim: int, is_fp4: bool):
#     if not is_fp4:
#         return data
#     if data.shape[dim] == 1:
#         return data
#     ret_shape = list(data.shape)
#     ret_shape[dim] *= 2
#     ret = torch.empty(ret_shape, dtype=data.dtype, device=data.device)
#     idx_lo = [slice(None)] * data.ndim
#     idx_hi = [slice(None)] * data.ndim
#     idx_lo[dim] = slice(0, data.shape[dim]*2, 2)
#     idx_hi[dim] = slice(1, data.shape[dim]*2, 2)
#     ret[tuple(idx_lo)] = data & 0x0F
#     ret[tuple(idx_hi)] = data & 0xF0
#     ret[tuple(idx_hi)] >>= 4
#     return ret

# def pack(data: torch.Tensor, dim: int, is_fp4: bool):
#     if not is_fp4:
#         return data
#     if data.shape[dim] == 1:
#         return data
#     size = data.shape[dim] // 2
#     idx_lo = [slice(None)] * data.ndim
#     idx_hi = [slice(None)] * data.ndim
#     idx_lo[dim] = slice(0, size*2, 2)
#     idx_hi[dim] = slice(1, size*2, 2)
#     out = (data[tuple(idx_hi)] << 4)
#     out |= data[tuple(idx_lo)]
#     return out

# def repack(data: torch.Tensor, old_dim: int, new_dim: int, is_fp4: bool):
#     old_dim %= data.ndim
#     new_dim %= data.ndim
#     if not is_fp4 or old_dim == new_dim:
#         return data
#     tmp = unpack(data, old_dim, is_fp4)
#     ret = pack(tmp, new_dim, is_fp4)
#     return ret


def repack(data: torch.Tensor, old_dim: int, new_dim: int, is_fp4: bool) -> torch.Tensor:
    old_dim %= data.ndim
    new_dim %= data.ndim
    if (not is_fp4) or (old_dim == new_dim):
        return data
    if data.shape[new_dim] % 2 != 0:
        raise ValueError(f"repack_fp4_generic requires data.shape[new_dim] even; "
                         f"got shape[{new_dim}]={data.shape[new_dim]}")
    if data.dtype.is_floating_point:
        raise TypeError(f"Expected integer dtype for bitwise ops, got {data.dtype}")
    ret_shape = list(data.shape)
    ret_shape[old_dim] *= 2
    ret_shape[new_dim] //= 2
    ret = torch.empty(ret_shape, dtype=data.dtype, device=data.device)

    def _idx(ndim: int, dim: int, sl: slice):
        idx = [slice(None)] * ndim
        idx[dim] = sl
        return tuple(idx)

    # data slices along new_dim (pairwise)
    d_even = _idx(data.ndim, new_dim, slice(0, None, 2))
    d_odd = _idx(data.ndim, new_dim, slice(1, None, 2))
    # ret slices along old_dim (interleave into even/odd positions)
    r_even = _idx(ret.ndim, old_dim, slice(0, None, 2))
    r_odd = _idx(ret.ndim, old_dim, slice(1, None, 2))
    a = data[d_even]
    b = data[d_odd]
    ret[r_even] = (a & 0x0F) | ((b & 0x0F) << 4)
    ret[r_odd] = ((a & 0xF0) >> 4) | (b & 0xF0)
    return ret
