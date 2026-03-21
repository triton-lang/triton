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


def repack(data: torch.Tensor, old_dim: int, new_dim: int, is_fp4: bool, out=None) -> torch.Tensor:
    old_dim %= data.ndim
    new_dim %= data.ndim
    if (not is_fp4) or (old_dim == new_dim):
        if out is not None:
            out.copy_(data)
            return out
        return data
    if data.dtype.is_floating_point:
        raise TypeError(f"Expected integer dtype for bitwise ops, got {data.dtype}")
    out_shape = list(data.shape)
    out_shape[old_dim] *= 2
    out_shape[new_dim] //= 2
    if out is None:
        out = torch.empty(out_shape, dtype=data.dtype, device=data.device)

    def _idx(ndim: int, dim: int, sl: slice):
        idx = [slice(None)] * ndim
        idx[dim] = sl
        return tuple(idx)

    # data slices along new_dim (pairwise)
    d_even = _idx(data.ndim, new_dim, slice(0, None, 2))
    d_odd = _idx(data.ndim, new_dim, slice(1, None, 2))
    # out slices along old_dim (interleave into even/odd positions)
    r_even = _idx(out.ndim, old_dim, slice(0, None, 2))
    r_odd = _idx(out.ndim, old_dim, slice(1, None, 2))
    #
    out_even = out[r_even]
    out_odd = out[r_odd]
    a = data[d_even]
    b = data[d_odd]

    # ---- build out_odd first, using out_even as scratch ----
    out_odd.copy_(b)
    out_odd.bitwise_and_(0xF0)  # out_odd = b & 0xF0

    out_even.copy_(a)
    out_even.bitwise_right_shift_(4)  # out_even (scratch) = a >> 4

    out_odd.bitwise_or_(out_even)  # out_odd = (a >> 4) | (b & 0xF0)

    # ---- now build out_even, no tmp by using add_(alpha=16) ----
    out_even.copy_(a)
    out_even.bitwise_and_(0x0F)  # out_even = a & 0x0F
    out_even.add_(b, alpha=16)  # out_even += 16*b  == (b << 4) | (a & 0x0F)

    return out
