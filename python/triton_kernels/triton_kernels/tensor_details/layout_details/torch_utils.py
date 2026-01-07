import torch
import sys


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
    # if data.ndim == 3:
    #     shape = list(data.shape)
    #     shape[old_dim] = shape[old_dim] * 2
    #     shape[new_dim] = shape[new_dim] // 2
    #     ret = torch.empty(shape, dtype=data.dtype, device=data.device)
    #     assert old_dim % data.ndim != 0
    #     assert new_dim % data.ndim != 0
    #     for i in range(data.shape[0]):
    #         tmp = unpack(data[i], (old_dim % data.ndim) - 1, is_fp4)
    #         ret[i] = pack(tmp, (new_dim % data.ndim) - 1, is_fp4)
    # else:
    print("data refcount", sys.getrefcount(data))
    tmp = unpack(data, old_dim, is_fp4)
    ret = pack(tmp, new_dim, is_fp4)
    return ret


# def repack(data: torch.Tensor, old_dim: int, new_dim: int, is_fp4: bool):
#     """
#     Repack planar-nibble fp4 bytes from packing along old_dim to packing along new_dim,
#     without materializing the fully unpacked tensor.

#     Assumes your current convention:
#       unpack(dim): cat([lo, hi], dim=dim)  (NOT interleaved)
#       pack(dim):   lo | (hi << 4)          (first half = lo plane, second half = hi plane)
#     """
#     if not is_fp4 or old_dim == new_dim:
#         return data
#     ndim = data.ndim
#     old_dim %= ndim
#     new_dim %= ndim

#     # Match your semantics / requirements for packing:
#     if data.shape[new_dim] == 1:
#         # Packing along new_dim would be a no-op in your pack(), so this becomes just "unpack old_dim".
#         # Do it without torch.cat:
#         out_shape = list(data.shape)
#         out_shape[old_dim] *= 2
#         out = data.new_empty(out_shape)

#         L_old = data.shape[old_dim]
#         idx_lo = [slice(None)] * ndim
#         idx_hi = [slice(None)] * ndim
#         idx_lo[old_dim] = slice(0, L_old)
#         idx_hi[old_dim] = slice(L_old, 2 * L_old)

#         out_lo = out[tuple(idx_lo)]
#         out_hi = out[tuple(idx_hi)]

#         torch.bitwise_and(data, 0x0F, out=out_lo)
#         torch.bitwise_right_shift(data, 4, out=out_hi)
#         out_hi &= 0x0F
#         return out

#     if data.shape[new_dim] % 2 != 0:
#         raise ValueError(f"Cannot pack along new_dim={new_dim}: size {data.shape[new_dim]} is not even")

#     # Output shape: old_dim doubles (becomes unpacked), new_dim halves (becomes packed)
#     out_shape = list(data.shape)
#     out_shape[old_dim] *= 2
#     out_shape[new_dim] //= 2
#     out = data.new_empty(out_shape)

#     L_old = data.shape[old_dim]
#     L_new_half = data.shape[new_dim] // 2

#     def _slice(dim, sl):
#         idx = [slice(None)] * ndim
#         idx[dim] = sl
#         return tuple(idx)

#     # Split input along new_dim into the two halves that will become low/high nibbles in output packing
#     in_lo = data[_slice(new_dim, slice(0, L_new_half))]
#     in_hi = data[_slice(new_dim, slice(L_new_half, 2 * L_new_half))]

#     # Split output along old_dim into the two halves (corresponding to low-plane and high-plane from input bytes)
#     out0 = out[_slice(old_dim, slice(0, L_old))]         # will get packed low-nibbles
#     out1 = out[_slice(old_dim, slice(L_old, 2 * L_old))] # will get packed high-nibbles

#     # One reusable scratch buffer (half the output size)
#     tmp = out0.new_empty(out0.shape)

#     # out0 = (in_lo & 0x0F) | ((in_hi & 0x0F) << 4)
#     torch.bitwise_and(in_lo, 0x0F, out=out0)
#     torch.bitwise_and(in_hi, 0x0F, out=tmp)
#     torch.bitwise_left_shift(tmp, 4, out=tmp)
#     torch.bitwise_or(out0, tmp, out=out0)

#     # out1 = ((in_lo >> 4) & 0x0F) | (((in_hi >> 4) & 0x0F) << 4)
#     torch.bitwise_right_shift(in_lo, 4, out=out1)
#     out1 &= 0x0F
#     torch.bitwise_right_shift(in_hi, 4, out=tmp)
#     tmp &= 0x0F
#     torch.bitwise_left_shift(tmp, 4, out=tmp)
#     out1 |= tmp

#     return out

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
