import torch
from .swizzle_details.hopper_value import swizzle_mxfp4_value_hopper, unswizzle_mxfp4_value_hopper_torch
from .swizzle_details.hopper_scale import swizzle_mxfp4_scale_hopper, unswizzle_mxfp4_scale_hopper_torch
from .swizzle_details.blackwell_scale import swizzle_mx_scale_bw, unswizzle_mx_scale_bw_torch
from enum import Enum


class SwizzlingType(Enum):
    HOPPER_VALUE = 0
    HOPPER_SCALE = 1
    BLACKWELL_SCALE = 2


def perm_to_contig(ndim: int, axis: int, swizzle_axis: int | None = None) -> tuple[int, ...]:
    """
    Permute the shape so that axis is the last dimension and swizzle_axis is the second to last dimension.
    """
    # FIXME(Lezcano): This API is not very good as it's too generic.
    # Chances are we just care about the cases
    # - axis=-2 and swizzle_axis=-1
    # - axis=-1 and swizzle_axis=-2
    # - axis=anything and swizzle_axis=None
    # We could probably just implement
    # perm_to_contig(ndim, transpose: bool)
    # where we transpose the last two dimensions if transpose is True and otherwise we leave them as is.
    axis = axis if axis >= 0 else axis + ndim
    if swizzle_axis is not None:
        swizzle_axis = swizzle_axis if swizzle_axis >= 0 else swizzle_axis + ndim

    assert axis != swizzle_axis
    shape = list(range(ndim))
    shape[axis], shape[-1] = shape[-1], shape[axis]
    if swizzle_axis is not None:
        if swizzle_axis == len(shape) - 1:
            swizzle_axis = axis
        shape[swizzle_axis], shape[-2] = shape[-2], shape[swizzle_axis]
    return tuple(shape)


def perm_from_contig(ndim: int, axis: int, swizzle_axis: int | None = None) -> tuple[int, ...]:
    # Invert the permutation via argsort
    perm = perm_to_contig(ndim, axis, swizzle_axis)
    inv = [0] * ndim
    for i, v in enumerate(perm):
        inv[v] = i
    return tuple(inv)


def perm_tensor_to_contig(x: torch.Tensor, axis: int, swizzle_axis: int | None = None) -> torch.Tensor:
    """
    Permute the tensor x moving axis to the last dimension and swizzle_axis to the second to last dimension.
    """
    return x.permute(perm_to_contig(x.ndim, axis, swizzle_axis))


def perm_tensor_from_contig(x: torch.Tensor, axis: int, swizzle_axis: int | None = None) -> torch.Tensor:
    """
    Permute the tensor x moving the last dimension to axis and the second to last dimension to swizzle_axis.
    """
    return x.permute(perm_from_contig(x.ndim, axis, swizzle_axis))


def swizzle(tensor, axis, swizzle_axis, swizzle_mode):
    # Permute the tensor so that axis is the last dimension and swizzle_axis is the second to last dimension.
    perm = list(range(tensor.ndim))
    perm[tensor.ndim - 1], perm[axis] = axis, tensor.ndim - 1
    if swizzle_axis is not None:
        perm[tensor.ndim - 2], perm[swizzle_axis] = swizzle_axis, tensor.ndim - 2
    tensor = torch.permute(tensor, perm).contiguous()
    if swizzle_mode == SwizzlingType.HOPPER_VALUE:
        tensor = swizzle_mxfp4_value_hopper(tensor, op_idx=0, mma_version=3)
    elif swizzle_mode == SwizzlingType.BLACKWELL_SCALE:
        tensor = swizzle_mx_scale_bw(tensor, allow_pad=True)
    elif swizzle_mode == SwizzlingType.HOPPER_SCALE:
        tensor = swizzle_mxfp4_scale_hopper(tensor, num_warps=8)
    tensor = perm_tensor_from_contig(tensor, axis, swizzle_axis)
    return tensor


def unswizzle(tensor, axis, swizzle_axis, swizzle_mode):
    tensor = perm_tensor_to_contig(tensor, axis, swizzle_axis)
    if swizzle_mode == SwizzlingType.HOPPER_VALUE:
        tensor = unswizzle_mxfp4_value_hopper_torch(tensor, op_idx=0, mma_version=3)
    elif swizzle_mode == SwizzlingType.BLACKWELL_SCALE:
        tensor = unswizzle_mx_scale_bw_torch(tensor)
    elif swizzle_mode == SwizzlingType.HOPPER_SCALE:
        tensor = unswizzle_mxfp4_scale_hopper_torch(tensor, num_warps=8)
    # permute
    ndim = tensor.ndim
    perm = list(range(ndim))
    perm[ndim - 1], perm[axis] = axis, ndim - 1
    if swizzle_axis is not None:
        perm[ndim - 2], perm[swizzle_axis] = swizzle_axis, ndim - 2
    return torch.permute(tensor, perm).contiguous()
