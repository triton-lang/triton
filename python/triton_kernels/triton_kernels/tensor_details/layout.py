import math
import torch
from abc import ABC, abstractmethod
from .layout_details.hopper_value import swizzle_mxfp4_value_hopper, unswizzle_mxfp4_value_hopper_torch
from .layout_details.hopper_scale import swizzle_mxfp4_scale_hopper, unswizzle_mxfp4_scale_hopper_torch


class Layout(ABC):

    def __init__(self, shape) -> None:
        self.initial_shape = shape

    @abstractmethod
    def swizzle_data(self, data):
        pass

    @abstractmethod
    def unswizzle_data(self, data):
        pass

    @abstractmethod
    def swizzle_block_shape(self, block_shape):
        pass


class DefaultLayout(Layout):
    name: str = None

    def __init__(self, shape) -> None:
        super().__init__(shape)

    def swizzle_data(self, data):
        return data

    def unswizzle_data(self, data):
        return data

    def swizzle_block_shape(self, block_shape):
        return block_shape


class BlackwellMXScaleLayout(Layout):
    name: str = "BLACKWELL_SCALE"

    def __init__(self, shape) -> None:
        super().__init__(shape)
        *self.leading_shape, self.K, self.N, = shape
        self.B = math.prod(self.leading_shape)
        self.ALIGN_K = 8
        self.ALIGN_N = 128
        self.SWIZZLE_K = 4
        self.K_pad = (self.K + self.ALIGN_K - 1) // self.ALIGN_K * self.ALIGN_K
        self.N_pad = (self.N + self.ALIGN_N - 1) // self.ALIGN_N * self.ALIGN_N

    def swizzle_data(self, data):
        data = torch.nn.functional.pad(data, (0, self.N_pad - self.N, 0, self.K_pad - self.K))
        data = data.transpose(-1, -2).contiguous()
        data = data.reshape(self.B, self.N_pad // self.ALIGN_N, self.ALIGN_N // 32, 32, self.K_pad // self.SWIZZLE_K,
                            self.SWIZZLE_K)
        data = data.transpose(2, 4).contiguous()
        data = data.view(1, self.B * self.N_pad // 128, self.K_pad // 4, 2, 256)
        return data

    def unswizzle_data(self, data):
        data = data.reshape(self.B, self.N_pad // self.ALIGN_N, self.K_pad // self.SWIZZLE_K, 32, self.ALIGN_N // 32,
                            self.SWIZZLE_K)
        data = data.transpose(2, 4)
        data = data.reshape(*self.leading_shape, self.N_pad, self.K_pad)
        data = data.transpose(-1, -2)
        return data[..., :self.K, :self.N]

    def swizzle_block_shape(self, block_shape):
        MX_PACK_DIVISOR = 32
        MX_SCALE_BLOCK_K = block_shape[1] // MX_PACK_DIVISOR
        return [1, block_shape[0] // 128, MX_SCALE_BLOCK_K // 4, 2, 256]


class HopperMXScaleLayout(Layout):
    name: str = "HOPPER_SCALE"

    def swizzle_data(self, data):
        swizzle_axis = 2
        axis = data.stride().index(1)
        perm = list(range(data.ndim))
        perm[data.ndim - 1], perm[axis] = axis, data.ndim - 1
        if swizzle_axis is not None:
            perm[data.ndim - 2], perm[swizzle_axis] = swizzle_axis, data.ndim - 2
        data = torch.permute(data, perm).contiguous()
        data = swizzle_mxfp4_scale_hopper(data, num_warps=8)
        return perm_tensor_from_contig(data, axis, swizzle_axis)

    def unswizzle_data(self, data):
        swizzle_axis = 2
        axis = data.stride().index(1)
        data = perm_tensor_to_contig(data, axis, swizzle_axis)
        data = unswizzle_mxfp4_scale_hopper_torch(data, num_warps=8)
        ndim = data.ndim
        perm = list(range(ndim))
        perm[ndim - 1], perm[axis] = axis, ndim - 1
        if swizzle_axis is not None:
            perm[ndim - 2], perm[swizzle_axis] = swizzle_axis, ndim - 2
        return torch.permute(data, perm).contiguous()

    def swizzle_block_shape(self, block_shape):
        return block_shape


class HopperMXValueLayout(Layout):
    name: str = "HOPPER_VALUE"

    def swizzle_data(self, data):
        swizzle_axis = 2
        axis = data.stride().index(1)
        perm = list(range(data.ndim))
        perm[data.ndim - 1], perm[axis] = axis, data.ndim - 1
        if swizzle_axis is not None:
            perm[data.ndim - 2], perm[swizzle_axis] = swizzle_axis, data.ndim - 2
        data = torch.permute(data, perm).contiguous()
        data = swizzle_mxfp4_value_hopper(data, op_idx=0, mma_version=3)
        return perm_tensor_from_contig(data, axis, swizzle_axis)

    def unswizzle_data(self, data):
        swizzle_axis = 2
        axis = data.stride().index(1)
        data = perm_tensor_to_contig(data, axis, swizzle_axis)
        data = unswizzle_mxfp4_value_hopper_torch(data, op_idx=0, mma_version=3)
        ndim = data.ndim
        perm = list(range(ndim))
        perm[ndim - 1], perm[axis] = axis, ndim - 1
        if swizzle_axis is not None:
            perm[ndim - 2], perm[swizzle_axis] = swizzle_axis, ndim - 2
        return torch.permute(data, perm).contiguous()

    def swizzle_block_shape(self, block_shape):
        return block_shape


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
