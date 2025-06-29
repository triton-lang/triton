import torch
from .tensor_details.memory_layout.hopper_value import unswizzle_mxfp4_value_hopper_torch
from .tensor_details.memory_layout.hopper_scale import unswizzle_mxfp4_scale_hopper_torch
from .tensor_details.memory_layout.blackwell_scale import swizzle_mx_scale_bw, unswizzle_mx_scale_bw_torch
from .reduction_details.reduce_bitmatrix import clear_sums, sum_bitmatrix_rows
from enum import Enum
from dataclasses import dataclass
from triton.tools.tensor_descriptor import TensorDescriptor


class SwizzlingType(Enum):
    HOPPER_VALUE = 0
    HOPPER_SCALE = 1
    BLACKWELL_SCALE = 2


class StridedLayout:
    data: torch.Tensor
    shape: list[int]
    strides: list[int]

    def __init__(self, tensor) -> None:
        self.data = tensor
        self.shape = list(tensor.shape)
        self.strides = list(tensor.stride())

    def make_tma(self, tensor, block_shape, transpose):
        shape = list(self.shape)
        strides = list(self.strides)
        if transpose:
            block_shape = block_shape[:-2] + [block_shape[-1], block_shape[-2]]
            shape = self.shape[:-2] + [shape[-1], shape[-2]]
            strides = strides[:-2] + [strides[-1], strides[-2]]
        PACK_DIVISOR = 2 if tensor.dtype == torch.uint8 else 1
        indx = strides.index(1)
        block_shape[indx] = block_shape[indx] // PACK_DIVISOR
        # TODO: is this needed?
        # is_microscaled_format = mx_ctx.weight_scale is not None and w.dtype == torch.uint8
        # if is_microscaled_format:
        #     # Pad the inner shape to 128 for mxfp4 weights; TMA requires this when the compiler uses
        #     # CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B.
        #     # This technically makes the shape masking incorrect, but it's fine because:
        #     #  - When the N dim is padded, the scales will be masked to 0.
        #     #  - When the K dim is padded, the activations we perform tl.dot with will be masked to 0.
        #     #    Note: the scales can't be relied on for zeroing in this case, because they apply to groups
        #     #    of 32 elements in the K dimension.
        #     pad = 128
        #     dim_to_pad = -1
        #     old_size = w_desc.shape[dim_to_pad]
        #     padded_size = triton.cdiv(old_size, pad) * pad
        #     if padded_size != old_size:
        #         w_desc.shape = list(w_desc.shape)
        #         w_desc.shape[dim_to_pad] = padded_size

        return TensorDescriptor(base=tensor, shape=shape, strides=strides, block_shape=block_shape)


class BlackwellScaleBlockLayout(StridedLayout):

    def __init__(self, tensor):
        swizzle_axis = 2
        axis = tensor.stride().index(1)
        perm = list(range(tensor.ndim))
        perm[tensor.ndim - 1], perm[axis] = axis, tensor.ndim - 1
        if swizzle_axis is not None:
            perm[tensor.ndim - 2], perm[swizzle_axis] = swizzle_axis, tensor.ndim - 2
        tensor = torch.permute(tensor, perm).contiguous()
        tensor = swizzle_mx_scale_bw(tensor, allow_pad=True)
        tensor = perm_tensor_from_contig(tensor, axis, swizzle_axis)
        layout_shape = (1, tensor.shape[0] * tensor.shape[2] // 128, tensor.shape[1] // 4, 2, 256)
        self.data = tensor
        self.shape = list(layout_shape)
        self.strides = list(make_strides(layout_shape))

    def make_tma(self, tensor, block_shape, transpose):
        assert not transpose
        assert len(block_shape) == 2
        MX_PACK_DIVISOR = 32
        MX_SCALE_BLOCK_K = block_shape[1] // MX_PACK_DIVISOR
        block_shape = [1, block_shape[0] // 128, MX_SCALE_BLOCK_K // 4, 2, 256]
        return TensorDescriptor(base=tensor, shape=self.shape, strides=self.strides, block_shape=block_shape)


def make_strides(shape, order=None):
    n = len(shape)
    if order is None:
        order = list(reversed(range(n)))
    if sorted(order) != list(range(n)):
        raise ValueError("`order` must be a permutation of range(len(shape))")
    strides = [0] * n
    stride = 1
    for dim in order:
        strides[dim] = stride
        stride *= shape[dim]
    return strides


def make_layout(shape, type=StridedLayout, order=None):
    return type(shape=list(shape), strides=make_strides(shape, order))


@dataclass
class Tensor:

    handle: torch.Tensor
    shape: list[int] | None = None
    shape_max: list[int] | None = None
    swizzle_mode: SwizzlingType | None = None
    layout: StridedLayout | None = None

    def __post_init__(self):
        # initialize shape
        if self.shape is None:
            self.shape = self.handle.shape
        # validate shape: all elements must be `int` or numel-1 `torch.Tensor`
        is_int = lambda s: isinstance(s, int)
        is_item = lambda s: hasattr(s, "numel") and s.numel() == 1
        assert all(map(lambda s: is_int(s) or is_item(s), self.shape))
        # initialize shape_max
        if self.shape_max is None:
            self.shape_max = [None] * len(self.shape)
        for i, (s, smax) in enumerate(zip(self.shape, self.shape_max)):
            if smax is not None and not is_int(smax):
                raise ValueError(f"shape_max[{i}] must be `int` or `None`; got {type(smax)}")
            if smax is None:
                self.shape_max[i] = s
        # validate shape_max: all elements must be `int`
        assert all(map(is_int, self.shape_max))
        # initialize layouts
        if self.layout is None:
            self.layout = StridedLayout(self.handle)
        # TODO: should be @properties ?
        self.ndim = self.handle.ndim
        self.dtype = self.handle.dtype
        self.device = self.handle.device
        # TODO: i don't think this should be part of this dataclass
        self.is_fp4 = self.dtype == torch.uint8
        self.element_bitwidth = 4 if self.is_fp4 else self.dtype.itemsize * 8

    # torch compatibility layer

    def stride(self, i=None):
        return self.layout.strides if i is None else self.layout.strides[i]

    def data_ptr(self):
        return self.layout.data.data_ptr()

    def numel(self):
        return self.handle.numel()

    def element_size(self):
        return self.handle.element_size()

    def permute(self, *permutation):
        assert self.swizzle_mode is None
        h = self.handle.permute(*permutation)
        return Tensor(h, swizzle_mode=self.swizzle_mode)

    def view(self, *args):
        assert self.swizzle_mode is None
        h = self.handle.view(*args)
        return Tensor(h, swizzle_mode=self.swizzle_mode)


@dataclass
class Bitmatrix(Tensor):
    """
    Represents a boolean matrix in a packed format where each element occupies
    a single bit of memory.

    _scratchpad is either None or an all-zero array of size >= shape[-1]; we pass it along
    with the actual bitmatrix to avoid having to launch a separate memset
    kernel when we call Bitmatrix::sum().
    """

    scratchpad: torch.Tensor = None

    def __post_init__(self):
        super().__post_init__()
        assert self.handle.shape[-1] * 32 == self.shape[-1]
        assert self.handle.ndim == 2
        assert self.dtype == torch.uint32

    def sum(self, partials_block_size):
        _, n_cols = self.shape
        dev = self.device
        if self.scratchpad is None:
            self.scratchpad = clear_sums(n_cols, dev)
        out_ret = self.scratchpad[:n_cols]
        self.scratchpad = None  # throw error if we try to sum again
        return sum_bitmatrix_rows(self, out_ret, partials_block_size)


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


def swizzle(tensor, swizzle_mode):
    LayoutCls = {
        SwizzlingType.BLACKWELL_SCALE: BlackwellScaleBlockLayout,
        None: StridedLayout,
    }[swizzle_mode]
    # Permute the tensor so that axis is the last dimension and swizzle_axis is the second to last dimension.
    ret_shape = list(tensor.shape)
    if tensor.dtype == torch.uint8:
        ret_shape[1] *= 2
    return Tensor(tensor, shape=ret_shape, layout=LayoutCls(tensor))


def make_tma(tensor, block_shape, transpose=False):
    if not isinstance(tensor, Tensor):
        tensor = Tensor(tensor)
    return tensor.layout.make_tma(tensor, block_shape, transpose=transpose)


def unswizzle(tensor, swizzle_mode):
    if isinstance(tensor, Tensor):
        tensor = tensor.handle
    swizzle_axis = 2
    axis = tensor.stride().index(1)
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
