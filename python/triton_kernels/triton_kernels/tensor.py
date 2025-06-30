import torch
from .tensor_details.memory_layout.hopper_value import swizzle_mxfp4_value_hopper, unswizzle_mxfp4_value_hopper_torch
from .tensor_details.memory_layout.hopper_scale import swizzle_mxfp4_scale_hopper, unswizzle_mxfp4_scale_hopper_torch
from .tensor_details.memory_layout.blackwell_scale import swizzle_mx_scale_bw, unswizzle_mx_scale_bw_torch
from .reduction_details.reduce_bitmatrix import clear_sums, sum_bitmatrix_rows
from dataclasses import dataclass
from triton.tools.tensor_descriptor import TensorDescriptor
from abc import ABC, abstractmethod


class Layout(ABC):

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

    def swizzle_data(self, data):
        return data

    def unswizzle_data(self, data):
        return data

    def swizzle_block_shape(self, block_shape):
        return block_shape


class BlackwellMXScaleLayout(Layout):
    name: str = "BLACKWELL_SCALE"

    def swizzle_data(self, data):
        swizzle_axis = 2
        axis = data.stride().index(1)
        perm = list(range(data.ndim))
        perm[data.ndim - 1], perm[axis] = axis, data.ndim - 1
        if swizzle_axis is not None:
            perm[data.ndim - 2], perm[swizzle_axis] = swizzle_axis, data.ndim - 2
        data = torch.permute(data, perm).contiguous()
        data = swizzle_mx_scale_bw(data, allow_pad=True)
        layout_shape = (1, data.shape[0] * data.shape[1] // 128, data.shape[2] // 4, 2, 256)
        return data.view(layout_shape)

    def unswizzle_data(self, data):
        swizzle_axis = 2
        axis = data.stride().index(1)
        data = perm_tensor_to_contig(data, axis, swizzle_axis)
        data = unswizzle_mx_scale_bw_torch(data)
        ndim = data.ndim
        perm = list(range(ndim))
        perm[ndim - 1], perm[axis] = axis, ndim - 1
        if swizzle_axis is not None:
            perm[ndim - 2], perm[swizzle_axis] = swizzle_axis, ndim - 2
        return torch.permute(data, perm).contiguous()

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
        data = swizzle_mxfp4_value_hopper(data, num_warps=8)
        return perm_tensor_from_contig(data, axis, swizzle_axis)

    def unswizzle_data(self, data):
        swizzle_axis = 2
        axis = data.stride().index(1)
        data = perm_tensor_to_contig(data, axis, swizzle_axis)
        data = unswizzle_mxfp4_value_hopper_torch(data, num_warps=8)
        ndim = data.ndim
        perm = list(range(ndim))
        perm[ndim - 1], perm[axis] = axis, ndim - 1
        if swizzle_axis is not None:
            perm[ndim - 2], perm[swizzle_axis] = swizzle_axis, ndim - 2
        return torch.permute(data, perm).contiguous()

    def swizzle_block_shape(self, block_shape):
        return block_shape


@dataclass
class Storage:
    data: torch.Tensor
    layout: Layout

    def __post_init__(self):
        self.data = self.layout.swizzle_data(self.data)

    def make_tma(self, block_shape, transpose):
        strides = list(self.data.stride())
        shape = list(self.data.shape)
        if transpose:
            block_shape = block_shape[:-2] + [block_shape[-1], block_shape[-2]]
            shape = shape[:-2] + [shape[-1], shape[-2]]
            strides = strides[:-2] + [strides[-1], strides[-2]]
        if self.data.dtype == torch.uint8 and self.layout.name is None:
            # physical block size is half logical block size along packed dimension
            indx = strides.index(1)
            block_shape[indx] = block_shape[indx] // 2
            # Pad the inner shape to 128 for mxfp4 weights; TMA requires this when the compiler uses
            # CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B.
            pad = 128
            shape[-1] = (shape[-1] + pad - 1) // pad * pad
        block_shape = self.layout.swizzle_block_shape(block_shape)
        return TensorDescriptor(base=self.data, shape=shape, strides=strides, block_shape=block_shape)


@dataclass
class IntegerType:
    bitwidth: int


@dataclass
class FloatType:
    bitwidth_exponent: int
    bitwidth_mantissa: int
    is_signed: bool

    def __post_init__(self):
        self.bitwidth = int(self.is_signed) + self.bitwidth_exponent + self.bitwidth_mantissa


BIT = IntegerType(1)
FP4 = FloatType(bitwidth_exponent=2, bitwidth_mantissa=1, is_signed=True)


def bitwidth(type: IntegerType | FloatType | torch.dtype):
    if isinstance(type, torch.dtype):
        return type.itemsize * 8
    return type.bitwidth


@dataclass
class Tensor:

    handle: torch.Tensor
    dtype: IntegerType | FloatType | torch.dtype = None
    shape: list[int] | None = None
    shape_max: list[int] | None = None
    storage: Storage = None

    def __post_init__(self):
        # initialize dtype
        if self.dtype is None:
            self.dtype = self.handle.dtype
        if bitwidth(self.dtype) < 8 and self.shape is None and not isinstance(self.handle, Tensor):
            raise ValueError("shape must be provided for sub-byte types")
        # initialize shape
        if self.shape is None:
            self.shape = list(self.handle.shape)
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
        if self.storage is None:
            handle = self.handle.handle if isinstance(self.handle, Tensor) else self.handle
            self.storage = Storage(handle, DefaultLayout())
        #
        self.ndim = self.handle.ndim
        self.device = self.handle.device

    # torch compatibility layer

    def stride(self, i=None):
        return self.storage.data.stride() if i is None else self.storage.data.stride(i)

    def data_ptr(self):
        return self.storage.data.data_ptr()

    def numel(self):
        return self.handle.numel()

    def element_size(self):
        return self.handle.element_size()

    def permute(self, *permutation):
        assert self.storage.layout.name is None
        h = self.handle.permute(*permutation)
        return Tensor(h)

    def view(self, *args):
        assert self.storage.layout.name is None
        h = self.handle.view(*args)
        return Tensor(h)


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
        assert self.dtype == BIT

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


def expand_shape(packed_shape, packed_dim, pack_width):
    ret = list(packed_shape)
    ret[packed_dim] *= pack_width
    return ret


def swizzle(tensor, layout):
    shape = expand_shape(tensor.shape, tensor.stride().index(1), 2)
    tmp = Tensor(tensor, shape=shape, dtype=FP4)
    storage = Storage(data=tensor, layout=layout)
    return Tensor(tmp, storage=storage)


def make_tma(tensor, block_shape, transpose=False):
    if not isinstance(tensor, Tensor):
        tensor = Tensor(tensor)
    return tensor.storage.make_tma(block_shape, transpose=transpose)
