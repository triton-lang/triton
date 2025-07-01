import torch
from .reduction_details.reduce_bitmatrix import clear_sums, sum_bitmatrix_rows
from dataclasses import dataclass
from triton.tools.tensor_descriptor import TensorDescriptor
from .tensor_details.layout import Layout, DefaultLayout


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


def swizzle(tensor, layout):
    shape = list(tensor.shape)
    shape[tensor.stride().index(1)] *= 2
    tmp = Tensor(tensor, shape=shape, dtype=FP4)
    storage = Storage(data=tensor, layout=layout)
    return Tensor(tmp, storage=storage)


def make_tma(tensor, block_shape, transpose=False):
    if not isinstance(tensor, Tensor):
        tensor = Tensor(tensor)
    return tensor.storage.make_tma(block_shape, transpose=transpose)
