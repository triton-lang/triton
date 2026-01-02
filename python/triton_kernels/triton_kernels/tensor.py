from dataclasses import dataclass, fields

import torch
from triton.tools.ragged_tma import create_ragged_descriptor
from triton.tools.tensor_descriptor import TensorDescriptor

from .target_info import cuda_capability_geq
from .tensor_details import bitmatrix as bitmatrix_details
from .tensor_details import ragged_tensor as ragged_tensor_details
from .tensor_details.layout import BlackwellMXValueLayout, Layout, StridedLayout
from .tensor_details.ragged_tensor import RaggedTensorMetadata
from .tensor_details.dtype import IntegerType, FloatType, DataType, FP4, UINT8, FP8_E4M3FN, FP8_E5M2, FP16, BF16, FP32, BIT


# storage
# ---------------------------------------------------------------------------- #
@dataclass
class Storage:
    data: torch.Tensor
    layout: Layout

    @property
    def device(self):
        return self.data.device


# main tensor class
# ---------------------------------------------------------------------------- #


@dataclass
class Tensor:
    storage: Storage
    dtype: IntegerType | FloatType
    shape: list[int] | None = None
    shape_max: list[int] | None = None

    def __post_init__(self):
        assert isinstance(self.storage, Storage)
        # initialize dtype
        if self.dtype.bitwidth < 8 and self.shape is None:
            raise ValueError("shape must be provided for sub-byte types")
        # initialize shape
        if self.shape is None:
            self.shape = list(self.storage.data.shape)
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

    # torch compatibility layer
    @property
    def ndim(self):
        return len(self.shape)

    @property
    def device(self):
        return self.storage.device

    def stride(self, i=None):
        return self.storage.data.stride() if i is None else self.storage.data.stride(i)

    def data_ptr(self):
        return self.storage.data.data_ptr()

    def numel(self):
        return self.storage.data.numel()

    def element_size(self):
        return self.dtype.bitwidth // 8

    @property
    def data(self):
        t = self.storage
        return t.data if isinstance(t, Storage) else t

    def dim(self):
        return self.ndim

    def size(self, i=None):
        if i is None:
            return self.shape
        return self.shape[i]


def is_tma_compliant(tensor):
    storage = tensor.storage
    # TMAs didn't exist until Hopper
    if not cuda_capability_geq(9, 0):
        return False
    # TMAs only exist for 2D, 3D, 5D inputs
    if len(storage.data.shape) not in [2, 3, 5]:
        return False
    # TMAs need at most one stride equal to 1
    # and all other strides divisble by 16
    strides = list(storage.data.stride())
    try:
        major_dim = strides.index(1)
    except ValueError:
        major_dim = -1
    ndim = storage.data.ndim
    bitwidth = 4 if storage.data.dtype == torch.uint8 else storage.data.element_size() * 8
    compliant = [strides[i] * bitwidth % 128 == 0 for i in range(ndim) if i != major_dim]
    return all(compliant)


def make_dense_tma(tensor, block_shape, is_scale):
    storage = tensor.storage
    strides = list(storage.data.stride())
    shape = list(storage.data.shape)
    block_shape = storage.layout.swizzle_block_shape(block_shape)
    transpose = strides[-1] != 1
    if transpose:
        # Need to transpose since tensor descriptor expects strides except for the last dimension 16-byte aligned
        # https://github.com/triton-lang/triton/blob/e5e0081db3335e7755e2c67c784cb1c92769812f/python/triton/tools/tensor_descriptor.py#L26
        block_shape = block_shape[:-2] + [block_shape[-1], block_shape[-2]]
        shape = shape[:-2] + [shape[-1], shape[-2]]
        strides = strides[:-2] + [strides[-1], strides[-2]]
    if storage.data.dtype == torch.uint8 and not is_scale:
        indx = strides.index(1)
        block_shape[indx] = block_shape[indx] // 2
        if isinstance(storage.layout, BlackwellMXValueLayout) and shape[-1] % 128 != 0:
            raise ValueError(
                "inner shape need to be multiple of 128 for mxfp4 (CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B) TMAs.")
    return TensorDescriptor(storage.data, shape, strides, block_shape)


def make_tma(tensor, block_shape, mode, is_scale=False):
    if mode in ["dense", "gather", "scatter"]:
        return make_dense_tma(tensor, block_shape, is_scale)
    assert mode == "ragged"
    storage = tensor.storage
    ragged_dim = len(storage.data.shape) - 2
    return create_ragged_descriptor(storage.data, block_shape, ragged_dim=ragged_dim)


# ---------------------------------------------------------------------------- #
# bitmatrix
# ---------------------------------------------------------------------------- #
@dataclass
class Bitmatrix(Tensor):

    def __post_init__(self):
        assert self.dtype == BIT
        super().__post_init__()


make_bitmatrix_metadata = bitmatrix_details.make_bitmatrix_metadata
make_bitmatrix_metadata_torch = bitmatrix_details.make_bitmatrix_metadata_torch


# ---------------------------------------------------------------------------- #
# ragged tensor
# ---------------------------------------------------------------------------- #
@dataclass
class RaggedTensor:
    """
    A ragged `tensor` is a collection of 2D tensors that share the same number of columns.
    Each tensor in this collection is called a `slice`.
    """

    # slice_sizes[i] is the number of rows in slice `i`
    slice_sizes: torch.Tensor
    # ragged tensors are stored in memory as (potentially padded) 2D tensors of shape
    # [num_total_rows, num_cols]
    # where `num_total_rows` >= sum(slice_sizes)
    data: torch.Tensor
    # `metadata`` contains information about the ragged tensor
    # see `tensor_details/ragged_tensor.py` for more details
    metadata: RaggedTensorMetadata


# construct ragged tensor metadata from `slice_sizes` and `max_n_blocks`
make_ragged_tensor_metadata = ragged_tensor_details.make_ragged_tensor_metadata
make_ragged_tensor_metadata_torch = ragged_tensor_details.make_ragged_tensor_metadata_torch

# remap ragged tensor metadata to a new slice assignment
remap_ragged_tensor_metadata = ragged_tensor_details.remap_ragged_tensor_metadata
remap_ragged_tensor_metadata_torch = ragged_tensor_details.remap_ragged_tensor_metadata_torch

# ---------------------------------------------------------------------------- #
# sparse matrix
# ---------------------------------------------------------------------------- #


@dataclass
class SparseMatrix:
    indx: torch.Tensor
    vals: torch.Tensor
    mask: Bitmatrix

    def __post_init__(self):
        self.mask_metadata = make_bitmatrix_metadata(self.indx, self.mask)


# layout utilities
# ---------------------------------------------------------------------------- #


def wrap_torch_tensor(torch_tensor, dtype=None):
    shape = list(torch_tensor.shape)
    if dtype is None:
        dtype = {
            torch.uint8: UINT8,
            torch.float8_e4m3fn: FP8_E4M3FN,
            torch.float8_e5m2: FP8_E5M2,
            torch.float16: FP16,
            torch.bfloat16: BF16,
        }[torch_tensor.dtype]
    shape[torch_tensor.stride().index(1)] *= (8 * torch_tensor.dtype.itemsize) // dtype.bitwidth
    order = sorted(range(torch_tensor.ndim), key=lambda d: torch_tensor.stride()[d])
    order = [x for x in order]
    return Tensor(Storage(torch_tensor, StridedLayout(order)), dtype=dtype, shape=shape)


def convert_layout(tensor: Tensor, layout: Layout, **layout_transformation_kwargs):
    print("convert from ", tensor.storage.layout, "to", layout)
    # convert `tensor` into canonical form
    transformation = tensor.storage.layout.make_transformation(tensor.shape, tensor.dtype == FP4)
    canonical_data = transformation.unswizzle_data(tensor.storage.data)
    # convert canonical form to `layout`
    transformation = layout.make_transformation(tensor.shape, tensor.dtype == FP4, **layout_transformation_kwargs)
    new_data = transformation.swizzle_data(canonical_data)
    # return new tensor
    attrs = {k.name: getattr(tensor, k.name) for k in fields(tensor) if k.name != "storage"}
    return Tensor(Storage(new_data, layout), **attrs)


def dtype_to_torch_dtype(dtype: DataType) -> torch.dtype:
    if dtype is None:
        return None
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype == FP4:
        return torch.uint8
    elif dtype == FP8_E4M3FN:
        return torch.float8_e4m3fn
    elif dtype == BF16:
        return torch.bfloat16
    elif dtype == FP32:
        return torch.float32
    assert False, f"Unsupported dtype: {dtype}"


def empty(shape: tuple[int], dtype: DataType, device: torch.device, mx_axis: int = 1):
    storage_dtype = torch.uint8 if dtype == FP4 else dtype_to_torch_dtype(dtype)
    storage_shape = (*shape[:1], shape[1] // 2, *shape[2:]) if dtype == FP4 else shape
    storage = torch.empty(storage_shape, device=device, dtype=storage_dtype)
    if dtype == FP4:
        storage = storage.mT.contiguous().mT
    return Tensor(Storage(storage), dtype=FP4 if dtype == FP4 else storage_dtype, shape=shape)
