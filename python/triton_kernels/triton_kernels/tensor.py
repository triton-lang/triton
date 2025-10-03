from dataclasses import dataclass, fields
from typing import Type

import torch
from triton.tools.tensor_descriptor import TensorDescriptor
from triton.tools.ragged_tma import create_ragged_descriptor

from .reduction_details.reduce_bitmatrix import clear_sums, sum_bitmatrix_rows
from .target_info import cuda_capability_geq
from .tensor_details.layout import Layout, StridedLayout
from .target_info import is_hip
from .tensor_details._bitmatrix_metadata import _bitmatrix_metadata_compute, _bitmatrix_metadata_memset
from .tensor_details._ragged_tensor_metadata import _ragged_tensor_metadata_compute, _ragged_tensor_metadata_memset


@dataclass
class Storage:
    data: torch.Tensor
    layout: Layout = None

    def __post_init__(self):
        assert isinstance(self.data, torch.Tensor)
        if self.layout is None:
            self.layout = StridedLayout(self.data.shape)

    @property
    def device(self):
        return self.data.device

    def is_tma_compliant(self):
        # TMAs didn't exist until Hopper
        if not cuda_capability_geq(9, 0):
            return False
        # TMAs only exist for 2D, 3D, 5D inputs
        if len(self.data.shape) not in [2, 3, 5]:
            return False
        # TMAs need at most one stride equal to 1
        # and all other strides divisble by 16
        strides = list(self.data.stride())
        try:
            major_dim = strides.index(1)
        except ValueError:
            major_dim = -1
        ndim = self.data.ndim
        bitwidth = 4 if self.data.dtype == torch.uint8 else self.data.element_size() * 8
        compliant = [strides[i] * bitwidth % 128 == 0 for i in range(ndim) if i != major_dim]
        return all(compliant)

    def make_dense_tma(self, block_shape, transpose=False):
        strides = list(self.data.stride())
        shape = list(self.data.shape)
        transpose = self.data.stride()[-1] != 1
        if transpose:
            block_shape = block_shape[:-2] + [block_shape[-1], block_shape[-2]]
            shape = shape[:-2] + [shape[-1], shape[-2]]
            strides = strides[:-2] + [strides[-1], strides[-2]]
        if self.data.dtype == torch.uint8 and self.layout.name == "BLACKWELL_VALUE":
            indx = strides.index(1)
            block_shape[indx] = block_shape[indx] // 2
            if shape[-1] % 128 != 0:
                raise ValueError("inner shape need to be multiple of 128 for "
                                 "mxfp4 (CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B) TMAs.")
        block_shape = self.layout.swizzle_block_shape(block_shape)
        return TensorDescriptor(self.data, shape, strides, block_shape)

    def make_tma(self, block_shape, mode, transpose=False):
        if mode in ["dense", "gather", "scatter"]:
            return self.make_dense_tma(block_shape, transpose)
        assert mode == "ragged"
        ragged_dim = len(self.data.shape) - 2
        return create_ragged_descriptor(self.data, block_shape, ragged_dim=ragged_dim)


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
    storage: Storage | torch.Tensor
    dtype: IntegerType | FloatType | torch.dtype = None
    shape: list[int] | None = None
    shape_max: list[int] | None = None

    def __post_init__(self):
        # set storage
        if isinstance(self.storage, torch.Tensor):
            self.storage = Storage(self.storage)
        # initialize dtype
        if self.dtype is None:
            self.dtype = self.storage.data.dtype
        if bitwidth(self.dtype) < 8 and self.shape is None:
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
        return bitwidth(self.dtype) // 8

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

    def __init__(self, storage, shape, shape_max=None, scratchpad=None):
        super().__init__(storage, dtype=BIT, shape=shape, shape_max=shape_max)
        self.scratchpad = scratchpad

    def sum(self, partials_block_size):
        _, n_cols = self.shape
        dev = self.device
        if self.scratchpad is None:
            self.scratchpad = clear_sums(n_cols, dev)
        out_ret = self.scratchpad[:n_cols]
        self.scratchpad = None  # throw error if we try to sum again
        return sum_bitmatrix_rows(self, out_ret, partials_block_size)


# ---------------------------------------------------------------------------- #
#  Metadata
# ---------------------------------------------------------------------------- #

# ragged tensor metadata


@dataclass
class RaggedTensorMetadata:
    """
    Example:
    `batch_sizes`= [15 17 0 127]
    `batch_offs`= [0 15 32 32 332]
    `block_offs_data` = {
        16: [0 1 3 3 11]
        32: [0 1 2 2 6]
        64: [0 1 2 2 4]
        128: [0 1 2 2 3]
    }
    `block_schedule_data` = {
        16:  [(0, 0), (0, 1), (0, 3), (1, 3), (2, 3), ..., (7, 3), -1, ..., -1]
        32:  [(0, 0), (0, 1), (0, 3), (1, 3), (2, 3), (3, 3), -1, ...,      -1]
        64:  [(0, 0), (0, 1), (0, 3), (1, 3), (2, 3), -1, ...,              -1]
        128: [(0, 0), (0, 1), (0, 3), (1, 3), -1, ...,                      -1]
    }
    """
    # batch_sizes[i] is the number of tokens in batch i
    batch_sizes: torch.Tensor
    # batch_offs = [0] + cumsum(batch_sizes)
    # i.e., batch_offs[i] is the offset of the first token for
    # batch `i` in a `batch_sizes`-shaped ragged tensor
    batch_offs: torch.Tensor
    # block_offs_data[k] = [0] + cumsum(ceil_div(batch_sizes, 16 * k))
    # i.e., `block_offs_data[k][i]` is the offset of the first block of
    # `16*k`` token for batch `i` in a `bath_sizes`-shaped ragged tensor
    block_offs_data: torch.Tensor
    # let `num_blocks[k] = block_offs_data[k, 1:] - block_offs_data[k, :-1]
    # block_schedule_data[k] = cat(*[[(batch, blk) for blk in range(blks)] for batch, blks in enumerate(num_blocks)])
    # i.e., if the schedule of batch `i` is [(i, 0), (i, 1), ..., (i, num_blocks[k][i] - 1)]
    # then `block_schedule_data[k]` is the concatenation of the schedules for all batches
    # NOTE 1: `block_schedule_data[k][j]` is a packed 32-bit integer
    # NOTE 2: because the size of `block_schedule_data[k]` is data-dependent, we pad it with -1s
    # up to an user-provided upper bound
    block_schedule_data: torch.Tensor

    def __post_init__(self):
        assert self.block_offs_data.shape[0] == len(RaggedTensorMetadata.block_sizes())
        assert self.block_schedule_data.shape[0] == len(RaggedTensorMetadata.block_sizes())
        assert self.block_offs_data.dtype == torch.int32
        assert self.block_schedule_data.dtype == torch.int32
        if self.batch_sizes is not None:
            assert self.batch_sizes.dtype == torch.int32
        if self.batch_offs is not None:
            assert self.batch_offs.dtype == torch.int32

    def block_offs(self, block_size):
        return self.block_offs_data[RaggedTensorMetadata.block_sizes().index(block_size)]

    def block_schedule(self, block_size):
        return self.block_schedule_data[RaggedTensorMetadata.block_sizes().index(block_size)]

    @staticmethod
    def block_sizes_log2():
        return range(4, 9) if is_hip() else range(4, 8)

    @staticmethod
    def block_sizes():
        return [2**x for x in RaggedTensorMetadata.block_sizes_log2()]


def exact_div(x, y):
    assert x % y == 0
    return x // y


def empty_aligned(shape, dtype, device, pad_size):
    cdiv = lambda x, y: (x + y - 1) // y
    pad = lambda x: cdiv(x, pad_size) * pad_size
    ret = torch.empty((*shape[:-1], pad(shape[-1])), dtype=dtype, device=device)
    ret_slices = (*[slice(None)] * (len(shape) - 1), slice(0, shape[-1]))
    return ret[ret_slices]


def max_n_tiles(n_expts_tot, n_gates):
    if n_gates <= n_expts_tot:
        return n_gates
    return n_expts_tot - 1 - ((n_expts_tot - n_gates - 1) // RaggedTensorMetadata.block_sizes()[0])


def make_ragged_tensor_metadata(expt_hist, n_expts_tot, n_gates):

    if expt_hist is None:
        return RaggedTensorMetadata(None, None, None, None)

    block_sizes_log2 = RaggedTensorMetadata.block_sizes_log2()
    block_size_num = len(block_sizes_log2)
    MEMSET_BLOCK = 512
    dtype = torch.int32
    device = expt_hist.device
    batch_offs_combined = empty_aligned((block_size_num + 1, n_expts_tot + 1), dtype, device, MEMSET_BLOCK)
    block_schedule_data = empty_aligned((block_size_num, max_n_tiles(n_expts_tot, n_gates)), dtype, device,
                                        MEMSET_BLOCK)
    batch_offs, block_offs_data = batch_offs_combined[0], batch_offs_combined[1:]
    n_memset_blocks = exact_div(block_schedule_data.storage().size(), MEMSET_BLOCK)

    _ragged_tensor_metadata_memset[(batch_offs_combined.shape[0] + n_memset_blocks, )](
        expt_hist, n_expts_tot,  #
        batch_offs_combined, batch_offs_combined.stride(0),  #
        block_schedule_data,  #
        block_sizes_log2[0], SIZES=len(block_sizes_log2), BLOCK=MEMSET_BLOCK,  # optimization parameters
        num_warps=4)

    _ragged_tensor_metadata_compute[(block_size_num * n_expts_tot, )](
        expt_hist, block_offs_data, block_offs_data.stride(0), block_schedule_data,
        block_schedule_data.stride(0),  # outputs
        block_sizes_log2[0], SIZES=len(block_sizes_log2), BLOCK=512,  # optimization parameters
        num_warps=4)

    return RaggedTensorMetadata(expt_hist, batch_offs, block_offs_data, block_schedule_data)


# bitmatrix metadata
@dataclass
class BitmatrixMetadata:
    """
    Example:
    `bitmatrix` = [0 0 1 0 1 1 0
                   0 1 0 0 0 1 0
                   1 1 1 0 0 0 1
                   0 0 1 0 1 0 0]
    `col_sum` = [1 2 3 0 2 2 1]
    `row_sorted_indx` = cat([3 6 8], [1 9], [0 2 4 10], [5 7])
    `col_sorted_indx` = cat([5], [3 6], [0 7], [], [9 1 10], [2 4], [8])
    """
    # the number of entries equal to 1 in each column
    col_sum: torch.Tensor
    # indices of nonzero values numbered col-major, grouped by rows, concatenated
    row_sorted_indx: torch.Tensor
    # indices of nonzero values numbered row-major, grouped by cols, concatenated
    col_sorted_indx: torch.Tensor


def make_bitmatrix_metadata(expt_indx, bitmatrix):
    HIST_BLOCK_M = 32
    cdiv = lambda x, y: (x + y - 1) // y
    device = bitmatrix.device
    col_sum, col_partial_sum = bitmatrix.sum(partials_block_size=HIST_BLOCK_M)
    assert col_sum.dtype == torch.int32
    # allocate memory
    n_indx = expt_indx.numel()
    n_cols = bitmatrix.shape[1]
    col_offs = torch.empty(n_cols, dtype=torch.int32, device=device)
    combined_indx = torch.empty(n_indx * 2, dtype=torch.int32, device=device)
    col_sorted_indx = combined_indx[:n_indx]
    row_sorted_indx = combined_indx[n_indx:]
    # memset the output
    MEMSET_BLOCK = 1024
    INDX_OFFS_BLOCK_M = 512
    memset_grid = (cdiv(n_indx * 2, MEMSET_BLOCK) + n_cols + 1, )
    _bitmatrix_metadata_memset[memset_grid](
        combined_indx, n_indx * 2, -1, MEMSET_BLOCK, col_sum,  #
        col_offs, col_sum.shape[0], n_cols, col_partial_sum,  # inputs
        col_partial_sum.shape[0], col_partial_sum.stride(0), col_partial_sum.stride(1),  # outputs
        BLOCK_N=512, BLOCK_M=INDX_OFFS_BLOCK_M,  # tunable parameters
    )
    # compute the output
    compute_grid = (cdiv(bitmatrix.shape_max[0], HIST_BLOCK_M), )
    _bitmatrix_metadata_compute[compute_grid](
        col_sorted_indx, row_sorted_indx,  # outputs
        expt_indx, col_partial_sum, col_partial_sum.stride(0), col_partial_sum.stride(1),  # inputs
        col_offs, bitmatrix.shape[0],  # input shape
        HIST_BLOCK_M, expt_indx.shape[-1],  # constants
    )

    return BitmatrixMetadata(col_sum, col_sorted_indx, row_sorted_indx)


# ---------------------------------------------------------------------------- #


def get_layout(tensor: torch.Tensor | Tensor | None):
    if tensor is None:
        return None
    if isinstance(tensor, Tensor):
        return tensor.storage.layout
    return StridedLayout


def wrap_torch_tensor(torch_tensor, dtype=None):
    if dtype is None:
        dtype = torch_tensor.dtype
    shape = list(torch_tensor.shape)
    shape[torch_tensor.stride().index(1)] *= bitwidth(torch_tensor.dtype) // bitwidth(dtype)
    return Tensor(Storage(torch_tensor), dtype=dtype, shape=shape)


def convert_layout(tensor: Tensor, layout_cls: Type[Layout], **layout_kwargs):
    assert isinstance(tensor, Tensor)
    old_storage = tensor.storage
    old_data = old_storage.layout.unswizzle_data(old_storage.data)
    new_layout = layout_cls(old_data.shape, **layout_kwargs)
    new_data = new_layout.swizzle_data(old_data)
    attrs = {k.name: getattr(tensor, k.name) for k in fields(tensor) if k.name != "storage"}
    return Tensor(Storage(new_data, new_layout), **attrs)
