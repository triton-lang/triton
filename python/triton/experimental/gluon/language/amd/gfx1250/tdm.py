from __future__ import annotations
from typing import List, Tuple, TYPE_CHECKING
from dataclasses import dataclass

import triton.experimental.gluon.language._core as ttgl
from triton.experimental.gluon.language._layouts import PaddedSharedLayout, SwizzledSharedLayout
from triton.experimental.gluon.language.amd.gfx1250 import PartitionedSharedLayout
from triton.experimental.gluon.language._core import builtin, _unwrap_if_constexpr

if TYPE_CHECKING:
    from triton._C import ir
    from triton.experimental.gluon.language._core import shared_memory_descriptor

__all__ = [
    "update_tensor_descriptor", "async_load", "async_wait", "make_tensor_descriptor", "tensor_descriptor",
    "tensor_descriptor_type", "prefetch", "async_scatter"
]


@dataclass(eq=True)
class tensor_descriptor_type(ttgl.base_type):
    """The type for a tensor descriptor."""

    block_type: ttgl.block_type
    shape_type: ttgl.tuple_type
    strides_type: ttgl.tuple_type
    layout: PaddedSharedLayout | SwizzledSharedLayout | PartitionedSharedLayout

    def __str__(self) -> str:
        return f"tensor_descriptor<{self.block_type}, {self.layout}>"

    def _unflatten_ir(self, handles: List[ir.value], cursor: int) -> Tuple[tensor_descriptor, int]:
        handle = handles[cursor]
        cursor += 1
        shape, cursor = self.shape_type._unflatten_ir(handles, cursor)
        strides, cursor = self.strides_type._unflatten_ir(handles, cursor)
        value = tensor_descriptor(handle, shape, strides, self)
        return value, cursor

    def _to_ir(self, builder: ir.builder) -> ir.type:
        is_signed = self.block_type.element_ty.is_int_signed()
        return builder.get_tensor_descriptor_layout_type(
            self.block_type.to_ir(builder),
            is_signed,
            self.layout._to_ir(builder),
        )

    def _flatten_ir_types(self, builder: ir.builder, out: List[ir.type]) -> None:
        out.append(self._to_ir(builder))
        self.shape_type._flatten_ir_types(builder, out)
        self.strides_type._flatten_ir_types(builder, out)

    def mangle(self) -> str:
        return f"TD{self.block_type.mangle()}_{self.shape_type.mangle()}_{self.strides_type.mangle()}_{self.layout.mangle()}TD"


@dataclass
class tensor_descriptor(ttgl.base_value):
    """A descriptor representing a tensor in global memory."""

    handle: ir.value
    shape: ttgl.tuple
    strides: ttgl.tuple
    type: tensor_descriptor_type

    def _set_name(self, builder: ir.builder, name: str) -> None:
        self.handle.set_loc(builder.create_name_loc(name, self.handle.get_loc()))
        self.shape._set_name(builder, name + ".shape")
        self.strides._set_name(builder, name + ".stride")

    def _flatten_ir(self, handles: List[ir.value]) -> None:
        handles.append(self.handle)
        self.shape._flatten_ir(handles)
        self.strides._flatten_ir(handles)

    @property
    def block_type(self):
        return self.type.block_type

    @property
    def block_shape(self):
        return self.type.block_type.shape

    @property
    def dtype(self):
        return self.type.block_type.element_ty

    @property
    def layout(self):
        return self.type.layout


@builtin
def make_tensor_descriptor(base: ttgl.tensor, shape: List[ttgl.constexpr | ttgl.tensor],
                           strides: List[ttgl.constexpr | ttgl.tensor], block_shape: List[ttgl.constexpr],
                           layout: PaddedSharedLayout | SwizzledSharedLayout | PartitionedSharedLayout,
                           _semantic=None) -> tensor_descriptor:
    """Make a tensor descriptor object.

    Args:
        base (tensor): base pointer of the tensor in global memory.
        shape (List[int]): shape of the tensor.
        strides (List[int]): strides of the tensor.
        block_shape (List[int]): block shape of the tensor.
        layout (PaddedSharedLayout | SwizzledSharedLayout | PartitionedSharedLayout): the layout of the tensor in shared memory.

    Returns:
        tensor_descriptor: the created tensor descriptor object
    """
    ndim = len(shape)
    assert 1 <= ndim <= 5, f"Expected 1 <= ndim <= 5 but got {ndim} dimensions"
    assert len(strides) == ndim, f"Expected {ndim} strides but got {len(strides)}"
    block_shape = _unwrap_if_constexpr(block_shape)
    assert len(block_shape) == ndim, f"Expected block_shape to have {ndim} dimensions but got {len(block_shape)}"
    assert isinstance(base.dtype, ttgl.pointer_type), "Expected base to be a pointer"

    layout = _unwrap_if_constexpr(layout)
    assert isinstance(layout, (PaddedSharedLayout, SwizzledSharedLayout, PartitionedSharedLayout)), \
        "Expected layout to be a PaddedSharedLayout, SwizzledSharedLayout, or PartitionedSharedLayout"
    if isinstance(layout, SwizzledSharedLayout):
        assert layout.max_phase == 1, "Expected max_phase to be 1 for SwizzledSharedLayout"
    elif isinstance(layout, PartitionedSharedLayout):
        assert isinstance(layout.partition_layout, (PaddedSharedLayout, SwizzledSharedLayout)), \
            "PartitionedSharedLayout partition_layout must be PaddedSharedLayout or SwizzledSharedLayout"
        if isinstance(layout.partition_layout, SwizzledSharedLayout):
            assert layout.partition_layout.max_phase == 1, \
                "Expected max_phase to be 1 for SwizzledSharedLayout in PartitionedSharedLayout"

    base_handle = base.handle
    shape_handles = _semantic._convert_to_ir_values(shape, require_i64=False)  # i32 shape
    stride_handles = _semantic._convert_to_ir_values(strides, require_i64=True)  # i64 stride

    shape = ttgl.tuple(shape)
    strides = ttgl.tuple(strides)
    block_type = ttgl.block_type(base.type.element_ty, block_shape)
    type = tensor_descriptor_type(block_type, shape.type, strides.type, layout)

    padding = _semantic._str_to_padding_option("zero")
    handle = _semantic.builder.create_make_tensor_descriptor(type._to_ir(_semantic.builder), base_handle, shape_handles,
                                                             stride_handles, padding)

    return tensor_descriptor(handle, shape, strides, type)


def _handle_i32_pred(pred, _semantic):
    pred = _unwrap_if_constexpr(pred)
    if isinstance(pred, bool):
        pred = int(pred)
    pred = _semantic.to_tensor(pred)
    if pred.type.is_int1():
        pred = _semantic.cast(pred, ttgl.int32)
    assert pred.type.is_int32(), f"Expected pred to be an int32 or int1 value, but got {pred.type}"
    return pred


@builtin
def update_tensor_descriptor(desc: tensor_descriptor, add_offsets: List[ttgl.constexpr | ttgl.tensor] = None,
                             set_bounds: List[ttgl.constexpr | ttgl.tensor] = None,
                             dest: shared_memory_descriptor = None, pred=None, barrier: shared_memory_descriptor = None,
                             _semantic=None) -> tensor_descriptor:
    """Update selected fields of a TDM descriptor; return a new descriptor SSA value.

    Each parameter is independently optional; only the fields the caller
    names are written.  Everything else is inherited from the input
    descriptor.

    NOTE: Unlike the standard tensor-descriptor mental model, `add_offsets`
    here moves the tile position only.  It does NOT update the descriptor's
    bounds.  If the loop crosses an OOB boundary, you must also pass
    `set_bounds` explicitly to install the correct OOB extent.

    Args:
        desc (tensor_descriptor): the input descriptor.
        add_offsets (List[int], optional): per-dim deltas in element units
            that move the tile position.  Does not touch the bounds.
        set_bounds (List[int], optional): per-dim absolute rewrite of the
            descriptor's bounds.  Use to install OOB extent at a peel
            epilogue.
        dest (shared_memory_descriptor, optional): set the descriptor's
            shared-memory slot used by subsequent loads/stores.
        pred (int, optional): set the descriptor's predicate.
        barrier (shared_memory_descriptor, optional): enable barrier
            signaling on this descriptor and set the barrier address.

    Returns:
        tensor_descriptor: a new descriptor SSA value with the requested
        fields rewritten.

    Raises:
        ValueError: if no parameter is provided (no-op updates are forbidden).

    Example:
        # K-loop interior: bump tile position only
        d = tdm.update_tensor_descriptor(d, add_offsets=[0, BLOCK_K])

        # Prologue: position at first tile + wire LDS and barrier
        d = tdm.update_tensor_descriptor(d, add_offsets=[pid_m * BLOCK_M, 0],
                                         dest=a_shared, barrier=a_bar)

        # Peel epilogue: install real OOB extent for the partial last tile
        d = tdm.update_tensor_descriptor(d, set_bounds=[M - pid_m * BLOCK_M, K - k_main])
    """
    if add_offsets is None and set_bounds is None and dest is None and pred is None and barrier is None:
        raise ValueError("tdm.update_tensor_descriptor requires at least one of add_offsets, "
                         "set_bounds, dest, pred, barrier")

    rank = len(desc.block_shape)

    add_offset_handles = []
    if add_offsets is not None:
        if len(add_offsets) != rank:
            raise ValueError(f"add_offsets must have length {rank} (descriptor rank), got {len(add_offsets)}")
        add_offset_handles = _semantic._convert_to_ir_values(add_offsets, require_i64=False)

    set_bounds_handles = []
    if set_bounds is not None:
        if len(set_bounds) != rank:
            raise ValueError(f"set_bounds must have length {rank} (descriptor rank), got {len(set_bounds)}")
        set_bounds_handles = _semantic._convert_to_ir_values(set_bounds, require_i64=False)

    dest = _unwrap_if_constexpr(dest)
    dest_handle = dest.handle if dest is not None else ttgl.ir.value()

    pred_handle = ttgl.ir.value()
    if pred is not None:
        pred_t = _semantic.to_tensor(pred)
        pred_handle = pred_t.handle

    barrier = _unwrap_if_constexpr(barrier)
    barrier_handle = barrier.handle if barrier is not None else ttgl.ir.value()

    new_handle = _semantic.builder.create_update_tensor_descriptor(desc.handle, add_offset_handles, set_bounds_handles,
                                                                   dest_handle, pred_handle, barrier_handle)
    # Rebuild shape/strides tuples so the returned descriptor's tuple objects
    # don't alias the original's (the frontend's SSA tracking assumes distinct
    # tuples per descriptor value).
    new_shape = ttgl.tuple(list(desc.shape))
    new_strides = ttgl.tuple(list(desc.strides))
    return tensor_descriptor(new_handle, new_shape, new_strides, desc.type)


@builtin
def async_load(src: tensor_descriptor, offsets: List[ttgl.constexpr | ttgl.tensor], dest: shared_memory_descriptor,
               pred=True, mbarrier: shared_memory_descriptor = None, warp_used_hint=None, cache_modifier="",
               _semantic=None) -> None:
    """Load a block of tensor specified in tensor descriptor from global memory to shared memory asynchronously.

    Args:
        src (tensor_descriptor): the source tensor descriptor.
        offsets (List[int]): the offsets from the base pointer in the tensor descriptor.
        dest (shared_memory_descriptor): the shared memory destination to store the loaded data.
        pred (bool, optional): Predicate to enable or disable the load. Defaults to True.
        mbarrier (shared_memory_descriptor, optional): The barrier object to signal "arrive" on.
        warp_used_hint (int, optional): Bitmask selecting which warps issue
            the TDM copy (bit ``n`` => warp ``n``); cleared warps become HW
            no-ops.  Doesn't affect the data in ``dest``, only the work split.
            The number of active warps must be a power of two, and the active
            warps must follow a regular bit pattern for efficient lowering.
            Examples: ``0b00001111`` (warps 0..3), ``0b11110000`` (warps
            4..7), ``0b01010101`` (warps 0,2,4,6).  Omit / ``None`` = all
            warps participate; explicit ``0`` and other invalid hints are
            rejected by the verifier.
        cache_modifier (str, optional): Cache behavior.
    """
    offset_handles = _semantic._convert_to_ir_values(offsets, require_i64=False)
    pred = _handle_i32_pred(pred, _semantic)
    mbarrier = _unwrap_if_constexpr(mbarrier)
    mbarrier_handle = mbarrier.handle if mbarrier is not None else ttgl.ir.value()
    cache_modifier = _semantic._str_to_load_cache_modifier(cache_modifier)

    warp_used_hint = _unwrap_if_constexpr(warp_used_hint)
    if warp_used_hint is not None:
        warp_used_hint = int(warp_used_hint)

    _semantic.builder.create_async_tdm_copy_global_to_local(src.handle, offset_handles, dest.handle, pred.handle,
                                                            mbarrier_handle, cache_modifier, warp_used_hint)


@builtin
def async_store(dest: tensor_descriptor, offsets: List[ttgl.constexpr | ttgl.tensor], src: shared_memory_descriptor,
                mbarrier: shared_memory_descriptor = None, cache_modifier="", _semantic=None) -> None:
    """Store a block of tensor specified in tensor descriptor from shared memory to global memory asynchronously.

    Args:
        dest (tensor_descriptor): the destination tensor descriptor.
        offsets (List[int]): the offsets from the base pointer in the tensor descriptor.
        src (shared_memory_descriptor): the shared memory source to load the data.
        mbarrier (shared_memory_descriptor, optional): The barrier object to signal "arrive" on.
    """
    offset_handles = _semantic._convert_to_ir_values(offsets, require_i64=False)
    mbarrier = _unwrap_if_constexpr(mbarrier)
    mbarrier_handle = mbarrier.handle if mbarrier is not None else ttgl.ir.value()
    cache_modifier = _semantic._str_to_store_cache_modifier(cache_modifier)
    _semantic.builder.create_async_tdm_copy_local_to_global(dest.handle, offset_handles, src.handle, mbarrier_handle,
                                                            cache_modifier)


@builtin
def async_wait(num_outstanding=0, _semantic=None) -> None:
    """Wait for the outstanding asynchronous tensor operations to complete.

    Args:
        num_outstanding (int): number of outstanding async tensor operations to wait for.
    """
    num_outstanding = _unwrap_if_constexpr(num_outstanding)
    _semantic.builder.create_async_tdm_wait(num_outstanding)


@builtin
def async_scatter(desc: tensor_descriptor, dst_row_indices: ttgl.tensor, dst_col_offset, src: shared_memory_descriptor,
                  mbarrier: shared_memory_descriptor = None, _semantic=None) -> None:
    """Scatter data from shared memory to non-contiguous rows in global memory asynchronously.

    This operation uses TDM scatter mode to write data to non-contiguous rows in global memory.
    Unlike async_store which writes to contiguous rows, scatter allows writing to arbitrary
    rows specified by the dst_row_indices tensor.

    The dtype of dst_row_indices determines the index size:
    - int16: up to 16 rows can be scattered per TDM instruction
    - int32: up to 8 rows can be scattered per TDM instruction
    If more rows are needed, multiple TDM instructions will be automatically issued.

    Args:
        desc (tensor_descriptor): the destination tensor descriptor. Must be 2D.
        dst_row_indices (tensor): 1D tensor of row indices (int16 or int32) in the destination tensor.
        dst_col_offset (int or tensor): the starting column offset in the destination tensor
                                        for all scattered rows.
        src (shared_memory_descriptor): the shared memory source containing data to scatter. Must be 2D.
        mbarrier (shared_memory_descriptor, optional): The barrier object to signal "arrive" on.
    """
    ndim = len(desc.block_shape)
    assert ndim == 2, f"TDM scatter only supports 2D tensors, got {ndim}D"

    src_ndim = len(src.shape)
    assert src_ndim == 2, f"TDM scatter src must be 2D, got {src_ndim}D"

    # Convert dst_col_offset to i32
    dst_col_offset_handle = _semantic._convert_to_ir_values([dst_col_offset], require_i64=False)[0]

    mbarrier = _unwrap_if_constexpr(mbarrier)
    mbarrier_handle = mbarrier.handle if mbarrier is not None else ttgl.ir.value()

    _semantic.builder.create_async_tdm_scatter(desc.handle, dst_row_indices.handle, dst_col_offset_handle, src.handle,
                                               mbarrier_handle)


@builtin
def async_gather(desc: tensor_descriptor, src_row_indices: ttgl.tensor, src_col_offset, dst: shared_memory_descriptor,
                 pred=True, mbarrier: shared_memory_descriptor = None, _semantic=None) -> None:
    """Gather data from non-contiguous rows in global memory to shared memory asynchronously.

    This operation uses TDM gather mode to read data from non-contiguous rows in global memory.
    Unlike async_load which reads from contiguous rows, gather allows reading from arbitrary
    rows specified by the src_row_indices tensor.

    The dtype of src_row_indices determines the index size:
    - int16: up to 16 rows can be gathered per TDM instruction
    - int32: up to 8 rows can be gathered per TDM instruction
    If more rows are needed, multiple TDM instructions will be automatically issued.

    Args:
        desc (tensor_descriptor): the source tensor descriptor. Must be 2D.
        src_row_indices (tensor): 1D tensor of row indices (int16 or int32) in the source tensor.
        src_col_offset (int or tensor): the starting column offset in the source tensor
                                        for all gathered rows.
        dst (shared_memory_descriptor): the shared memory destination to store gathered data. Must be 2D.
        pred (bool, optional): Predicate to enable or disable the gather. Defaults to True.
        mbarrier (shared_memory_descriptor, optional): The barrier object to signal "arrive" on.
    """
    ndim = len(desc.block_shape)
    assert ndim == 2, f"TDM gather only supports 2D tensors, got {ndim}D"

    dst_ndim = len(dst.shape)
    assert dst_ndim == 2, f"TDM gather dst must be 2D, got {dst_ndim}D"

    # Convert src_col_offset to i32
    src_col_offset_handle = _semantic._convert_to_ir_values([src_col_offset], require_i64=False)[0]

    pred = _handle_i32_pred(pred, _semantic)

    mbarrier = _unwrap_if_constexpr(mbarrier)
    mbarrier_handle = mbarrier.handle if mbarrier is not None else ttgl.ir.value()

    _semantic.builder.create_async_tdm_gather(desc.handle, src_row_indices.handle, src_col_offset_handle, dst.handle,
                                              pred.handle, mbarrier_handle)


@builtin
def prefetch(src: tensor_descriptor, offsets: List[ttgl.constexpr | ttgl.tensor], pred: bool = True,
             speculative: bool = False, _semantic=None) -> None:
    """Prefetches a block of tensor specified in tensor descriptor from global memory into L2. Speculative prefetches can generate more
    efficient assembly because they do not require out of bounds checks. However, they are dropped by the hardware if their virtual address translation is not cached.
    So speculative should only be set if previous iterations have accessed the same virtual page (e.g. column major)
    Args:
        src (tensor_descriptor): the source tensor descriptor.
        offsets (List[int]): the offsets from the base pointer in the tensor descriptor.
        pred (bool, optional): Predicate to enable or disable the prefetch. Defaults to True.
        speculative (bool, optional): Whether the prefetch is speculative. Defaults to False.
    """
    offset_handles = _semantic._convert_to_ir_values(offsets, require_i64=False)
    pred = _semantic.to_tensor(pred)
    pred_handle = pred.handle
    speculative = _unwrap_if_constexpr(speculative)
    _semantic.builder.create_tdm_prefetch(src.handle, offset_handles, pred_handle, speculative, False)


@builtin
def _test_prefetch_with_offsets(src: tensor_descriptor, offsets: List[ttgl.constexpr | ttgl.tensor], pred: bool = True,
                                speculative: bool = False, _semantic=None) -> ttgl.tensor:
    """Test-only prefetch variant that returns offsets for validation."""
    offset_handles = _semantic._convert_to_ir_values(offsets, require_i64=False)
    pred = _semantic.to_tensor(pred)
    pred_handle = pred.handle
    speculative = _unwrap_if_constexpr(speculative)
    handle = _semantic.builder.create_tdm_prefetch(src.handle, offset_handles, pred_handle, speculative, True)
    shape = _semantic.builder.get_shape_from_tensor(handle)
    layout = _semantic.builder.get_gluon_layout_from_tensor(handle)
    ret_ty = ttgl.distributed_type(ttgl.int64, shape, layout)
    tensor = ttgl.tensor(handle, ret_ty)
    return tensor
