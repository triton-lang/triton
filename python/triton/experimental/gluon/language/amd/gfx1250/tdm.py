from __future__ import annotations
from typing import List, Tuple, TYPE_CHECKING
from dataclasses import dataclass

import triton.experimental.gluon.language._core as ttgl
from triton.experimental.gluon.language._layouts import PaddedSharedLayout, SwizzledSharedLayout
from triton.experimental.gluon.language._core import builtin, _unwrap_if_constexpr

if TYPE_CHECKING:
    from triton._C import ir
    from triton.experimental.gluon.language._core import shared_memory_descriptor

__all__ = ["async_load", "async_wait", "make_tensor_descriptor", "tensor_descriptor", "tensor_descriptor_type"]


@dataclass(eq=True)
class tensor_descriptor_type(ttgl.base_type):
    """The type for a tensor descriptor."""

    block_type: ttgl.block_type
    shape_type: ttgl.tuple_type
    strides_type: ttgl.tuple_type
    layout: PaddedSharedLayout | SwizzledSharedLayout

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
                           layout: PaddedSharedLayout | SwizzledSharedLayout, _semantic=None) -> tensor_descriptor:
    """Make a tensor descriptor object.

    Args:
        base (tensor): base pointer of the tensor in global memory.
        shape (List[int]): shape of the tensor.
        strides (List[int]): strides of the tensor.
        block_shape (List[int]): block shape of the tensor.
        layout (PaddedSharedLayout | SwizzledSharedLayout): the layout of the tensor in shared memory.

    Returns:
        tensor_descriptor: the created tensor descriptor object
    """
    ndim = len(shape)
    assert 1 <= ndim <= 5, f"Expected 1 <= ndim <= 5 but got {ndim} dimensions"
    assert len(strides) == ndim, f"Expected {ndim} strides but got {len(strides)}"
    assert len(block_shape) == ndim, f"Expected block_shape to have {ndim} dimensions but got {len(strides)}"
    assert isinstance(base.dtype, ttgl.pointer_type), "Expected base to be a pointer"

    layout = _unwrap_if_constexpr(layout)
    assert isinstance(layout, (PaddedSharedLayout, SwizzledSharedLayout)), \
        "Expected layout to be a PaddedSharedLayout or SwizzledSharedLayout"
    if isinstance(layout, SwizzledSharedLayout):
        assert layout.max_phase == 1, "Expected max_phase to be 1 for SwizzledSharedLayout"

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


@builtin
def async_load(src: tensor_descriptor, offsets: List[ttgl.constexpr | ttgl.tensor], dest: shared_memory_descriptor,
               pred: bool = True, mbarrier: shared_memory_descriptor = None, _semantic=None) -> None:
    """Load a block of tensor specified in tensor descriptor from global memory to shared memory asynchronously.

    Args:
        src (tensor_descriptor): the source tensor descriptor.
        offsets (List[int]): the offsets from the base pointer in the tensor descriptor.
        dest (shared_memory_descriptor): the shared memory destination to store the loaded data.
        pred (bool, optional): Predicate to enable or disable the load. Defaults to True.
        mbarrier (shared_memory_descriptor, optional): The barrier object to signal "arrive" on.
    """
    offset_handles = _semantic._convert_to_ir_values(offsets, require_i64=False)
    pred = _semantic.to_tensor(pred)
    pred_handle = pred.handle
    mbarrier = _unwrap_if_constexpr(mbarrier)
    mbarrier_handle = mbarrier.handle if mbarrier is not None else ttgl.ir.value()
    _semantic.builder.create_async_tdm_copy_global_to_local(src.handle, offset_handles, dest.handle, pred_handle,
                                                            mbarrier_handle)


@builtin
def async_store(dest: tensor_descriptor, offsets: List[ttgl.constexpr | ttgl.tensor], src: shared_memory_descriptor,
                mbarrier: shared_memory_descriptor = None, _semantic=None) -> None:
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
    _semantic.builder.create_async_tdm_copy_local_to_global(dest.handle, offset_handles, src.handle, mbarrier_handle)


@builtin
def async_wait(num_outstanding=0, _semantic=None) -> None:
    """Wait for the outstanding asynchronous tensor operations to complete.

    Args:
        num_outstanding (int): number of outstanding async tensor operations to wait for.
    """
    num_outstanding = _unwrap_if_constexpr(num_outstanding)
    _semantic.builder.create_async_tdm_wait(num_outstanding)
