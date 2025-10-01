from __future__ import annotations
from typing import List, Tuple, TYPE_CHECKING
from dataclasses import dataclass

import triton.experimental.gluon.language._core as ttgl
from triton.experimental.gluon.language._layouts import PaddedSharedLayout
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
    layout: PaddedSharedLayout

    def __str__(self) -> str:
        return f"tensor_descriptor<{self.block_type}, {self.layout}>"

    def _unflatten_ir(self, handles: List[ir.value], cursor: int) -> Tuple[tensor_descriptor, int]:
        handle = handles[cursor]
        cursor += 1
        shape, cursor = self.shape_type._unflatten_ir(handles, cursor)
        strides, cursor = self.strides_type._unflatten_ir(handles, cursor)
        value = tensor_descriptor(handle, shape, strides, self.block_type, self.layout)
        return value, cursor

    def _flatten_ir_types(self, builder: ir.builder, out: List[ir.type]) -> None:
        is_signed = self.block_type.element_ty.is_int_signed()
        ty = builder.get_tensor_descriptor_layout_type(
            self.block_type.to_ir(builder),
            is_signed,
            self.layout._to_ir(builder),
        )
        out.append(ty)
        self.shape_type._flatten_ir_types(builder, out)
        self.strides_type._flatten_ir_types(builder, out)

    def mangle(self) -> str:
        return f"TD{self.block_type.mangle()}_{self.shape_type.mangle()}_{self.strides_type.mangle()}_{self.layout.mangle()}TD"


class tensor_descriptor(ttgl.base_value):
    """A descriptor representing a tensor in global memory."""

    def __init__(self, handle: ir.value, shape: List[ttgl.tensor], strides: List[ttgl.tensor],
                 block_type: ttgl.distributed_type, layout: PaddedSharedLayout):
        self.handle = handle
        self.shape = ttgl.tuple(shape)
        self.strides = ttgl.tuple(strides)
        self.type = tensor_descriptor_type(
            block_type,
            shape_type=self.shape.type,
            strides_type=self.strides.type,
            layout=layout,
        )

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
                           layout: PaddedSharedLayout, _semantic=None) -> tensor_descriptor:
    """Make a tensor descriptor object.

    Args:
        base (tensor): base pointer of the tensor in global memory.
        shape (List[int]): shape of the tensor.
        strides (List[int]): strides of the tensor.
        block_shape (List[int]): block shape of the tensor.
        layout (PaddedSharedLayout): the layout of the tensor in shared memory.

    Returns:
        tensor_descriptor: the created tensor descriptor object
    """
    ndim = len(shape)
    # TODO: support 1D-5D tensor descriptors
    assert ndim == 2, f"Expected 2 dimensions but got {ndim} dimensions"
    assert len(strides) == ndim, f"Expected {ndim} strides but got {len(strides)}"
    assert len(block_shape) == ndim, f"Expected block_shape to have {ndim} dimensions but got {len(strides)}"
    assert isinstance(base.dtype, ttgl.pointer_type), "Expected base to be a pointer"

    layout = _unwrap_if_constexpr(layout)
    assert isinstance(layout, PaddedSharedLayout), "Expected layout to be a PaddedSharedLayout"

    # convert tensor into scalar
    base_handle = base.handle
    shape_handles = _semantic._convert_to_ir_values(shape, require_i64=False)
    stride_handles = _semantic._convert_to_ir_values(strides,
                                                     require_i64=True)  # strides must be i64 for make_tensor_descriptor

    # fill default arguments for create_make_tensor_descriptor
    is_signed_int = base.type.element_ty.is_int_signed()
    padding = _semantic._str_to_padding_option("zero")

    handle = _semantic.builder.create_make_tensor_descriptor(base_handle, shape_handles, stride_handles, block_shape,
                                                             is_signed_int, padding)
    type = ttgl.block_type(base.type.element_ty, block_shape)

    return tensor_descriptor(handle, shape, strides, type, layout)


@builtin
def async_load(src: tensor_descriptor, offsets: List[ttgl.constexpr | ttgl.tensor], dest: shared_memory_descriptor,
               _semantic=None) -> None:
    """Load a block of tensor specified in tensor descriptor from global memory to shared memory asynchronously.

    Args:
        src (tensor_descriptor): the source tensor descriptor.
        offsets (List[int]): the offsets from the base pointer in the tensor descriptor.
        dest (shared_memory_descriptor): the shared memory destination to store the loaded data.
    """
    offset_handles = _semantic._convert_to_ir_values(offsets, require_i64=False)
    _semantic.builder.create_async_tdm_copy_global_to_local(src.handle, offset_handles, dest.handle)


@builtin
def async_wait(num_outstanding=0, _semantic=None) -> None:
    """Wait for the outstanding asynchronous tensor operations to complete.

    Args:
        num_outstanding (int): number of outstanding async tensor operations to wait for.
    """
    num_outstanding = _unwrap_if_constexpr(num_outstanding)
    _semantic.builder.create_async_tdm_wait(num_outstanding)
