from __future__ import annotations
from typing import List, Tuple, TYPE_CHECKING
from dataclasses import dataclass
from triton.language.core import base_type, base_value
import triton.experimental.gluon.language._core as ttgl
from triton.experimental.gluon.language._layouts import NVMMASharedLayout
from triton.experimental.gluon.language._core import builtin, _unwrap_if_constexpr

if TYPE_CHECKING:
    from triton._C import ir

__all__ = ["async_copy_global_to_shared", "async_copy_shared_to_global", "store_wait"]


@dataclass(eq=True)
class tensor_descriptor_type(base_type):
    block_type: ttgl.block_type
    shape_type: ttgl.tuple_type
    strides_type: ttgl.tuple_type
    layout: NVMMASharedLayout

    def __str__(self) -> str:
        return f"tensor_descriptor<{self.block_type}, {self.layout}>"

    def _to_ir(self, builder: ir.builder) -> ir.type:
        is_signed = self.block_type.element_ty.is_int_signed()
        return builder.get_tensor_descriptor_layout_type(
            self.block_type.to_ir(builder),
            is_signed,
            self.layout._to_ir(builder),
        )

    def _unflatten_ir(self, handles: List[ir.value], cursor: int) -> Tuple[tensor_descriptor, int]:
        handle = handles[cursor]
        cursor += 1
        shape, cursor = self.shape_type._unflatten_ir(handles, cursor)
        strides, cursor = self.strides_type._unflatten_ir(handles, cursor)
        value = tensor_descriptor(handle, shape, strides, self.block_type, layout=self.layout)
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
        return f"TD{self.block_type.mangle()}_{self.layout.mangle()}TD"


class tensor_descriptor(base_value):

    def __init__(self, handle, shape: List[ttgl.tensor], strides: List[ttgl.tensor], block_type: ttgl.block_type,
                 layout: NVMMASharedLayout):
        self.handle = handle
        self.shape = ttgl.tuple(shape)
        self.strides = ttgl.tuple(strides)
        self.type = tensor_descriptor_type(block_type, shape_type=self.shape.type, strides_type=self.strides.type,
                                           layout=layout)

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
def async_copy_global_to_shared(tensor_desc, coord, barrier, result, pred=True, _semantic=None):
    coord = _semantic._convert_to_ir_values(coord, require_i64=False)
    pred = _semantic.to_tensor(pred)
    _semantic.builder.create_async_tma_copy_global_to_local(tensor_desc.handle, coord, barrier.handle, result.handle,
                                                            pred.handle)


@builtin
def async_copy_shared_to_global(tensor_desc, coord, src, _semantic=None):
    coord = _semantic._convert_to_ir_values(coord, require_i64=False)
    _semantic.builder.create_async_tma_copy_local_to_global(tensor_desc.handle, coord, src.handle)


@builtin
def store_wait(pendings, _semantic=None):
    pendings = _unwrap_if_constexpr(pendings)
    _semantic.builder.create_async_tma_store_wait(pendings)


@builtin
def make_tensor_descriptor(
    base: ttgl.tensor,
    shape: List[ttgl.tensor],
    strides: List[ttgl.tensor],
    block_shape: List[ttgl.constexpr],
    layout: NVMMASharedLayout,
    padding_option="zero",
    _semantic=None,
) -> tensor_descriptor:
    padding_option = _unwrap_if_constexpr(padding_option)
    block_shape = _unwrap_if_constexpr(block_shape)

    ndim = len(shape)
    if not (1 <= ndim <= 5):
        raise ValueError(f"Expected 1 <= ndim <= 5 but got {ndim} dimensions")
    if len(strides) != ndim:
        raise ValueError(f"Expected {ndim} strides but got {len(strides)}")
    if len(block_shape) != ndim:
        raise ValueError(f"Expected block_shape to have {ndim} dimensions but got {len(block_shape)}")
    assert isinstance(base.dtype, ttgl.pointer_type)
    elem_size = base.dtype.element_ty.primitive_bitwidth // 8
    contig_dim_size = ttgl._unwrap_if_constexpr(block_shape[-1])
    if contig_dim_size * elem_size < 16:
        raise ValueError(
            f"Descriptor block shape must have at least 16 bytes in the last dimension, but got {contig_dim_size} * {elem_size} = {contig_dim_size * elem_size} bytes"
        )

    last_stride = ttgl._unwrap_if_constexpr(strides[-1])
    if last_stride != 1:
        raise ValueError(f"Tensor descriptor last dim must be 1 but got {last_stride}")

    shape = [_semantic.make_scalar(x, ttgl.int32) for x in shape]
    strides = [_semantic.make_scalar(ttgl._unwrap_if_constexpr(x), ttgl.int64) for x in strides]

    # Check whether `block_shape` is static
    block_shape = ttgl._unwrap_shape(block_shape)

    assert isinstance(base.type, ttgl.pointer_type)
    block_type = ttgl.block_type(base.type.element_ty, block_shape)
    base_handle = base.handle

    padding = _semantic._str_to_padding_option(padding_option)

    layout = _unwrap_if_constexpr(layout)
    assert isinstance(layout, NVMMASharedLayout), \
        "Expected layout to be a NVMMASharedLayout"

    shape_type = ttgl.tuple(shape).type
    strides_type = ttgl.tuple(strides).type
    ty = tensor_descriptor_type(block_type, shape_type, strides_type, layout)

    if base.type.element_ty.is_int() and padding == ttgl.ir.PADDING_OPTION.PAD_NAN:
        raise ValueError("Padding option `nan` is not supported for integer blocks")
    handle = _semantic.builder.create_make_tensor_descriptor(
        ty._to_ir(_semantic.builder),
        base_handle,
        [s.handle for s in shape],
        [s.handle for s in strides],
        padding,
    )
    return tensor_descriptor(handle, shape, strides, block_type, layout)
