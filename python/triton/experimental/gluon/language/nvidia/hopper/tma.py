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
