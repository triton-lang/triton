from __future__ import annotations
from typing import Optional, Tuple, List, TYPE_CHECKING

from dataclasses import dataclass
from triton.experimental.gluon.language import _core as ttgl
from triton.experimental.gluon.language._core import builtin, base_type, base_value, _unwrap_if_constexpr

if TYPE_CHECKING:
    from triton._C.libtriton.gluon_ir import GluonOpBuilder
    from triton._C.libtriton import gluon_ir as ir

__all__ = ["TensorMemoryLayout", "tensor_memory_descriptor", "allocate_tensor_memory"]


@dataclass(frozen=True, eq=True)
class TensorMemoryLayout:
    block: Tuple[int, int]
    unpacked: bool
    cta_split_num: Optional[Tuple[int, int]] = None

    def __post_init__(self):
        assert len(self.block) == 2
        assert self.cta_split_num is None or len(self.cta_split_num) == 2

    def _to_ir(self, builder):
        cta_split_num = self.cta_split_num or [1, 1]
        return builder.get_tensor_memory_layout(
            self.block,
            self.unpacked,
            cta_split_num,
        )


class tensor_memory_descriptor_type(base_type):

    def __init__(self, element_ty, shape, layout, alloc_shape):
        self.element_ty = element_ty
        self.shape = shape
        self.layout = layout
        self.alloc_shape = alloc_shape
        assert isinstance(layout, TensorMemoryLayout)

    def to_ir(self, builder: GluonOpBuilder) -> None:
        return builder.get_tensor_mem_desc_ty(
            self.element_ty.to_ir(builder),
            self.shape,
            self.layout._to_ir(builder),
            self.alloc_shape,
        )

    def _unflatten_ir(self, handles: List[ir.Value], cursor: int) -> Tuple[tensor_memory_descriptor, int]:
        value = tensor_memory_descriptor(handles[cursor], self.element_ty, self.shape, self.layout, self.alloc_shape)
        return value, cursor + 1

    def _flatten_ir_types(self, builder: GluonOpBuilder, out: List[ir.type]) -> None:
        out.append(self.to_ir(builder))

    def __str__(self) -> str:
        return f"tensor_memory_descriptor<{self.element_ty}, {self.shape}, {self.layout}>"

    def __eq__(self, other) -> bool:
        return (type(self) is type(other) and self.shape == other.shape and self.layout == other.layout
                and self.alloc_shape == other.alloc_shape)

    def __neq__(self, other) -> bool:
        return not (self == other)

    def mangle(self) -> str:
        shape_str = "_".join(self.shape)
        return f"MD{self.element_ty.mangle()}S{shape_str}SL{self.layout.mangle()}LAS{self.alloc_shape}ASMD"


class tensor_memory_descriptor(base_value):

    def __init__(self, handle, element_ty, shape, layout, alloc_shape):
        self.handle = handle
        self.type = tensor_memory_descriptor_type(element_ty, shape, layout, alloc_shape)

    def _flatten_ir(self, handles: List[ir.value]) -> None:
        handles.append(self.handle)

    @property
    def dtype(self):
        return self.type.element_ty

    @property
    def shape(self):
        return self.type.shape

    def __str__(self) -> str:
        return str(self.type)

    @builtin
    def load(self, layout, _builder: GluonOpBuilder) -> ttgl.tensor:
        layout = _unwrap_if_constexpr(layout)
        ret_ty = ttgl.distributed_type(self.dtype, self.shape, layout)
        handle = _builder.create_tmem_load(ret_ty.to_ir(_builder), self.handle)
        return ttgl.tensor(handle, ret_ty)

    @builtin
    def store(self, value, pred=True, _builder: GluonOpBuilder = None) -> None:
        pred = _unwrap_if_constexpr(pred)
        pred = ttgl.to_tensor(pred, _builder=_builder)
        _builder.create_tmem_store(self.handle, value.handle, pred.handle)

    @builtin
    def subslice(self, start, length, _builder: GluonOpBuilder) -> None:
        start = _unwrap_if_constexpr(start)
        length = _unwrap_if_constexpr(length)
        assert isinstance(start, int)
        assert isinstance(length, int)
        shape = [self.shape[0], length]
        ret = tensor_memory_descriptor(None, self.dtype, shape, self.type.layout, self.type.alloc_shape)
        ret.handle = _builder.create_tmem_subslice(ret.type.to_ir(_builder), self.handle, start)
        return ret


@builtin
def allocate_tensor_memory(element_ty, shape, layout, value=None, _builder=None):
    element_ty = _unwrap_if_constexpr(element_ty)
    shape = _unwrap_if_constexpr(shape)
    layout = _unwrap_if_constexpr(layout)

    ty = tensor_memory_descriptor_type(element_ty, shape, layout, shape)
    handle = _builder.create_tmem_alloc(ty.to_ir(_builder), value.handle)
    return tensor_memory_descriptor(handle, element_ty, shape, layout, shape)
