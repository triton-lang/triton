from __future__ import annotations
from typing import Optional, Tuple, List, TYPE_CHECKING

from dataclasses import dataclass
from triton.experimental.gluon.language import _core as ttgl
from triton.experimental.gluon.language._core import builtin, base_type, base_value, _unwrap_if_constexpr
from triton.experimental.gluon.language._semantic import _check

from . import tma
from ..hopper import mbarrier, fence_async_shared

if TYPE_CHECKING:
    from triton._C.libtriton.gluon_ir import GluonOpBuilder
    from triton._C.libtriton import gluon_ir as ir
    from ..._semantic import GluonSemantic

__all__ = [
    "allocate_tensor_memory",
    "fence_async_shared",
    "mbarrier",
    "tensor_memory_descriptor",
    "TensorMemoryLayout",
    "tma",
]


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

    def mangle(self) -> str:
        block_str = f"{self.block[0]}x{self.block[1]}"
        unpacked_str = "U" if self.unpacked else "P"
        cta_split_str = f"CS{self.cta_split_num[0]}x{self.cta_split_num[1]}" if self.cta_split_num else ""
        return f"TL{block_str}{unpacked_str}{cta_split_str}TL"


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
        shape_str = "_".join([str(s) for s in self.shape])
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

    @property
    def rank(self):
        return len(self.shape)

    @property
    def layout(self):
        return self.type.layout

    def __str__(self) -> str:
        return str(self.type)

    @builtin
    def load(self, layout, _semantic: GluonSemantic) -> ttgl.tensor:
        layout = _unwrap_if_constexpr(layout)
        ret_ty = ttgl.distributed_type(self.dtype, self.shape, layout)
        builder = _semantic.builder
        handle = builder.create_tmem_load(ret_ty.to_ir(builder), self.handle)
        return ttgl.tensor(handle, ret_ty)

    @builtin
    def store(self, value, pred=True, _semantic: GluonSemantic = None) -> None:
        pred = _unwrap_if_constexpr(pred)
        pred = _semantic.to_tensor(pred)
        _semantic.builder.create_tmem_store(self.handle, value.handle, pred.handle)

    @builtin
    def slice(self, start, length, _semantic: GluonSemantic) -> None:
        start = _unwrap_if_constexpr(start)
        length = _unwrap_if_constexpr(length)
        _check(isinstance(start, int), lambda: "start must be a constant int")
        _check(isinstance(length, int), lambda: "length must be a constant int")
        shape = self.shape[:-1] + [length]
        layout = self.type.layout
        layout = TensorMemoryLayout((layout.block[0], min(layout.block[1], length)), layout.unpacked,
                                    layout.cta_split_num)
        ret = tensor_memory_descriptor(None, self.dtype, shape, layout, self.type.alloc_shape)
        builder = _semantic.builder
        ret.handle = builder.create_tmem_subslice(ret.type.to_ir(builder), self.handle, start)
        return ret

    @builtin
    def index(self, index, _semantic: GluonSemantic = None) -> tensor_memory_descriptor:
        index = _semantic.to_tensor(index)
        builder = _semantic.builder
        offsets = [builder.get_int32(0)] * self.rank
        offsets[0] = index.handle
        shape = self.shape[1:]
        layout = self.layout
        ret = tensor_memory_descriptor(None, self.dtype, shape, layout, self.type.alloc_shape)
        ret.handle = builder.create_memdesc_subview(ret.type.to_ir(builder), self.handle, offsets)
        return ret

    @builtin
    def _reinterpret(self, dtype, shape, layout, _semantic: GluonSemantic = None) -> tensor_memory_descriptor:
        dtype = _unwrap_if_constexpr(dtype)
        shape = [_unwrap_if_constexpr(s) for s in shape]
        layout = _unwrap_if_constexpr(layout)

        ty = tensor_memory_descriptor_type(dtype, shape, layout, shape)
        handle = _semantic.builder.create_memdesc_reinterpret(ty.to_ir(_semantic.builder), self.handle)
        return tensor_memory_descriptor(handle, **ty.__dict__)


@builtin
def allocate_tensor_memory(element_ty, shape, layout, value=None, _semantic=None):
    element_ty = _unwrap_if_constexpr(element_ty)
    shape = _unwrap_if_constexpr(shape)
    layout = _unwrap_if_constexpr(layout)
    value = value.handle if value is not None else None

    ty = tensor_memory_descriptor_type(element_ty, shape, layout, shape)
    builder = _semantic.builder
    handle = builder.create_tmem_alloc(ty.to_ir(builder), value)
    return tensor_memory_descriptor(handle, element_ty, shape, layout, shape)


@builtin
def tcgen05_mma(a, b, acc, *, use_acc=True, pred=True, mbarriers=None, mbarrier_preds=None, _semantic=None):
    use_acc = _semantic.to_tensor(use_acc)
    pred = _semantic.to_tensor(pred)

    if mbarriers is None:
        assert mbarrier_preds is None
        mbarriers = []
        mbarrier_preds = []
    else:
        mbarriers = [bar.handle for bar in mbarriers]
        if mbarrier_preds is None:
            true = _semantic.to_tensor(True)
            mbarrier_preds = [true] * len(mbarriers)
        else:
            mbarrier_preds = _semantic._convert_to_ir_values(mbarrier_preds, require_i64=False)

    _semantic.builder.create_tcgen05_mma(a.handle, b.handle, acc.handle, use_acc.handle, pred.handle, mbarriers,
                                         mbarrier_preds)
