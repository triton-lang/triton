from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from triton.experimental.gluon.language._core import _unwrap_if_constexpr, builtin

from ..blackwell import (
    TensorMemoryLayout,
    _TensorMemoryLayoutBase,
    _TensorMemoryLinearLayout,
    allocate_tensor_memory,
    async_copy,
    async_store,
    clc,
    fence_async_shared,
    float2,
    mma_v2,
    tensor_memory_descriptor,
    tensor_memory_descriptor_type,
    tcgen05_commit,
    tcgen05_copy,
    tcgen05_mma_barrier_count,
    tma,
)
from ..blackwell import TensorMemoryScalesLayout as _BlackwellTensorMemoryScalesLayout
from . import mbarrier

__all__ = [
    "allocate_tensor_memory",
    "async_copy",
    "async_store",
    "clc",
    "fence_async_shared",
    "float2",
    "mbarrier",
    "mma_v2",
    "tensor_memory_descriptor",
    "tensor_memory_descriptor_type",
    "tcgen05_commit",
    "tcgen05_copy",
    "tcgen05_mma",
    "tcgen05_mma_barrier_count",
    "tcgen05_mma_scaled",
    "TensorMemoryLayout",
    "TensorMemoryLUTLayout",
    "TensorMemoryScalesLayout",
    "tma",
    "_TensorMemoryLinearLayout",
]


@dataclass(frozen=True, eq=True)
class TensorMemoryLUTLayout(_TensorMemoryLayoutBase):
    """Describes the Tensor Memory layout for Rubin LUT decompression data."""
    cga_layout: List[List[int]] = field(default_factory=list)

    def __post_init__(self):
        super().__setattr__("cga_layout", _unwrap_if_constexpr(self.cga_layout))
        assert all(len(basis) == 2 for basis in self.cga_layout)

    def _to_ir(self, builder):
        return builder.get_tensor_memory_lut_layout([list(basis) for basis in self.cga_layout])

    def mangle(self) -> str:
        cga_layout_str = "_".join("~".join(map(str, basis)) for basis in self.cga_layout)
        return f"TLLUT{cga_layout_str}TLLUT"

    def __hash__(self):
        return hash(tuple(tuple(b) for b in self.cga_layout))


@builtin
def tcgen05_mma(a, b, acc, *, use_acc=True, pred=True, multicast=False, mbarriers=None, mbarrier_preds=None, lut=None,
                _semantic=None):
    """
    Emit a Rubin 5th generation TensorCore MMA instruction.
    acc = a * b + (acc if use_acc else 0)

    Args:
        a (shared_memory_descriptor or tensor_memory_descriptor): Left hand side operand in shared or tensor memory.
        b (shared_memory_descriptor): Right hand side operand in shared memory.
        acc (tensor_memory_descriptor): Accumulator value in tensor memory (mutated).
        use_acc (bool): Whether to use the initial value of the accumulator. Defaults to True.
        pred (bool): Scalar predicate. Operation is skipped if predicate is False. Defaults to True.
        multicast (bool): Whether tcgen05 commit should multicast across a CTA cluster. Defaults to False.
        mbarriers (Sequence[shared_memory_descriptor], optional): Barriers to signal when the operation is complete.
        mbarrier_preds (Sequence[bool], optional): Predicates for barriers. Defaults to None.
        lut (tensor_memory_descriptor, optional): Lookup table used to decompress B. Defaults to None.
    """
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
            mbarrier_preds = [true.handle] * len(mbarriers)
        else:
            mbarrier_preds = _semantic._convert_to_ir_values(mbarrier_preds, require_i64=False)

    multicast = _unwrap_if_constexpr(multicast)
    lut_handle = lut.handle if lut is not None else None
    _semantic.builder.create_tcgen05_mma(a.handle, b.handle, acc.handle, use_acc.handle, pred.handle, mbarriers,
                                         mbarrier_preds, acc.layout.two_ctas, multicast, lut_handle)


@builtin
def tcgen05_mma_scaled(a, b, acc, a_scale, b_scale, a_type, b_type, *, use_acc=True, pred=True, multicast=False,
                       mbarriers=None, mbarrier_preds=None, lut=None, _semantic=None):
    """Emit a Rubin scaled TensorCore MMA instruction with an optional LUT used to decompress B."""
    use_acc = _semantic.to_tensor(use_acc)
    pred = _semantic.to_tensor(pred)
    assert acc.type.layout.block[0] != 64, "tcgen05_mma_scaled does not support blockM=64"

    if mbarriers is None:
        assert mbarrier_preds is None
        mbarriers = []
        mbarrier_preds = []
    else:
        mbarriers = [bar.handle for bar in mbarriers]
        if mbarrier_preds is None:
            true = _semantic.to_tensor(True)
            mbarrier_preds = [true.handle] * len(mbarriers)
        else:
            mbarrier_preds = _semantic._convert_to_ir_values(mbarrier_preds, require_i64=False)

    allowed_formats = {"e2m1", "e4m3", "e5m2"}
    assert a_type.value in allowed_formats, f"Unsupported lhs_format: {a_type.value}"
    assert b_type.value in allowed_formats, f"Unsupported rhs_format: {b_type.value}"
    a_type = _semantic._str_to_fp_type(a_type.value)
    b_type = _semantic._str_to_fp_type(b_type.value)
    multicast = _unwrap_if_constexpr(multicast)
    lut_handle = lut.handle if lut is not None else None
    _semantic.builder.create_tcgen05_mma_scaled(a.handle, b.handle, acc.handle, a_scale.handle, b_scale.handle, a_type,
                                                b_type, use_acc.handle, pred.handle, mbarriers, mbarrier_preds,
                                                acc.layout.two_ctas, multicast, lut_handle)


@dataclass(frozen=True, eq=True)
class TensorMemoryScalesLayout(_BlackwellTensorMemoryScalesLayout):
    """
    Describes the layout for tensor memory scales in Rubin architecture.

    Args:
        cga_layout (Optional[List[List[int]]]): CGA layout bases. Defaults to [].
        block_rep_order (str): Order of repeated scale blocks. Must be either
            ``"mnThenK"`` or ``"kThenMn"``. Defaults to ``"mnThenK"``.
    """
    block_rep_order: str = "mnThenK"

    def __post_init__(self):
        super().__post_init__()
        super().__setattr__("block_rep_order", _unwrap_if_constexpr(self.block_rep_order))
        assert self.block_rep_order in ("mnThenK", "kThenMn")

    def _to_ir(self, builder):
        return builder.get_tensor_memory_scales_layout([list(basis) for basis in self.cga_layout], self.block_rep_order)

    def mangle(self) -> str:
        cga_layout_str = "_".join("~".join(map(str, basis)) for basis in self.cga_layout)
        return f"TLS{self.block_rep_order}_{cga_layout_str}TLS"

    def __hash__(self):
        return hash((tuple(tuple(b) for b in self.cga_layout), self.block_rep_order))
