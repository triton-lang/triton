from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from triton.experimental.gluon.language._core import _unwrap_if_constexpr, builtin

from ..blackwell import (
    TensorMemoryLayout,
    _tcgen05_mma,
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
    tcgen05_mma_scaled,
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
    return _tcgen05_mma(
        a,
        b,
        acc,
        use_acc=use_acc,
        pred=pred,
        multicast=multicast,
        mbarriers=mbarriers,
        mbarrier_preds=mbarrier_preds,
        lut=lut,
        _semantic=_semantic,
    )


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
