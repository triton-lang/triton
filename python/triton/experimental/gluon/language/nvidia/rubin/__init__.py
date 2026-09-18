from __future__ import annotations

from dataclasses import dataclass

from triton.experimental.gluon.language._core import _unwrap_if_constexpr, builtin

from ..blackwell import (
    CachePolicy,
    FractionalEvictionPolicy,
    TensorMemoryLayout,
    _TensorMemoryLinearLayout,
    _packed_arith,
    add2,
    allocate_tensor_memory,
    async_copy,
    async_store,
    clc,
    cluster,
    fence_async_shared,
    fma2,
    max2,
    min2,
    mma_v2,
    mul2,
    sub2,
    tensor_memory_descriptor,
    tensor_memory_descriptor_type,
    tcgen05_commit,
    tcgen05_copy,
    tcgen05_mma,
    tcgen05_mma_barrier_count,
    tcgen05_mma_scaled,
    tma,
)
from ..blackwell import TensorMemoryScalesLayout as _BlackwellTensorMemoryScalesLayout
from . import mbarrier

__all__ = [
    "CachePolicy",
    "FractionalEvictionPolicy",
    "add2",
    "add4",
    "allocate_tensor_memory",
    "async_copy",
    "async_store",
    "clc",
    "cluster",
    "fence_async_shared",
    "fma2",
    "fma4",
    "mbarrier",
    "max2",
    "min2",
    "mma_v2",
    "mul2",
    "mul4",
    "sub2",
    "sub4",
    "tensor_memory_descriptor",
    "tensor_memory_descriptor_type",
    "tcgen05_commit",
    "tcgen05_copy",
    "tcgen05_mma",
    "tcgen05_mma_barrier_count",
    "tcgen05_mma_scaled",
    "TensorMemoryLayout",
    "TensorMemoryScalesLayout",
    "tma",
    "_TensorMemoryLinearLayout",
]


@builtin
def add4(lhs, rhs, dtype=None, _semantic=None):
    """Add four-lane FP8 or FP4 operands with a Rubin packed instruction."""
    return _packed_arith("add", (lhs, rhs), dtype, _semantic)


@builtin
def sub4(lhs, rhs, dtype=None, _semantic=None):
    """Subtract four-lane FP8 or FP4 operands with a Rubin packed instruction."""
    return _packed_arith("sub", (lhs, rhs), dtype, _semantic)


@builtin
def mul4(lhs, rhs, dtype=None, _semantic=None):
    """Multiply four-lane FP8 or FP4 operands with a Rubin packed instruction."""
    return _packed_arith("mul", (lhs, rhs), dtype, _semantic)


@builtin
def fma4(lhs, rhs, acc, dtype=None, _semantic=None):
    """Perform a Rubin four-lane packed FP8 or FP4 fused multiply-add."""
    return _packed_arith("fma", (lhs, rhs, acc), dtype, _semantic)


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
