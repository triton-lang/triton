from __future__ import annotations

from dataclasses import dataclass

from triton.experimental.gluon.language._core import _unwrap_if_constexpr

from ..blackwell import (
    TensorMemoryLayout,
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
    tcgen05_mma,
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
    "TensorMemoryScalesLayout",
    "tma",
    "_TensorMemoryLinearLayout",
]


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
