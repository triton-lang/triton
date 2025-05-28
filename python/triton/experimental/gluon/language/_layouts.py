from dataclasses import dataclass
from typing import List, Optional

__all__ = ["BlockedLayout"]


@dataclass(frozen=True)
class BlockedLayout:
    size_per_thread: List[int]
    threads_per_warp: List[int]
    warps_per_cta: List[int]
    order: List[int]
    ctas_per_cga: Optional[List[int]] = None
    cta_split_num: Optional[List[int]] = None
    cta_order: Optional[List[int]] = None

    def __post_init__(self):
        rank = len(self.size_per_thread)
        assert len(self.threads_per_warp) == rank
        assert len(self.warps_per_cta) == rank
        assert len(self.order) == rank
        assert self.ctas_per_cga is None or len(self.ctas_per_cga) == rank
        assert self.cta_split_num is None or len(self.cta_split_num) == rank
        assert self.cta_order is None or len(self.cta_order) == rank

    def _to_ir(self, builder):
        rank = len(self.size_per_thread)
        ctas_per_cga = self.ctas_per_cga or [1] * rank
        cta_split_num = self.cta_split_num or [1] * rank
        cta_order = self.cta_order or list(reversed(range(rank)))
        return builder.get_blocked_layout(
            self.size_per_thread,
            self.threads_per_warp,
            self.warps_per_cta,
            self.order,
            ctas_per_cga,
            cta_split_num,
            cta_order,
        )

    def mangle(self) -> str:

        def stringify(x):
            if x is None:
                return ""
            return "_".join(map(str, x))

        size_per_thread = stringify(self.size_per_thread)
        threads_per_warp = stringify(self.threads_per_warp)
        warps_per_cta = stringify(self.warps_per_cta)
        order = stringify(self.order)
        ctas_per_cga = stringify(self.ctas_per_cga)
        cta_split_num = stringify(self.cta_split_num)
        cta_order = stringify(self.cta_order)
        return f"B{size_per_thread}B{threads_per_warp}B{warps_per_cta}B{order}B{ctas_per_cga}B{cta_split_num}B{cta_order}B"
