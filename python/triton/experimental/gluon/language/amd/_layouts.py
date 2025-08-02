from __future__ import annotations

from dataclasses import dataclass
from typing import List
from triton.language.core import _unwrap_if_constexpr

from triton.experimental.gluon.language._layouts import _realize_cta_layout, DistributedLayout
from triton.experimental.gluon import language as ttgl

__all__ = ["AMDMFMALayout", "PaddedSharedLayout"]


def _stringify(x):
    if x is None:
        return ""
    return "_".join(map(str, x))


@dataclass(frozen=True)
class AMDMFMALayout(DistributedLayout):
    """
    Represents a layout for AMD MFMA (matrix core) operations.

    Args:
        version (int): Major and minor identifier for the MFMA instruction.
        instr_shape: (M, N) dimension for the instrinsic shape.
        transposed: indicates the result tensor is transposed so that each thread holds consecutive elements in the same row instead of column, which is good for chained dot and global write.
        warps_per_cta (List[int]): Number of warps per CTA.
        tiles_per_warp: (List[int]): Number of tiles per WARP.
        elem_type: fp32 or fp64
        ctas_per_cga (Optional[List[int]]): CTAs per CGA grouping.
        cta_split_num (Optional[List[int]]): Split factors for CTAs.
        cta_order (Optional[List[int]]): CTA ordering.
    """
    version: int
    instr_shape: List[int]
    transposed: bool
    warps_per_cta: List[int]
    tiles_per_warp: List[int]
    elem_type: ttgl.dtype = ttgl.float32
    ctas_per_cga: List[int] | None = None
    cta_split_num: List[int] | None = None
    cta_order: List[int] | None = None

    def __post_init__(self):
        super().__setattr__("version", _unwrap_if_constexpr(self.version))
        super().__setattr__("instr_shape", _unwrap_if_constexpr(self.instr_shape))
        super().__setattr__("transposed", _unwrap_if_constexpr(self.transposed))
        super().__setattr__("warps_per_cta", _unwrap_if_constexpr(self.warps_per_cta))
        super().__setattr__("tiles_per_warp", _unwrap_if_constexpr(self.tiles_per_warp))
        super().__setattr__("elem_type", _unwrap_if_constexpr(self.elem_type))
        super().__setattr__("ctas_per_cga", _unwrap_if_constexpr(self.ctas_per_cga))
        super().__setattr__("cta_split_num", _unwrap_if_constexpr(self.cta_split_num))
        super().__setattr__("cta_order", _unwrap_if_constexpr(self.cta_order))

        assert self.elem_type.is_fp32() or self.elem_type.is_fp64(
        ), "The element type in AMDMFMALayout should be float32 or float64 type"

        rank = len(self.cta_order)
        _realize_cta_layout(self, rank)
        assert len(self.ctas_per_cga) == rank
        assert len(self.cta_split_num) == rank
        assert len(self.cta_order) == rank

    def _to_ir(self, builder):
        type = builder.get_float_ty() if self.elem_type is ttgl.float32 else builder.get_double_ty()
        return builder.get_amd_mfma_layout(self.version, self.tiles_per_warp, self.warps_per_cta, self.ctas_per_cga,
                                           self.cta_split_num, self.cta_order, self.instr_shape, self.transposed, type)

    def mangle(self) -> str:

        return f"MFMA_{self.version}_{_stringify(self.instr_shape)}_{self.transposed}_{_stringify(self.warps_per_cta)}_{_stringify(self.tiles_per_warp)}_{self.elem_type}_{_stringify(self.ctas_per_cga)}_{_stringify(self.cta_split_num)}_{_stringify(self.cta_order)}_MFMA"


@dataclass(frozen=True)
class PaddedSharedLayout(DistributedLayout):
    """
    Represents a layout for the access to AMD shared memory. Compared to SwizzledSharedLayout,
    it uses padding to avoid shared memory bank conflicts.
    One concrete example, using `eM` to mean tensor elements and `pN` to mean padding and interval-padding pairs are [2:+1, 4:+2]

    #ttg.padded_shared<[2:+1, 4:+2]>
    [e0, e1, p0,
     e2, e3, p1, p2, p3,
     e4, e5, p4,
     e6, e7, p5, p6, p7,
     ...]

    Args:
        intervals: List[int], list of intervals in the interval-padding pairs, e.g [2, 4] in the exmaple above.
        paddings: List[int], list of paddings in the interval-padding pairs, e.g [1, 2] in the exmaple above.
        order: List[int], order of logical tensor dimensions; fastest-varying first.
        ctas_per_cga (Optional[List[int]]): CTAs per CGA grouping.
        cta_split_num (Optional[List[int]]): Split factors for CTAs.
        cta_order (Optional[List[int]]): CTA ordering.
    """
    intervals: List[int]
    paddings: List[int]
    order: List[int]
    ctas_per_cga: List[int] | None = None
    cta_split_num: List[int] | None = None
    cta_order: List[int] | None = None

    def __post_init__(self):
        super().__setattr__("intervals", _unwrap_if_constexpr(self.intervals))
        super().__setattr__("paddings", _unwrap_if_constexpr(self.paddings))
        super().__setattr__("order", _unwrap_if_constexpr(self.order))
        super().__setattr__("ctas_per_cga", _unwrap_if_constexpr(self.ctas_per_cga))
        super().__setattr__("cta_split_num", _unwrap_if_constexpr(self.cta_split_num))
        super().__setattr__("cta_order", _unwrap_if_constexpr(self.cta_order))

        self.verify()

    def _to_ir(self, builder):
        return builder.get_padded_shared_layout(self.intervals, self.paddings, self.order, self.ctas_per_cga,
                                                self.cta_split_num, self.cta_order)

    def mangle(self) -> str:
        return f"PaddesShared_{_stringify(self.intervals)}_{_stringify(self.paddings)}_{_stringify(self.order)}_{_stringify(self.ctas_per_cga)}_{_stringify(self.cta_split_num)}_{_stringify(self.cta_order)}_PaddedShared"

    def verify(self):
        assert len(self.intervals) > 0, "must have at least one interval-padding pair"
        assert len(self.intervals) == len(self.paddings), "intervals size must match paddings size"
        unique_intervals = list(set(self.intervals))
        assert len(unique_intervals) == len(self.intervals)

        is_power_of_2 = lambda n: n > 0 and n & (n - 1) == 0
        assert all(is_power_of_2(n) for n in self.intervals), "interval values must all be power of two"
        assert all(is_power_of_2(n) for n in self.paddings), "padding values must all be power of two"

        assert len(self.order) > 0, "order must not be empty"
        rank = len(self.cta_order)
        assert len(self.order) == rank, "order size must match must match CTALayout rank"
        _realize_cta_layout(self, rank)
        assert len(self.ctas_per_cga) == rank
        assert len(self.cta_split_num) == rank
        assert len(self.cta_order) == rank
