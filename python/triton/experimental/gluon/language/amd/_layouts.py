from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
from triton.language.core import _unwrap_if_constexpr

from triton.experimental.gluon.language._layouts import _realize_cta_layout, DistributedLayout

__all__ = [
    "AMDMFMALayout",
    "AMDWMMALayout",
]


@dataclass(frozen=True)
class AMDMFMALayout(DistributedLayout):
    """
    Represents a layout for AMD MFMA (matrix core) operations.

    Args:
        version (int): The GPU architecture.
        instr_shape (List[int]): The shape in the form of (M, N, K) of the matrix.
        transposed (bool): Indicates the result tensor is transposed so that each thread holds consecutive elements in the same row instead of column, which is good for chained dot and global write.
        warps_per_cta (List[int]): The warp layout in the block.
        element_bitwidth Optional(int): Bit width of the output element type. Supported values are 32 and 64. Defaults to 32.
        tiles_per_warp Optional(List[int]): The tile layout within a warp. Defaults to unit tile layout, i.e., single tile on all dimensions.
        ctas_per_cga (Optional[List[int]]): CTAs per CGA grouping.
        cta_split_num (Optional[List[int]]): Split factors for CTAs.
        cta_order (Optional[List[int]]): CTA ordering.

    Current supported versions:

    - 1: gfx908
    - 2: gfx90a
    - 3: gfx942
    - 4: gfx950
    """
    version: int
    instr_shape: List[int]
    transposed: bool
    warps_per_cta: List[int]
    element_bitwidth: Optional[int] = None
    tiles_per_warp: Optional[List[int]] = None
    ctas_per_cga: Optional[List[int]] = None
    cta_split_num: Optional[List[int]] = None
    cta_order: Optional[List[int]] = None

    def __post_init__(self):
        super().__setattr__("version", _unwrap_if_constexpr(self.version))
        super().__setattr__("instr_shape", _unwrap_if_constexpr(self.instr_shape))
        super().__setattr__("transposed", _unwrap_if_constexpr(self.transposed))
        super().__setattr__("warps_per_cta", _unwrap_if_constexpr(self.warps_per_cta))
        super().__setattr__("element_bitwidth", _unwrap_if_constexpr(self.element_bitwidth))
        super().__setattr__("tiles_per_warp", _unwrap_if_constexpr(self.tiles_per_warp))
        super().__setattr__("ctas_per_cga", _unwrap_if_constexpr(self.ctas_per_cga))
        super().__setattr__("cta_split_num", _unwrap_if_constexpr(self.cta_split_num))
        super().__setattr__("cta_order", _unwrap_if_constexpr(self.cta_order))

        if self.element_bitwidth is None:
            object.__setattr__(self, "element_bitwidth", 32)
        if self.tiles_per_warp is None:
            object.__setattr__(self, "tiles_per_warp", [1] * len(self.warps_per_cta))

        self.verify()

    def _to_ir(self, builder):
        return builder.get_amd_mfma_layout(self.version, self.warps_per_cta, self.instr_shape, self.transposed,
                                           self.ctas_per_cga, self.cta_split_num, self.cta_order, self.tiles_per_warp,
                                           self.element_bitwidth)

    def mangle(self) -> str:

        def stringify(x):
            if x is None:
                return ""
            return "_".join(map(str, x))

        return f"MFMA_{self.version}_{stringify(self.instr_shape)}_{self.transposed}_{stringify(self.warps_per_cta)}_{self.element_bitwidth}_{stringify(self.tiles_per_warp)}_{stringify(self.ctas_per_cga)}_{stringify(self.cta_split_num)}_{stringify(self.cta_order)}_MFMA"

    def verify(self):
        assert self.version >= 1 and self.version <= 4, "version must be in the [1, 4] range"
        assert len(self.instr_shape) == 3, "instr_shape must follow the (M, N, K) format"
        valid_shapes = [[32, 32], [16, 16], [64, 4], [4, 64]]
        assert self.instr_shape[0:2] in valid_shapes, f"invalid intrinsic shape {self.instr_shape}"
        assert self.element_bitwidth in [32, 64], "element bitwidth must be 32 or 64"

        rank = len(self.warps_per_cta)
        _realize_cta_layout(self, rank)
        assert len(self.ctas_per_cga) == rank
        assert len(self.cta_split_num) == rank
        assert len(self.cta_order) == rank

    def __hash__(self):
        return hash((
            self.version,
            tuple(self.instr_shape),
            self.transposed,
            tuple(self.warps_per_cta),
            self.element_bitwidth if self.element_bitwidth else None,
            tuple(self.tiles_per_warp) if self.tiles_per_warp else None,
            tuple(self.ctas_per_cga) if self.ctas_per_cga else None,
            tuple(self.cta_split_num) if self.cta_split_num else None,
            tuple(self.cta_order) if self.cta_order else None,
        ))


@dataclass(frozen=True)
class AMDWMMALayout(DistributedLayout):
    """
    Represents a layout for AMD WMMA (matrix core) operations.

    Args:
        version (int): Indicates the GPU architecture.
        transposed (bool): Indicates the result tensor is transposed.
        warps_per_cta (List[int]): Number of warps per CTA.
        instr_shape (Optional[List[int]]): Instruction shape (M, N, K). Defaults to (16, 16, 16).
        ctas_per_cga (Optional[List[int]]): CTAs per CGA grouping.
        cta_split_num (Optional[List[int]]): Split factors for CTAs.
        cta_order (Optional[List[int]]): CTA ordering.

    Current supported versions:

    - 1: RDNA3; e.g., gfx1100, gfx1101
    - 2: RDNA4; e.g., gfx1200, gfx1201
    - 3: gfx1250
    """
    version: int
    transposed: bool
    warps_per_cta: List[int]
    instr_shape: Optional[List[int]] = None
    ctas_per_cga: Optional[List[int]] = None
    cta_split_num: Optional[List[int]] = None
    cta_order: Optional[List[int]] = None

    def __post_init__(self):
        super().__setattr__("version", _unwrap_if_constexpr(self.version))
        super().__setattr__("transposed", _unwrap_if_constexpr(self.transposed))
        super().__setattr__("warps_per_cta", _unwrap_if_constexpr(self.warps_per_cta))
        super().__setattr__("ctas_per_cga", _unwrap_if_constexpr(self.ctas_per_cga))
        super().__setattr__("cta_split_num", _unwrap_if_constexpr(self.cta_split_num))
        super().__setattr__("cta_order", _unwrap_if_constexpr(self.cta_order))

        instr_shape = _unwrap_if_constexpr(self.instr_shape) if self.instr_shape is not None else [16, 16, 16]
        super().__setattr__("instr_shape", _unwrap_if_constexpr(instr_shape))
        self.verify()

    def _to_ir(self, builder):
        return builder.get_amd_wmma_layout(self.version, self.transposed, self.warps_per_cta, self.ctas_per_cga,
                                           self.cta_split_num, self.cta_order, self.instr_shape)

    def mangle(self) -> str:

        def stringify(x):
            if x is None:
                return ""
            return "_".join(map(str, x))

        return f"WMMA_{self.version}_{self.transposed}_{stringify(self.warps_per_cta)}_{stringify(self.instr_shape)}_{stringify(self.ctas_per_cga)}_{stringify(self.cta_split_num)}_{stringify(self.cta_order)}_WMMA"

    def verify(self):
        assert self.version >= 1 and self.version <= 3, "version must be in the [1, 3] range"

        rank = len(self.warps_per_cta)
        _realize_cta_layout(self, rank)
        assert len(self.ctas_per_cga) == rank
        assert len(self.cta_split_num) == rank
        assert len(self.cta_order) == rank

    def __hash__(self):
        return hash((
            self.version,
            self.transposed,
            tuple(self.warps_per_cta),
            tuple(self.instr_shape) if self.instr_shape else None,
            tuple(self.ctas_per_cga) if self.ctas_per_cga else None,
            tuple(self.cta_split_num) if self.cta_split_num else None,
            tuple(self.cta_order) if self.cta_order else None,
        ))
