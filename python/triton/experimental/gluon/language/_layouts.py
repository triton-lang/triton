from dataclasses import dataclass
from typing import List, Optional
from triton.language.core import _unwrap_if_constexpr, _unwrap_shape

__all__ = [
    "BlockedLayout",
    "SliceLayout",
    "DistributedLinearLayout",
    "NVMMADistributedLayout",
    "NVMMASharedLayout",
    "SwizzledSharedLayout",
]


def _realize_cta_layout(layout, rank):
    ctas_per_cga = layout.ctas_per_cga or [1] * rank
    cta_split_num = layout.cta_split_num or [1] * rank
    cta_order = layout.cta_order or list(reversed(range(rank)))
    object.__setattr__(layout, "ctas_per_cga", ctas_per_cga)
    object.__setattr__(layout, "cta_split_num", cta_split_num)
    object.__setattr__(layout, "cta_order", cta_order)


class DistributedLayout:
    """
    Base class for distributed memory layouts in Gluon IR.
    """
    pass


@dataclass(frozen=True)
class BlockedLayout(DistributedLayout):
    """
    Represents a blocked layout, partitioning a tensor across threads, warps, and CTAs.

    Args:
        size_per_thread (List[int]): Number of elements per thread per dimension.
        threads_per_warp (List[int]): Number of threads per warp per dimension.
        warps_per_cta (List[int]): Number of warps per CTA per dimension.
        order (List[int]): The ordering of dimensions for partitioning.
        ctas_per_cga (Optional[List[int]]): CTAs per CGA grouping.
        cta_split_num (Optional[List[int]]): Split factors for CTAs.
        cta_order (Optional[List[int]]): Ordering for CTAs.
    """
    size_per_thread: List[int]
    threads_per_warp: List[int]
    warps_per_cta: List[int]
    order: List[int]
    ctas_per_cga: Optional[List[int]] = None
    cta_split_num: Optional[List[int]] = None
    cta_order: Optional[List[int]] = None

    def __post_init__(self):
        super().__setattr__("size_per_thread", _unwrap_if_constexpr(self.size_per_thread))
        super().__setattr__("threads_per_warp", _unwrap_if_constexpr(self.threads_per_warp))
        super().__setattr__("warps_per_cta", _unwrap_if_constexpr(self.warps_per_cta))
        super().__setattr__("order", _unwrap_if_constexpr(self.order))
        super().__setattr__("ctas_per_cga", _unwrap_if_constexpr(self.ctas_per_cga))
        super().__setattr__("cta_split_num", _unwrap_if_constexpr(self.cta_split_num))
        super().__setattr__("cta_order", _unwrap_if_constexpr(self.cta_order))

        rank = len(self.size_per_thread)
        _realize_cta_layout(self, rank)
        assert len(self.threads_per_warp) == rank
        assert len(self.warps_per_cta) == rank
        assert len(self.order) == rank
        assert len(self.ctas_per_cga) == rank
        assert len(self.cta_split_num) == rank
        assert len(self.cta_order) == rank

    def _to_ir(self, builder):
        return builder.get_blocked_layout(
            self.size_per_thread,
            self.threads_per_warp,
            self.warps_per_cta,
            self.order,
            self.ctas_per_cga,
            self.cta_split_num,
            self.cta_order,
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


@dataclass(frozen=True)
class SliceLayout(DistributedLayout):
    """
    Represents a layout corresponding to slicing a distributed tensor along one dimension.

    Args:
        dim (int): The dimension index to slice.
        parent (DistributedLayout): The parent layout before slicing.
    """
    dim: int
    parent: DistributedLayout

    def __post_init__(self):
        super().__setattr__("dim", _unwrap_if_constexpr(self.dim))
        super().__setattr__("parent", _unwrap_if_constexpr(self.parent))

    def _to_ir(self, builder):
        return builder.get_slice_layout(
            self.dim,
            self.parent._to_ir(builder),
        )

    def mangle(self) -> str:
        return f"SL{self.dim}_{self.parent.mangle()}SL"


@dataclass(frozen=True)
class DistributedLinearLayout(DistributedLayout):
    """
    Represents a linear distributed layout with explicit bases at register, lane, warp, and block levels.
    See: https://arxiv.org/abs/2505.23819 for reference.

    Args:
        reg_bases (List[List[int]]): Bases for register-level distribution.
        lane_bases (List[List[int]]): Bases for lane-level distribution.
        warp_bases (List[List[int]]): Bases for warp-level distribution.
        block_bases (List[List[int]]): Bases for block-level distribution.
        shape (List[int]): The tensor global shape.
    """
    reg_bases: List[List[int]]
    lane_bases: List[List[int]]
    warp_bases: List[List[int]]
    block_bases: List[List[int]]
    shape: List[int]

    def __post_init__(self):
        super().__setattr__("reg_bases", _unwrap_shape(self.reg_bases))
        super().__setattr__("lane_bases", _unwrap_shape(self.lane_bases))
        super().__setattr__("warp_bases", _unwrap_shape(self.warp_bases))
        super().__setattr__("block_bases", _unwrap_shape(self.block_bases))
        super().__setattr__("shape", _unwrap_shape(self.shape))

        rank = len(self.shape)

        for basis in self.reg_bases:
            assert len(basis) == rank
        for basis in self.lane_bases:
            assert len(basis) == rank
        for basis in self.warp_bases:
            assert len(basis) == rank
        for basis in self.block_bases:
            assert len(basis) == rank

    def _to_ir(self, builder):
        return builder.get_distributed_linear_layout(self.reg_bases, self.lane_bases, self.warp_bases, self.block_bases,
                                                     self.shape)

    def mangle(self):
        return f"DLL{self.reg_bases}_{self.lane_bases}_{self.warp_bases}_{self.block_bases}_{self.shape}DLL"


@dataclass(frozen=True)
class NVMMADistributedLayout(DistributedLayout):
    """
    Represents a layout for NVIDIA MMA (tensor core) operations.

    Args:
        version (List[int]): Version identifier for the MMA instruction.
        warps_per_cta (List[int]): Number of warps per CTA.
        instr_shape (List[int]): Instruction shape for MMA.
        ctas_per_cga (Optional[List[int]]): CTAs per CGA grouping.
        cta_split_num (Optional[List[int]]): Split factors for CTAs.
        cta_order (Optional[List[int]]): CTA ordering.
    """
    version: List[int]
    warps_per_cta: List[int]
    instr_shape: List[int]
    ctas_per_cga: Optional[List[int]] = None
    cta_split_num: Optional[List[int]] = None
    cta_order: Optional[List[int]] = None

    def __post_init__(self):
        super().__setattr__("version", _unwrap_if_constexpr(self.version))
        super().__setattr__("warps_per_cta", _unwrap_if_constexpr(self.warps_per_cta))
        super().__setattr__("instr_shape", _unwrap_if_constexpr(self.instr_shape))
        super().__setattr__("ctas_per_cga", _unwrap_if_constexpr(self.ctas_per_cga))
        super().__setattr__("cta_split_num", _unwrap_if_constexpr(self.cta_split_num))
        super().__setattr__("cta_order", _unwrap_if_constexpr(self.cta_order))

        rank = 2
        _realize_cta_layout(self, rank)
        assert len(self.ctas_per_cga) == rank
        assert len(self.cta_split_num) == rank
        assert len(self.cta_order) == rank

    def _to_ir(self, builder):
        return builder.get_mma_layout(self.version, self.warps_per_cta, self.ctas_per_cga, self.cta_split_num,
                                      self.cta_order, self.instr_shape)

    def mangle(self) -> str:
        return f"MMA_{self.version}_{self.warps_per_cta}_{self.instr_shape}_{self.ctas_per_cga}_{self.cta_split_num}_{self.cta_order}_MMA"


class SharedLayout:
    """
    Base class for shared memory layouts in Gluon IR.
    """
    pass


@dataclass(frozen=True)
class NVMMASharedLayout(SharedLayout):
    """
    Represents a layout for shared memory suitable for NVIDIA MMA operations.

    Args:
        swizzle_byte_width (int): Width in bytes for swizzling.
        element_bitwidth (int): Bitwidth of element type.
        rank (int): Rank of the tensor.
        transposed (bool): Whether the layout is transposed.
        fp4_padded (bool): Whether FP4 padding is used.
        ctas_per_cga (Optional[List[int]]): CTAs per CGA grouping.
        cta_split_num (Optional[List[int]]): Split factors for CTAs.
        cta_order (Optional[List[int]]): CTA ordering.
    """
    swizzle_byte_width: int
    element_bitwidth: int
    rank: int
    transposed: bool = False
    fp4_padded: bool = False
    ctas_per_cga: Optional[List[int]] = None
    cta_split_num: Optional[List[int]] = None
    cta_order: Optional[List[int]] = None

    def __post_init__(self):
        super().__setattr__("swizzle_byte_width", _unwrap_if_constexpr(self.swizzle_byte_width))
        super().__setattr__("element_bitwidth", _unwrap_if_constexpr(self.element_bitwidth))
        super().__setattr__("rank", _unwrap_if_constexpr(self.rank))
        super().__setattr__("transposed", _unwrap_if_constexpr(self.transposed))
        super().__setattr__("fp4_padded", _unwrap_if_constexpr(self.fp4_padded))
        super().__setattr__("ctas_per_cga", _unwrap_if_constexpr(self.ctas_per_cga))
        super().__setattr__("cta_split_num", _unwrap_if_constexpr(self.cta_split_num))
        super().__setattr__("cta_order", _unwrap_if_constexpr(self.cta_order))

        assert self.element_bitwidth in [8, 16, 32, 64]
        assert self.swizzle_byte_width in [0, 32, 64, 128]
        rank = self.rank
        _realize_cta_layout(self, rank)
        assert len(self.ctas_per_cga) == rank
        assert len(self.cta_split_num) == rank
        assert len(self.cta_order) == rank

    def _to_ir(self, builder):
        return builder.get_nvmma_shared_layout(
            self.swizzle_byte_width,
            self.element_bitwidth,
            self.transposed,
            self.fp4_padded,
            self.ctas_per_cga,
            self.cta_split_num,
            self.cta_order,
        )

    def mangle(self) -> str:
        return f"NVMMA_{self.swizzle_byte_width}_{self.element_bitwidth}_{self.transposed}_{self.fp4_padded}_NVMMA"


@dataclass(frozen=True, eq=True)
class SwizzledSharedLayout(SharedLayout):
    """
    Represents a generic swizzled shared memory layout.

    Args:
        vec (int): Vector width for swizzling.
        per_phase (int): Elements per swizzle phase.
        max_phase (int): Maximum number of swizzle phases.
        order (List[int]): Dimension ordering for swizzling.
        ctas_per_cga (Optional[List[int]]): CTAs per CGA grouping.
        cta_split_num (Optional[List[int]]): Split factors for CTAs.
        cta_order (Optional[List[int]]): CTA ordering.
    """
    vec: int
    per_phase: int
    max_phase: int
    order: List[int]
    ctas_per_cga: Optional[List[int]] = None
    cta_split_num: Optional[List[int]] = None
    cta_order: Optional[List[int]] = None

    def __post_init__(self):
        super().__setattr__("vec", _unwrap_if_constexpr(self.vec))
        super().__setattr__("per_phase", _unwrap_if_constexpr(self.per_phase))
        super().__setattr__("max_phase", _unwrap_if_constexpr(self.max_phase))
        super().__setattr__("order", _unwrap_if_constexpr(self.order))
        super().__setattr__("ctas_per_cga", _unwrap_if_constexpr(self.ctas_per_cga))
        super().__setattr__("cta_split_num", _unwrap_if_constexpr(self.cta_split_num))
        super().__setattr__("cta_order", _unwrap_if_constexpr(self.cta_order))

        rank = len(self.order)
        _realize_cta_layout(self, rank)
        assert len(self.ctas_per_cga) == rank
        assert len(self.cta_split_num) == rank
        assert len(self.cta_order) == rank

    def _to_ir(self, builder):
        return builder.get_swizzled_shared_layout(
            self.vec,
            self.per_phase,
            self.max_phase,
            self.order,
            self.ctas_per_cga,
            self.cta_split_num,
            self.cta_order,
        )

    def mangle(self) -> str:

        def stringify(x):
            if x is None:
                return ""
            return "_".join(map(str, x))

        return f"SSS_{self.vec}_{self.per_phase}_{self.max_phase}_{stringify(self.order)}_{stringify(self.ctas_per_cga)}_{stringify(self.cta_split_num)}_{stringify(self.cta_order)}_SSS"
