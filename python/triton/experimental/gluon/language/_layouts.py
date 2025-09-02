from dataclasses import dataclass
from typing import List, Optional
from triton.language.core import _unwrap_if_constexpr, _unwrap_shape, constexpr_type
from triton.runtime.jit import constexpr_function


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

    @property
    def type(self):
        return constexpr_type(self)


@dataclass(frozen=True)
class AutoLayout(DistributedLayout):

    def _to_ir(self, builder):
        return builder.get_auto_layout()

    def mangle(self):
        return "AL"


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

    def __hash__(self):
        return hash((
            tuple(self.size_per_thread),
            tuple(self.threads_per_warp),
            tuple(self.warps_per_cta),
            tuple(self.order),
            tuple(self.ctas_per_cga) if self.ctas_per_cga else None,
            tuple(self.cta_split_num) if self.cta_split_num else None,
            tuple(self.cta_order) if self.cta_order else None,
        ))


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

    def __hash__(self):
        return hash((self.dim, self.parent))


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

    def __hash__(self):
        return hash((
            tuple(map(tuple, self.reg_bases)),
            tuple(map(tuple, self.lane_bases)),
            tuple(map(tuple, self.warp_bases)),
            tuple(map(tuple, self.block_bases)),
            tuple(self.shape),
        ))


@dataclass(frozen=True)
class DotOperandLayout(DistributedLayout):
    """
    Represents a layout for a dot operand.

    Args:
        operand_index (int): 0 for LHS and 1 for RHS of the dot operation.
        parent (DistributedLayout): The parent layout, representing the MMA.
        k_width (int): Number of elements per 32-bits.
    """
    operand_index: int
    parent: DistributedLayout
    k_width: int

    def __post_init__(self):
        super().__setattr__("operand_index", _unwrap_if_constexpr(self.operand_index))
        super().__setattr__("parent", _unwrap_if_constexpr(self.parent))
        super().__setattr__("k_width", _unwrap_if_constexpr(self.k_width))

    def _to_ir(self, builder):
        return builder.get_dot_operand_layout(self.operand_index, self.parent._to_ir(builder), self.k_width)

    def mangle(self) -> str:
        return f"DO{self.operand_index}_{self.parent.mangle()}_{self.k_width}DO"

    def __hash__(self):
        return hash((self.operand_index, self.parent, self.k_width))


@dataclass(frozen=True, eq=True)
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

        rank = len(self.warps_per_cta)
        _realize_cta_layout(self, rank)
        assert len(self.ctas_per_cga) == rank
        assert len(self.cta_split_num) == rank
        assert len(self.cta_order) == rank

    def _to_ir(self, builder):
        return builder.get_mma_layout(self.version, self.warps_per_cta, self.ctas_per_cga, self.cta_split_num,
                                      self.cta_order, self.instr_shape)

    def mangle(self) -> str:
        return f"MMA_{self.version}_{self.warps_per_cta}_{self.instr_shape}_{self.ctas_per_cga}_{self.cta_split_num}_{self.cta_order}_MMA"

    def __hash__(self):
        return hash((tuple(self.version), tuple(self.warps_per_cta),
                     tuple(self.instr_shape), tuple(self.ctas_per_cga) if self.ctas_per_cga else None,
                     tuple(self.cta_split_num) if self.cta_split_num else None,
                     tuple(self.cta_order) if self.cta_order else None))


class SharedLayout:
    """
    Base class for shared memory layouts in Gluon IR.
    """

    @property
    def type(self):
        return constexpr_type(self)


@constexpr_function
def _get_shape_per_cta(shape, cta_split_num):
    shape_per_cta = shape
    if cta_split_num is not None:
        assert len(cta_split_num) == len(shape)
        for dim in range(len(shape_per_cta)):
            shape_per_cta[dim] /= cta_split_num[dim]
    return shape_per_cta


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

    @staticmethod
    @constexpr_function
    def get_default_for(block_shape, dtype, transposed=False, fp4_padded=False, ctas_per_cga=None, cta_split_num=None,
                        cta_order=None):
        """Returns an NVMMASharedLayout with default swizzling for a given shape.

        This picks the largest swizzle pattern compatible with the shape, which
        allows emitting the fewest TMA or MMA messages.
        """
        packing_factor = 2 if fp4_padded else 1
        shape_per_cta = _get_shape_per_cta(block_shape, cta_split_num)
        rank = len(block_shape)
        if transposed:
            shape_per_cta = shape_per_cta[1:] + shape_per_cta[:1]
        contig_dim_size = shape_per_cta[-1] * packing_factor
        contig_dim_bytes = contig_dim_size * dtype.primitive_bitwidth // 8
        if contig_dim_bytes >= 128 and contig_dim_bytes % 128 == 0:
            swizzle_byte_width = 128
        elif contig_dim_bytes >= 64 and contig_dim_bytes % 64 == 0:
            swizzle_byte_width = 64
        elif contig_dim_bytes >= 32 and contig_dim_bytes % 32 == 0:
            swizzle_byte_width = 32
        else:
            swizzle_byte_width = 0

        flatten_outer_dim = 1
        for size in shape_per_cta[:-1]:
            flatten_outer_dim *= size
        if len(block_shape) < 2 or flatten_outer_dim < 8:
            swizzle_byte_width = 0

        return NVMMASharedLayout(
            swizzle_byte_width=swizzle_byte_width,
            element_bitwidth=dtype.primitive_bitwidth,
            rank=rank,
            transposed=transposed,
            fp4_padded=fp4_padded,
            ctas_per_cga=ctas_per_cga,
            cta_split_num=cta_split_num,
            cta_order=cta_order,
        )

    def mangle(self) -> str:
        return f"NVMMA_{self.swizzle_byte_width}_{self.element_bitwidth}_{self.transposed}_{self.fp4_padded}_NVMMA"

    def __hash__(self):
        return hash((self.swizzle_byte_width, self.element_bitwidth, self.rank, self.transposed, self.fp4_padded,
                     tuple(self.ctas_per_cga) if self.ctas_per_cga else None,
                     tuple(self.cta_split_num) if self.cta_split_num else None,
                     tuple(self.cta_order) if self.cta_order else None))


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

    def __hash__(self):
        return hash((self.vec, self.per_phase, self.max_phase,
                     tuple(self.order), tuple(self.ctas_per_cga) if self.ctas_per_cga else None,
                     tuple(self.cta_split_num) if self.cta_split_num else None,
                     tuple(self.cta_order) if self.cta_order else None))


@dataclass(frozen=True, eq=True)
class PaddedSharedLayout(SharedLayout):
    """
    Represents a layout for the access to shared memory. Compared to SwizzledSharedLayout,
    it uses padding to avoid shared memory bank conflicts. After every interval tensor elements,
    the corresponding number of padding elements are inserted.
    If a position corresponds to multiple intervals, the padding amounts are summed.

    In the following example of a tensor,
    `eM` represents original elements in the and `pN` represents padded element.

    Before padding, the shared memory looks like:
    [e0, e1,
     e2, e3,
     e4, e5,
     e6, e7,
     ...]

    After padding with interval-padding list [[2, 1], [4, 2]],
    the shared memory will be
    [e0, e1, p0,
     e2, e3, p1, p2, p3,
     e4, e5, p4,
     e6, e7, p5, p6, p7,
     ...]

    Args:
        interval_padding_pairs (List[int]): List of [interval, padding] pair and both interval and padding must be powers of 2.
        order (List[int]): Order of logical tensor dimensions; fastest-varying first.
        ctas_per_cga (Optional[List[int]]): CTAs per CGA grouping.
        cta_split_num (Optional[List[int]]): Split factors for CTAs.
        cta_order (Optional[List[int]]): CTA ordering.
    """
    interval_padding_pairs: List[List[int]]
    order: List[int]
    ctas_per_cga: Optional[List[int]] = None
    cta_split_num: Optional[List[int]] = None
    cta_order: Optional[List[int]] = None

    def __post_init__(self):
        super().__setattr__("interval_padding_pairs", _unwrap_shape(self.interval_padding_pairs))
        super().__setattr__("order", _unwrap_if_constexpr(self.order))
        super().__setattr__("ctas_per_cga", _unwrap_if_constexpr(self.ctas_per_cga))
        super().__setattr__("cta_split_num", _unwrap_if_constexpr(self.cta_split_num))
        super().__setattr__("cta_order", _unwrap_if_constexpr(self.cta_order))

        self.verify()

    def _to_ir(self, builder):
        intervals, paddings = zip(*self.interval_padding_pairs)
        return builder.get_padded_shared_layout(intervals, paddings, self.order, self.ctas_per_cga, self.cta_split_num,
                                                self.cta_order)

    def mangle(self) -> str:

        def stringify(x):
            if x is None:
                return ""
            return "_".join(map(str, x))

        return f"PaddedShared_{stringify(self.interval_padding_pairs)}_{stringify(self.order)}_{stringify(self.ctas_per_cga)}_{stringify(self.cta_split_num)}_{stringify(self.cta_order)}_PaddedShared"

    def verify(self):
        pairs = self.interval_padding_pairs
        assert len(pairs) > 0, "PaddedSharedLayout interval_padding_pairs must have at least one interval-padding pair"
        assert all(len(pair) == 2 for pair in pairs)
        intervals, paddings = zip(*pairs)

        unique_intervals = list(set(intervals))
        assert len(unique_intervals) == len(intervals)

        is_power_of_2 = lambda n: n > 0 and n & (n - 1) == 0
        assert all(is_power_of_2(n) for n in intervals), "PaddedSharedLayout interval values must all be power of two"
        assert all(is_power_of_2(n) for n in paddings), "PaddedSharedLayout padding values must all be power of two"

        rank = len(self.order)
        assert rank > 0, "PaddedSharedLayout order must not be empty"
        _realize_cta_layout(self, rank)

        assert len(self.ctas_per_cga) == rank
        assert len(self.cta_split_num) == rank
        assert len(self.cta_order) == rank

    def __hash__(self):
        return hash((tuple(map(tuple, self.interval_padding_pairs)),
                     tuple(self.order), tuple(self.ctas_per_cga) if self.ctas_per_cga else None,
                     tuple(self.cta_split_num) if self.cta_split_num else None,
                     tuple(self.cta_order) if self.cta_order else None))


# Python impl of LinearEncodingAttr::basesPerDim
def bases_per_dim(bases, rank, skip_broadcast=True):
    result = [1] * rank

    if not bases:
        return result

    non_zero_idx = None

    for basis in bases:
        # Find the first non-zero index in the current basis
        idx = next((i for i, v in enumerate(basis) if v != 0), None)
        if idx is not None:
            non_zero_idx = idx
            result[idx] *= 2
        elif not skip_broadcast:
            # If no non-zero found and we're not skipping broadcasts, use the last found non-zero index
            assert non_zero_idx is not None
            result[non_zero_idx] *= 2

    return result


def warps_per_cta(layout, shape):
    if isinstance(layout, DistributedLinearLayout):
        return bases_per_dim(layout.warp_bases, len(shape))
    elif isinstance(layout, (SliceLayout, DotOperandLayout)):
        return warps_per_cta(layout.parent, shape)
    else:
        return layout.warps_per_cta
