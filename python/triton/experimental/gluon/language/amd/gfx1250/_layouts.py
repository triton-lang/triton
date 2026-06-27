from __future__ import annotations

from dataclasses import dataclass
from triton.language.core import _unwrap_if_constexpr
from triton.runtime.jit import constexpr_function

from triton.experimental.gluon.language._layouts import PaddedSharedLayout, SharedLayout

__all__ = [
    "PartitionedSharedLayout",
    "make_partitioned_dot_layouts",
]


@dataclass(frozen=True)
class PartitionedSharedLayout(SharedLayout):
    """
    Represents a partitioned shared memory layout that splits a tensor
    across multiple physical shared memory partitions.

    This reduces shared memory partition conflicts by placing different
    pieces of a tensor in separate physical memory slots.

    Args:
        num_partitions (int): Number of physical memory partitions.
        num_groups (int): Number of groups (each group has num_partitions pieces).
        partition_dim (int): Dimension along which to partition.
        partition_layout (SharedLayout): Inner layout for each piece
            (e.g., SwizzledSharedLayout or PaddedSharedLayout).
    """
    num_partitions: int
    num_groups: int
    partition_dim: int
    partition_layout: SharedLayout

    def __post_init__(self):
        super().__setattr__("num_partitions", _unwrap_if_constexpr(self.num_partitions))
        super().__setattr__("num_groups", _unwrap_if_constexpr(self.num_groups))
        super().__setattr__("partition_dim", _unwrap_if_constexpr(self.partition_dim))
        super().__setattr__("partition_layout", _unwrap_if_constexpr(self.partition_layout))

        is_power_of_2 = lambda n: n > 0 and n & (n - 1) == 0
        assert is_power_of_2(self.num_partitions), \
            f"PartitionedSharedLayout num_partitions must be a power of two, got {self.num_partitions}"
        assert is_power_of_2(self.num_groups), \
            f"PartitionedSharedLayout num_groups must be a power of two, got {self.num_groups}"
        assert self.partition_dim >= 0, \
            f"PartitionedSharedLayout partition_dim must be non-negative, got {self.partition_dim}"
        assert isinstance(self.partition_layout, SharedLayout), \
            f"PartitionedSharedLayout partition_layout must be a SharedLayout, got {type(self.partition_layout)}"

    def _to_ir(self, builder):
        partition_layout_ir = self.partition_layout._to_ir(builder)
        return builder.get_partitioned_shared_layout(
            self.num_partitions,
            self.num_groups,
            self.partition_dim,
            partition_layout_ir,
        )

    def mangle(self) -> str:
        partition_mangle = self.partition_layout.mangle()
        return f"Partitioned_{self.num_partitions}_{self.num_groups}_{self.partition_dim}_{partition_mangle}_Partitioned"

    def __hash__(self):
        return hash((
            self.num_partitions,
            self.num_groups,
            self.partition_dim,
            self.partition_layout,
        ))


@constexpr_function
def _make_partitioned_dot_operand_layout(sublayout, partition_dim, num_partitions, block_mn_size, warp_coverage, order):
    """Build a ``PartitionedSharedLayout`` for one GEMM operand.

    The operand tile (``sublayout``) is split along ``partition_dim`` into
    ``num_partitions * num_groups`` logical pieces, where
    ``num_groups = block_mn_size // warp_coverage``.  Each piece carries an
    inner ``PaddedSharedLayout`` rebuilt with an identity mapping using the
    original sublayout's ``interval_padding_pairs`` and ``cga_layout``.
    """
    is_power_of_2 = lambda n: n > 0 and n & (n - 1) == 0
    num_groups = block_mn_size // warp_coverage
    assert num_groups >= 1, f"block dim ({block_mn_size}) must be >= {warp_coverage}"
    assert is_power_of_2(num_groups), \
        f"block_dim / warp_coverage = {num_groups} must be a power of 2"

    num_logical_pieces = num_partitions * num_groups
    inner_shape = list(sublayout.shape)
    assert inner_shape[partition_dim] % num_logical_pieces == 0
    inner_shape[partition_dim] //= num_logical_pieces

    # TODO: when the partitioned dimension is the contiguous (fastest-varying)
    # dimension, the interval_padding_pairs may need adjustment to preserve
    # bank-conflict avoidance properties for the smaller piece shape.
    inner = PaddedSharedLayout.with_identity_for(sublayout.interval_padding_pairs, inner_shape, order,
                                                 sublayout.cga_layout)
    return PartitionedSharedLayout(num_partitions, num_groups, partition_dim, inner)


@constexpr_function
def make_partitioned_dot_layouts(block_m, block_n, original_layout_a, original_layout_b, num_warps, instr_shape,
                                 a_transposed=False, b_transposed=False):
    """Create partitioned shared memory layouts and WMMA layout for a GFX1250 GEMM
       in order to avoid LDS partition conflicts.

    Args:
        block_m: M dimension tile size.  Must be at least
            ``WARP_TILES_M * instr_shape[0]`` (the per-CTA M extent covered by
            one partition group), and a power-of-2 multiple thereof.
        block_n: N dimension tile size.  Must be at least
            ``WARP_TILES_N * instr_shape[1]``, and a power-of-2 multiple
            thereof.
        original_layout_a: ``PaddedSharedLayout`` for operand A.  Shape is
            ``[block_m, block_k]`` when not transposed (K contiguous) and
            ``[block_k, block_m]`` when transposed (M contiguous).
        original_layout_b: ``PaddedSharedLayout`` for operand B.  Shape is
            ``[block_k, block_n]`` when not transposed (N contiguous) and
            ``[block_n, block_k]`` when transposed (K contiguous).
        num_warps: Number of warps per CTA.  Currently must be 4 or 8.
        instr_shape: WMMA instruction shape as ``[M, N, K]``.
        a_transposed: Whether A is transposed in shared memory, i.e. M is
            the contiguous axis instead of K.
        b_transposed: Whether B is transposed in shared memory, i.e. K is
            the contiguous axis instead of N.

    Returns:
        A tuple ``(shared_layout_a, shared_layout_b, wmma_layout)``.
    """
    from triton.experimental.gluon.language.amd._layouts import AMDWMMALayout

    INSTR_SHAPE = list(instr_shape)

    assert num_warps in (4, 8), f"Only 4 or 8 warps are currently supported, got {num_warps}"

    NUM_PARTITIONS = 2

    # The caller passes each sublayout with its contiguous axis at dim 1 of the
    # tile (i.e. the tile's shape is already expressed in memory order):
    #   A non-transposed: [block_m, block_k] — K contiguous at dim 1
    #   A transposed:     [block_k, block_m] — M contiguous at dim 1
    #   B non-transposed: [block_k, block_n] — N contiguous at dim 1
    #   B transposed:     [block_n, block_k] — K contiguous at dim 1
    # So the linear component is always built with order [1, 0] regardless of
    # transposition. TDM additionally requires order [rank-1, ..., 0].
    order = [1, 0]

    # WMMA CTA layout: Below, M runs vertically (rows) and N
    # horizontally (cols); each cell is one INSTR_SHAPE-sized instruction tile,
    # labelled with the warp that computes it.  ``warp_bases`` map warp-id bits
    # to (M, N) tile offsets, ``reg_bases`` map register-id
    # (more specifically, instruction repetition registers) bits the same way.
    #
    # 4-warp case (warp_bases = [[2, 1], [1, 0]], reg_bases = [[2, 0]]):
    #
    #   M=0:  w0 w1   <- second tile computed by w1 (reg=1)
    #   M=1:  w2 w3
    #   M=2:  w0 w1   <- first tile computed by w1 (reg=0)
    #   M=3:  w2 w3
    #
    # Such layout allows w0 and w1 as well as w2 and w3 to read different A/B
    # operand data blocks in a single instruction, which is necessary precondition
    # for avoiding LDS partition conflicts.
    if num_warps == 4:
        warp_bases = [[2, 1], [1, 0]]
        reg_bases = [[2, 0]]
    else:  # num_warps == 8
        # Same idea as the 4-warp case, but the third warp bit replaces the
        # register bit. This means there's no need to define repetition registers,
        # single instruction CTA layout is enough to describe the layout we need.
        # Each warp now owns a single instruction tile and the extra warp dimension
        # is folded into the M-axis of the warp grid.
        warp_bases = [[2, 1], [1, 0], [2, 0]]
        reg_bases = []

    # The tile extent along each dimension is 2^m, where m is the largest
    # basis component in that dimension across both warp and register bases.
    def _tile_extent(dim):
        m = max((b[dim] for b in warp_bases + reg_bases), default=0)
        return 1 << m

    WARP_TILES_M = _tile_extent(0)
    WARP_TILES_N = _tile_extent(1)

    wmma_layout = AMDWMMALayout(3, True, warp_bases, reg_bases, INSTR_SHAPE)

    # Per-CTA extent that one warp+register cycle covers in each dimension.
    warp_coverage_m = WARP_TILES_M * INSTR_SHAPE[0]
    warp_coverage_n = WARP_TILES_N * INSTR_SHAPE[1]

    # Partition A along its M axis and B along its N axis.  These dims live
    # at different positions in the tile depending on transposition (the
    # contiguous axis is always at dim 1, so M / N moves to dim 0 or dim 1).
    a_partition_dim = 1 if a_transposed else 0
    b_partition_dim = 0 if b_transposed else 1

    shared_layout_a = _make_partitioned_dot_operand_layout(original_layout_a, partition_dim=a_partition_dim,
                                                           num_partitions=NUM_PARTITIONS, block_mn_size=block_m,
                                                           warp_coverage=warp_coverage_m, order=order)
    shared_layout_b = _make_partitioned_dot_operand_layout(original_layout_b, partition_dim=b_partition_dim,
                                                           num_partitions=NUM_PARTITIONS, block_mn_size=block_n,
                                                           warp_coverage=warp_coverage_n, order=order)

    return shared_layout_a, shared_layout_b, wmma_layout
