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
def _make_partitioned_dot_operand_layout(sublayout, partition_dim, num_partitions, num_groups, order):
    """Build a ``PartitionedSharedLayout`` for one GEMM operand.

    The operand tile (``sublayout``) is split along ``partition_dim`` into
    ``num_partitions * num_groups`` logical pieces. Each
    piece carries an inner ``PaddedSharedLayout`` rebuilt with an identity
    mapping using the original sublayout's parameters.
    """
    is_power_of_2 = lambda n: n > 0 and n & (n - 1) == 0
    assert num_groups >= 1, f"num_groups ({num_groups}) must be >= 1"
    assert is_power_of_2(num_groups), \
        f"num_groups = {num_groups} must be a power of 2"

    num_logical_pieces = num_partitions * num_groups
    inner_shape = list(sublayout.shape)
    assert inner_shape[partition_dim] % num_logical_pieces == 0, \
        f"partitioned dim ({inner_shape[partition_dim]}) must be divisible by " \
        f"num_partitions * num_groups = {num_logical_pieces}"
    inner_shape[partition_dim] //= num_logical_pieces

    # TODO: when the partitioned dimension is the contiguous (fastest-varying)
    # dimension, the interval_padding_pairs may need adjustment to preserve
    # bank-conflict avoidance properties for the smaller piece shape.
    inner = PaddedSharedLayout.with_identity_for(sublayout.interval_padding_pairs, inner_shape, order,
                                                 sublayout.cga_layout)
    return PartitionedSharedLayout(num_partitions, num_groups, partition_dim, inner)


@constexpr_function
def make_partitioned_dot_layouts(block_m, block_n, original_layout_a, original_layout_b, num_warps, instr_shape,
                                 a_transposed=False, b_transposed=False, slice_m=None, slice_n=None):
    """Create partitioned shared memory layouts and WMMA layout for a GFX1250 GEMM
       in order to avoid LDS partition conflicts.

    Args:
        block_m: M dimension tile size of the *shared* operand buffer.  Must be
            at least ``4 * instr_shape[0]`` because the M dimension is split into
            2 partitions and each partition must be at least 2 instructions wide.
        block_n: N dimension tile size of the *shared* operand buffer.  Must be
            at least ``2 * instr_shape[1]`` because the N dimension is split into
            2 partitions and each partition must be at least 1 instruction wide.
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
        slice_m: M dimension of a dot operation after slicing.  Defaults to
            ``block_m`` (unsliced).
        slice_n: N dimension of a dot operation after slicing.  Defaults to
            ``block_n`` (unsliced).

    Returns:
        A tuple ``(shared_layout_a, shared_layout_b, wmma_layout)``.  The
        ``wmma_layout`` is sized for ``slice_m x slice_n``; the two shared
        layouts partition the full ``block_m`` / ``block_n``.
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

    INSTR_M = INSTR_SHAPE[0]
    INSTR_N = INSTR_SHAPE[1]

    is_power_of_2 = lambda n: n > 0 and (n & (n - 1)) == 0

    # The sliced dot compute extent defaults to the full block (unsliced).
    slice_m = block_m if slice_m is None else slice_m
    slice_n = block_n if slice_n is None else slice_n

    assert is_power_of_2(INSTR_M) and is_power_of_2(INSTR_N), \
        f"instr_shape M/N must be powers of 2, got {INSTR_SHAPE}"
    assert block_m % INSTR_M == 0 and block_n % INSTR_N == 0, \
        f"block ({block_m}, {block_n}) must be a multiple of instr_shape ({INSTR_M}, {INSTR_N})"
    assert is_power_of_2(slice_m) and is_power_of_2(slice_n), \
        f"slice_m / slice_n must be powers of 2, got ({slice_m}, {slice_n})"
    assert block_m % slice_m == 0 and block_n % slice_n == 0, \
        f"slice ({slice_m}, {slice_n}) must divide block ({block_m}, {block_n})"
    assert slice_m % INSTR_M == 0 and slice_n % INSTR_N == 0, \
        f"slice ({slice_m}, {slice_n}) must be a multiple of instr_shape ({INSTR_M}, {INSTR_N})"

    def _log2(n):
        return n.bit_length() - 1

    # --- Derived layout rule -------------------------------------------------
    #
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
    #
    # The above defined CTALayout tile will then be repeated across the full
    # block (block_m x block_n).  However, this is not always the optimal way to
    # partition the full block.
    # For example, if the block 64x128, the block would look like this:
    #
    #   M=0:  w0 w1 w0 w1 w0 w1 w0 w1
    #   M=1:  w2 w3 w2 w3 w2 w3 w2 w3
    #   M=2:  w0 w1 w0 w1 w0 w1 w0 w1
    #   M=3:  w2 w3 w2 w3 w2 w3 w2 w3
    #
    # From the TDM perspective, wave can only transfer one strided logical piece
    # at a time.  Since we have 8 logical pieces for B tensor above, and only 4
    # warps, TDM transaction will be split into 2 instructions, which is not
    # efficient.
    #
    # to avoid this, we can increase number of consecutive instructions the wave
    # computes.  In the above example, we could create following CTA layout:
    #
    #   M=0:  w0 w0 w0 w0 w1 w1 w1 w1
    #   M=1:  w2 w2 w2 w2 w3 w3 w3 w3
    #   M=2:  w0 w0 w0 w0 w1 w1 w1 w1
    #   M=3:  w2 w2 w2 w2 w3 w3 w3 w3
    #
    # This way, each warp can read 2 larger strided logical pieces at a time,
    # which will produce 1 TDM transaction.
    # In addition to having less transaction, this can also help with global
    # bandwidth, because in N contiguous tensors, it's better if the cache line
    # is not split across multiple waves. This way we can read the whole cache
    # line with a single wave.

    instr_per_slice_m = slice_m // INSTR_M
    instr_per_slice_n = slice_n // INSTR_N
    instr_per_partition_m = 2
    instr_per_partition_n = 1

    # Each slice must cover at least one instruction tile per partition piece,
    # otherwise piece_m / piece_n would round down to 0 and produce a degenerate
    # (non-surjective) WMMA layout with zero warp/register bases.
    assert instr_per_slice_m >= NUM_PARTITIONS * instr_per_partition_m, \
        f"slice_m ({slice_m}) must be at least " \
        f"{NUM_PARTITIONS * instr_per_partition_m} * instr_shape[0] ({INSTR_M})"
    assert instr_per_slice_n >= NUM_PARTITIONS * instr_per_partition_n, \
        f"slice_n ({slice_n}) must be at least " \
        f"{NUM_PARTITIONS * instr_per_partition_n} * instr_shape[1] ({INSTR_N})"

    piece_m = instr_per_slice_m // (NUM_PARTITIONS * instr_per_partition_m)
    piece_n = instr_per_slice_n // (NUM_PARTITIONS * instr_per_partition_n)

    # Registers that walk the contiguous tiles inside one piece.
    m_within = [1 << i for i in range(_log2(piece_m))]
    n_within = [1 << i for i in range(_log2(piece_n))]
    # Register that jump between same-partition pieces of A inside one slice
    # (within-slice group repeats).
    m_group = [piece_m * 2]

    warp_bases = [[instr_per_slice_m // 2, piece_n], [piece_m, 0]]
    reg_bases = []

    if num_warps == 8:
        # In 8 warp case, instead of repetition of warp0, we use that tile for warp4.
        warp_bases.append([m_group[-1], 0])
        m_group = m_group[:-1]

    # Register bases, ordered N-within, M-within, M-group.
    for w in n_within:
        reg_bases.append([0, w])
    for w in m_within:
        reg_bases.append([w, 0])
    for w in m_group:
        reg_bases.append([w, 0])

    wmma_layout = AMDWMMALayout(3, True, warp_bases, reg_bases, INSTR_SHAPE)

    # Full-block partitioning (PartitionedSharedLayout): num_groups counts how
    # many times the [P0, P1] cycle repeats across the full block = block / slice.
    num_groups_m = block_m // slice_m
    num_groups_n = block_n // slice_n

    # Partition A along its M axis and B along its N axis.  These dims live
    # at different positions in the tile depending on transposition (the
    # contiguous axis is always at dim 1, so M / N moves to dim 0 or dim 1).
    a_partition_dim = 1 if a_transposed else 0
    b_partition_dim = 0 if b_transposed else 1

    shared_layout_a = _make_partitioned_dot_operand_layout(original_layout_a, partition_dim=a_partition_dim,
                                                           num_partitions=NUM_PARTITIONS, num_groups=num_groups_m,
                                                           order=order)
    shared_layout_b = _make_partitioned_dot_operand_layout(original_layout_b, partition_dim=b_partition_dim,
                                                           num_partitions=NUM_PARTITIONS, num_groups=num_groups_n,
                                                           order=order)

    return shared_layout_a, shared_layout_b, wmma_layout
