from __future__ import annotations

from dataclasses import dataclass
from triton.language.core import _unwrap_if_constexpr

from triton.experimental.gluon.language._layouts import SharedLayout

__all__ = [
    "PartitionedSharedLayout",
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
