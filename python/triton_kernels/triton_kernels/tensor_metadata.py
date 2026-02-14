from __future__ import annotations

from dataclasses import InitVar, dataclass, field, replace
from typing import TypeVar

import torch

from .tensor_details.dtype import FloatType, IntegerType
from .tensor_details.sharding import LocalSharding, RangeSharding, Sharding


@dataclass(kw_only=True)
class TensorSharding:
    dim: int
    sharding: Sharding
    sharding_size: int | torch.Tensor

    def map(self, idxs: torch.Tensor) -> torch.Tensor:
        return self.sharding.map(idxs, self.sharding_size)

    def range_for_rank(self, rank: int) -> torch.Tensor:
        return self.sharding.range_for_rank(rank, self.sharding_size)

    @property
    def uniform_width(self) -> int | None:
        return (self.sharding.uniform_width(self.sharding_size) if isinstance(self.sharding_size, int) else None)

    @property
    def is_fully_replicated(self) -> bool:
        return self.sharding.is_fully_replicated

    @property
    def is_local(self) -> bool:
        return self.sharding.is_local


@dataclass(kw_only=True)
class TensorMetadata:
    # Constructor arguments

    # Data type
    dtype: IntegerType | FloatType

    # Global shape
    shape: list[int | torch.Tensor] | None

    # Sharding and sharding dimension
    sharding: InitVar[Sharding | None] = None
    sharding_dim: InitVar[int | None] = None

    # Upper bounds on the shape of the local shard; if no dynamic dims, this may be
    # unspecified, in which case it will be assumed to be the same as self.local_shape.
    local_shape_max: list[int] | None = None

    # Computed values

    # Sharding, or None
    tensor_sharding: TensorSharding | None = field(init=False, default=None)

    # Dimension indices whose sizes are not known at compile time.
    dynamic_dims: list[int] = field(init=False)

    # Shape of the *local* shard, if sharded. (same as self.shape if unsharded)
    local_shape: list[int | torch.Tensor] = field(init=False)

    # Upper bounds on the shape of the global tensor. (same as self.local_shape_max if unsharded)
    global_shape_max: list[int] = field(init=False)

    def __post_init__(
        self,
        sharding: Sharding | None = None,
        sharding_dim: int | None = None,
        *,
        default_shape: list[int] | None = None,
    ) -> None:
        if self.shape is None:
            if default_shape is None:
                raise ValueError("shape must be provided")
            if self.dtype.bitwidth < 8:
                raise ValueError("shape must be provided for sub-byte types")
            if self.sharding is not None:
                raise ValueError("shape must be provided if sharding")
            self.shape = default_shape

        if (sharding is not None) != (sharding_dim is not None):
            raise ValueError("sharding and sharding_dim must be specified together")

        self.shape = list(self.shape)

        # validate shape: all elements must be `int` or numel-1 `torch.Tensor`
        is_int = lambda s: isinstance(s, int)
        is_item = lambda s: hasattr(s, "numel") and s.numel() == 1
        assert all(is_int(s) or is_item(s) for s in self.shape)

        self.dynamic_dims = [i for i, s in enumerate(self.shape) if not is_int(s)]

        self.local_shape = list(self.shape)

        if sharding is not None:
            assert not sharding.is_local, (
                "Do not use LocalSharding directly; use get_sharding_local_if_unset() if needed")
            if sharding_dim < 0 or sharding_dim >= len(self.shape):
                raise ValueError("sharding_dim out of range")
            s = self.shape[sharding_dim]
            self.tensor_sharding = TensorSharding(
                dim=sharding_dim,
                sharding=sharding,
                sharding_size=s,
            )
            local_range = self.tensor_sharding.range_for_rank(sharding.mesh.local_rank)
            local_size = local_range[1] - local_range[0]
            assert isinstance(local_size, torch.Tensor)
            if is_int(self.shape[sharding_dim]):
                local_size = int(local_size.item())
            self.local_shape[sharding_dim] = local_size

        if self.local_shape_max is None:
            if self.dynamic_dims:
                raise ValueError("local_shape_max must be specified with dynamic dims")
            self.local_shape_max = list(self.local_shape)
        else:
            if len(self.local_shape_max) != len(self.local_shape):
                raise ValueError("local_shape_max and shape must have the same length")
            for i, (s, smax) in enumerate(zip(self.local_shape, self.local_shape_max)):
                if is_int(s) and s > smax:
                    raise ValueError(f"Size {s} > max {smax} along dim {i}")

        self.global_shape_max = list(self.local_shape_max)
        if sharding is not None:
            self.global_shape_max[sharding_dim] = sharding.max_global_size(self.local_shape_max[sharding_dim])


def get_sharding_local_if_unset(t: TensorMetadata, dim: int) -> TensorSharding:
    if ts := t.tensor_sharding:
        if dim != ts.dim:
            raise ValueError(f"Tensor is already sharded along dimension {ts.dim}")
        return ts

    return TensorSharding(
        dim=dim,
        sharding=LocalSharding(),
        sharding_size=t.shape[dim],
    )


T = TypeVar("T", bound=TensorMetadata)


def extend_sharding(t: T, dim: int, sharding: RangeSharding | None = None) -> T:
    if t.tensor_sharding is not None:
        raise ValueError("Tensor is already sharded")
    if sharding is None:
        sharding = RangeSharding()

    new_shape = list(t.shape)
    new_shape[dim] *= sharding.n_shards

    return replace(t, shape=new_shape, sharding=sharding, sharding_dim=dim)
