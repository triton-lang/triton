from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Sequence, final, override

import torch

from triton_kernels.distributed import Mesh, default_mesh, local_mesh


def _size_to_scalar_tensor(
    size: int | torch.Tensor,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.int64,
) -> torch.Tensor:
    if isinstance(size, torch.Tensor):
        if size.numel() != 1:
            raise ValueError("size must be a scalar tensor")
        return size.reshape(()).to(device=device, dtype=dtype)
    return torch.tensor(size, device=device, dtype=dtype)


@dataclass(frozen=True, kw_only=True, slots=True)
class ShardLocation:
    rank: int
    shard: slice


@dataclass(frozen=True, kw_only=True)
class Sharding(ABC):
    mesh: Mesh = field(default=default_mesh())

    @abstractmethod
    def full_map(self, size: int) -> Sequence[ShardLocation]:
        """Return the entire mapping."""
        ...

    @abstractmethod
    def map(self, idxs: torch.Tensor, size: int | torch.Tensor) -> torch.Tensor:
        """Return the mapping of a set of indices.

        Given a set of indices (of shape [k]), return a tensor of shape [replication_factor, k, 2]
        locations. ret[.., 0] is the rank; ret[..., 1] is the local index on that rank.
        """
        ...

    @final
    def range_for_rank(self, rank: int, size: int | torch.Tensor) -> torch.Tensor:
        """Return the contiguous index range [start, end) mapped to a given rank."""
        return self._range_for_rank(rank, _size_to_scalar_tensor(size))

    @abstractmethod
    def _range_for_rank(self, rank: int, size: torch.Tensor) -> torch.Tensor:
        ...

    @property
    def is_fully_replicated(self) -> bool:
        return False

    def uniform_width(self, size: int) -> int | None:
        return None

    @property
    def is_local(self) -> bool:
        return False


@dataclass(frozen=True, kw_only=True, init=False)
class LocalSharding(Sharding):

    def __init__(self):
        super().__init__(mesh=local_mesh())

    @override
    def full_map(self, size: int) -> Sequence[ShardLocation]:
        return (ShardLocation(shard=slice(0, size)), )

    @override
    def map(self, idxs: torch.Tensor, size: int | torch.Tensor) -> torch.Tensor:
        ranks = torch.zeros_like(idxs)
        return torch.stack((ranks, idxs), dim=1).unsqueeze(0)

    @override
    def _range_for_rank(self, rank: int, size: torch.Tensor) -> torch.Tensor:
        zero = torch.zeros((), device=size.device, dtype=size.dtype)
        return torch.stack((zero, size))

    @override
    def uniform_width(self, size: int) -> int | None:
        return size

    @property
    @override
    def is_local(self) -> bool:
        return True

    @property
    @override
    def is_fully_replicated(self) -> bool:
        return True


@dataclass(frozen=True, kw_only=True)
class RangeSharding(Sharding):
    replication_factor: int = 1

    @property
    def n_shards(self) -> int:
        return self.mesh.world_size // self.replication_factor

    @property
    @override
    def is_fully_replicated(self) -> bool:
        return self.n_shards == 1

    # n_shards = n_ranks / replication_factor
    # Shard j is mapped to ranks [j * replication_factor, (j + 1) * replication_factor)
    # Index i is mapped to shard floor(i * n_shards / size)
    #
    # if q, r = divmod(size, n_shards)
    # then shards [0, r) will have size q + 1
    #      shards [r+1, s) will have size q

    def __post_init__(self):
        assert self.mesh.world_size % self.replication_factor == 0

    @override
    def full_map(self, size: int) -> Sequence[ShardLocation]:
        q, r = divmod(size, self.n_shards)
        locations = []
        start = 0
        for shard in range(self.n_shards):
            shard_size = q + (shard < r)
            end = start + shard_size
            for replica in range(self.replication_factor):
                locations.append(ShardLocation(rank=shard * self.replication_factor + replica, shard=slice(start, end)))
            start = end
        return locations

    @override
    def map(self, idxs: torch.Tensor, size: int | torch.Tensor) -> Sequence[torch.Tensor]:
        r = self.replication_factor
        n = idxs.shape[0]
        idxs = torch.atleast_1d(idxs)
        ranks = idxs * self.n_shards // size * self.replication_factor

        idxs = idxs.unsqueeze(0).expand(r, n)
        ranks = ranks.unsqueeze(0).expand(r, n)
        replication = (torch.arange(r, dtype=idxs.dtype, device=idxs.device).unsqueeze(1).expand(r, n))

        return torch.stack((ranks + replication, idxs), dim=2)

    @override
    def _range_for_rank(self, rank: int, size: torch.Tensor) -> torch.Tensor:
        n_shards = torch.tensor(self.n_shards, device=size.device, dtype=size.dtype)
        shard_t = torch.tensor(rank // self.replication_factor, device=size.device, dtype=size.dtype)
        q = size // n_shards
        r = size - q * n_shards
        start = shard_t * q + torch.minimum(shard_t, r)
        end = start + q + (shard_t < r).to(size.dtype)
        return torch.stack((start, end))

    @override
    def uniform_width(self, size: int) -> int | None:
        q, r = divmod(size, self.n_shards)
        return q if r == 0 else None


if __name__ == "__main__":
    SIZE_MAX = 18
    sharding = RangeSharding(replication_factor=2)
    idxs = torch.arange(SIZE_MAX)
    print(sharding.full_map(SIZE_MAX))
    print(sharding.map(idxs, SIZE_MAX))
