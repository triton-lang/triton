from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Never, override

import torch
import torch.distributed as tdist


class Collective(ABC):
    """Base class for collective operations."""

    @abstractmethod
    def sync(self) -> None:
        ...

    @abstractmethod
    def barrier(self) -> None:
        ...


@dataclass
class Team(Collective):
    """Team (process group).

    Maps 1:1 to a torch ProcessGroup; rank i in the team map 1:1 to ranks in the ProcessGroup."""

    process_group: tdist.ProcessGroup


class Buffer(ABC):
    """Buffer; represents a symmetric ByteTensor."""

    @property
    @abstractmethod
    def buffer(self) -> torch.Tensor:
        ...

    @abstractmethod
    def peer_buffer(self, global_rank: int) -> torch.Tensor:
        ...

    @abstractmethod
    def free(self) -> None:
        """Free the buffer; explicit deallocation is required."""
        ...


class Shmem(Collective):

    @abstractmethod
    def create_team(self, process_group: tdist.ProcessGroup) -> Team | None:
        """Create a team with the same members as `process_group`."""
        ...

    @abstractmethod
    def allocate(self, size: int) -> Buffer:
        """Allocate a symmetric buffer of given size."""
        ...


class NoOpShmem(Shmem):

    @override
    def create_team(self, process_group: tdist.ProcessGroup) -> Never:
        raise NotImplementedError()

    @override
    def allocate(self, size: int) -> Never:
        raise NotImplementedError()

    @override
    def sync(self) -> None:
        pass

    @override
    def barrier(self) -> None:
        pass
