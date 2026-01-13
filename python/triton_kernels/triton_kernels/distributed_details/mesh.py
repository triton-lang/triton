from abc import ABC, abstractmethod
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import cache
from math import prod
from typing import Iterator, override

import torch
import torch.distributed as dist

from .comms import shmem
from .shmem import Buffer, Collective, NoOpShmem

# ------------------------------------------------------------
# Symmetric memory pool
# ------------------------------------------------------------


class ProcessGroup(ABC):
    _shmem_collective: Collective | None

    @property
    def shmem_collective(self) -> Collective:
        c = self._shmem_collective
        if c is None:
            raise ValueError("Must be group member")
        return c

    @property
    def is_group_member(self) -> bool:
        return self._shmem_collective is not None

    def check_group_member(self) -> None:
        _ = self.shmem_collective

    @property
    @abstractmethod
    def local_rank(self) -> int:
        ...

    @property
    def global_rank(self):
        return default_process_group().local_rank

    @property
    @abstractmethod
    def world_size(self) -> int:
        ...

    @abstractmethod
    def get_global_rank(self, group_rank: int) -> int:
        ...

    @abstractmethod
    def get_group_rank(self, global_rank: int) -> int:
        ...

    def sync(self) -> None:
        self.shmem_collective.sync()

    def barrier(self) -> None:
        self.shmem_collective.barrier()


class LocalProcessGroup(ProcessGroup):

    def __init__(self):
        self._shmem_collective = NoOpShmem()

    @property
    @override
    def local_rank(self) -> int:
        return 0

    @property
    @override
    def world_size(self) -> int:
        return 1

    @override
    def get_global_rank(self, group_rank: int) -> int:
        if group_rank == 0:
            return self.global_rank
        raise ValueError(f"The only valid rank in local meshes is 0, not {group_rank}")

    @override
    def get_group_rank(self, global_rank: int) -> int:
        if global_rank == self.global_rank:
            return 0
        raise ValueError(f"Rank {global_rank} is not local")


class TorchProcessGroup(ProcessGroup):
    torch_process_group: dist.ProcessGroup

    def __init__(self, process_group: dist.ProcessGroup | None = None) -> None:
        assert dist.is_initialized(), "torch_process_group() called before initialization"

        if process_group is None:
            process_group = dist.group.WORLD
            assert process_group is not None

        self.torch_process_group = process_group

        if process_group == dist.GroupMember.NON_GROUP_MEMBER:
            self._shmem_collective = None
        elif process_group is dist.group.WORLD:
            self._shmem_collective = shmem()
        else:
            team = shmem().create_team(process_group)
            assert team is not None, "BUG: team is None on group member"
            self._shmem_collective = team

    @property
    @override
    def local_rank(self) -> int:
        self.check_group_member()
        return dist.get_rank(self.torch_process_group)

    @property
    @override
    def world_size(self) -> int:
        self.check_group_member()
        return dist.get_world_size(self.torch_process_group)

    @override
    def get_global_rank(self, group_rank: int) -> int:
        self.check_group_member()
        return dist.get_global_rank(self.torch_process_group, group_rank)

    @override
    def get_group_rank(self, global_rank: int) -> int:
        self.check_group_member()
        return dist.get_group_rank(self.torch_process_group, global_rank)


@cache
def default_process_group() -> ProcessGroup:
    return TorchProcessGroup()


@cache
def local_process_group() -> ProcessGroup:
    return LocalProcessGroup()


def torch_process_group(process_group: dist.ProcessGroup) -> ProcessGroup:
    return TorchProcessGroup(process_group=process_group)


_CURRENT_PROCESS_GROUP = ContextVar[ProcessGroup | None](__name__ + "._CURRENT_MESH", default=None)


def current_process_group() -> ProcessGroup:
    pg = _CURRENT_PROCESS_GROUP.get()
    return pg if pg is not None else default_process_group()


@contextmanager
def process_group(pg: ProcessGroup) -> Iterator[ProcessGroup]:
    # In Python 3.14, this won't be necessary, as the token returned by ContextVar.set
    # is already a context manager
    token = _CURRENT_PROCESS_GROUP.set(pg)
    try:
        yield pg
    finally:
        _CURRENT_PROCESS_GROUP.reset(token)


@dataclass
class _MemoryRegion:
    base: int
    size: int
    alignment: int


class SymmetricMemoryPool:
    _is_initialized: bool
    size: int
    shmem_buffer: Buffer | None = None
    bufs: tuple[torch.Tensor, ...]
    regions: dict[str, _MemoryRegion]
    process_group: TorchProcessGroup

    def __init__(self, process_group: ProcessGroup | None = None) -> None:
        if process_group is None:
            process_group = current_process_group()
        if not isinstance(process_group, TorchProcessGroup):
            raise ValueError(
                f"SymmetricMemoryPool() requires the current process group to be TorchProcessGroup, not {type(process_group)}"
            )

        self._is_initialized = False
        self.regions = {}
        self.process_group = process_group

    @staticmethod
    def align_up(value: int, alignment: int) -> int:
        if alignment <= 1:
            return value
        return ((value + alignment - 1) // alignment) * alignment

    def _reserve_region(self, name: str, size: int, alignment: int, offset: int) -> int:
        if self._is_initialized:
            raise RuntimeError("Cannot reserve regions after initialization")
        if name in self.regions:
            raise ValueError(f"Region {name} already reserved")
        alignment = max(alignment, 1)
        size_aligned = self.align_up(size, alignment)
        base = self.align_up(offset, alignment)
        end = base + size_aligned
        self.regions[name] = _MemoryRegion(base=base, size=size_aligned, alignment=alignment)
        return end

    def get_tensors(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        region: str,
        clear: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """
        Get symmetric tensors from a reserved region.

        Args:
            shape: Shape of the tensor to allocate.
            dtype: Data type of the tensor to allocate.
            region: Name of the reserved region to allocate from.
            clear: If True, zero out the allocated tensors.
        Returns:
            A tuple of tensors, one per rank in the process group.
        """
        if not self._is_initialized:
            raise RuntimeError("SymmetricMemoryPool is not initialized")
        if not self.process_group.is_group_member:
            raise RuntimeError("get_tensors() called on non-group member")

        region_info = self.regions.get(region)
        if region_info is None:
            raise ValueError(f"Region {region} not found")

        elem_size = torch.empty((), dtype=dtype).element_size()
        if region_info.base % elem_size != 0:
            raise ValueError(f"Region base {region_info.base} not aligned to element size {elem_size}")
        if region_info.alignment % elem_size != 0:
            raise ValueError(f"Region alignment {region_info.alignment} not compatible with element size {elem_size}")

        numel = prod(shape)
        nbytes = numel * elem_size
        region_start = region_info.base
        region_end = region_info.base + region_info.size

        if region_start + nbytes > region_end:
            raise ValueError(
                f"Slice [{region_start}:{region_start + nbytes}) exceeds region {region} bounds [{region_info.base}:{region_end})"
            )

        tensors = []
        for i, buf in enumerate(self.bufs):
            storage = buf.untyped_storage()
            total = storage.nbytes()
            assert region_start + nbytes <= total, (
                f"BUG: Slice [{region_start}:{region_start + nbytes}) exceeds storage size {total} bytes.")
            tensor = torch.empty(0, dtype=dtype, device=buf.device)

            # the torch docs say that stride is optional
            tensor.set_(storage, buf.storage_offset() + region_start // elem_size, torch.Size(shape))  # type: ignore
            if clear and i == self.process_group.local_rank:
                tensor.zero_()
            tensors.append(tensor)
        if clear:
            self.process_group.barrier()

        return tuple(tensors)

    def _initialize(self, ) -> None:
        if self._is_initialized:
            return

        self.size = int(sum(region.size for region in self.regions.values()))
        if not self.process_group.is_group_member:
            self.bufs = ()
        elif isinstance(self.process_group, LocalProcessGroup):
            buf = torch.empty((self.size, ), dtype=torch.uint8)
            self.bufs = (buf, )
        else:
            self.shmem_buffer = shmem().allocate(self.size)
            self.bufs = tuple(
                self.shmem_buffer.peer_buffer(self.process_group.get_global_rank(r))
                for r in range(self.process_group.world_size))
            self.process_group.barrier()

        self._is_initialized = True

    def free(self) -> None:
        if self.shmem_buffer is not None:
            self.shmem_buffer.free()
            self.shmem_buffer = None

    def initialize_matmul(
        self,
        n_tokens_global: int,
        d_input: int,
        d_model: int,
        n_expts_act: int,
        n_expts_tot: int,
        dtype: torch.dtype,
    ) -> None:
        if self._is_initialized:
            return

        BLOCK_N = 32
        BLOCK_M = 32
        n_bytes_topk_vals = (n_tokens_global * n_expts_act * 4)  # topk logits (float32): pessimistic estimate
        n_bytes_topk_indx = n_tokens_global * n_expts_act * 2  # topk indx (int16)
        cdiv = lambda x, y: (x + y - 1) // y
        num_blocks_m = cdiv(n_tokens_global, BLOCK_M)
        num_blocks_n = cdiv(n_expts_tot, BLOCK_N)
        n_bytes_topk_bitmatrix = (num_blocks_m * BLOCK_M * num_blocks_n * BLOCK_N // 32 * 4)  # expt bitmatrix (int32)
        elem_size = torch.empty((), dtype=dtype).element_size()
        n_bytes_dp_to_ep = n_tokens_global * n_expts_act * d_input * elem_size
        n_bytes_ep_to_dp = ((n_tokens_global // self.process_group.world_size) * n_expts_act * d_model * elem_size)

        offset = self._reserve_region("topk_vals", n_bytes_topk_vals, 128, 0)
        offset = self._reserve_region("topk_y_indx", n_bytes_topk_indx, 128, offset)
        offset = self._reserve_region("topk_bitmatrix", n_bytes_topk_bitmatrix, 128, offset)
        offset = self._reserve_region("ep_to_dp", n_bytes_ep_to_dp, 128, offset)
        offset = self._reserve_region("dp_to_ep", n_bytes_dp_to_ep, 128, offset)
        self._initialize()


@contextmanager
def symmetric_memory_pool(process_group: ProcessGroup | None = None, ) -> Iterator[SymmetricMemoryPool]:
    pool = SymmetricMemoryPool(process_group=process_group)
    try:
        yield pool
    finally:
        pool.free()
