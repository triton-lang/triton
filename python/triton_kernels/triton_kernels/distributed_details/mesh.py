from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import cache
from math import prod
from typing import Iterator, Protocol

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

# ------------------------------------------------------------
# Symmetric memory pool
# ------------------------------------------------------------


class Mesh(Protocol):

    @property
    def local_rank(self) -> int:
        ...

    @property
    def world_size(self) -> int:
        ...

    def get_global_rank(self, group_rank: int) -> int:
        ...

    def get_group_rank(self, global_rank: int) -> int:
        ...


class _LocalMesh:

    @property
    def local_rank(self) -> int:
        return 0

    @property
    def world_size(self) -> int:
        return 1

    def get_global_rank(self, group_rank: int) -> int:
        return default_mesh().local_rank

    def get_group_rank(self, global_rank: int) -> int:
        if global_rank == default_mesh().local_rank:
            return 0
        raise ValueError(f"Rank {global_rank} is not local")


class _TorchMesh:
    _process_group: dist.ProcessGroup | None

    @property
    def process_group(self) -> dist.ProcessGroup:
        if self._process_group is None:
            self._process_group = dist.group.WORLD
            assert self._process_group is not None, ("process_group called before comms initialization")
        return self._process_group

    def __init__(self, group: dist.ProcessGroup | None = None) -> None:
        self._process_group = group

    @property
    def local_rank(self) -> int:
        return dist.get_rank(self.process_group)

    @property
    def world_size(self) -> int:
        return dist.get_world_size(self.process_group)

    def get_global_rank(self, group_rank: int) -> int:
        return (group_rank if self.process_group is None else dist.get_global_rank(self.process_group, group_rank))

    def get_group_rank(self, global_rank: int) -> int:
        return (global_rank if self.process_group is None else dist.get_group_rank(self.process_group, global_rank))


@cache
def default_mesh() -> Mesh:
    return _TorchMesh()


@cache
def local_mesh() -> Mesh:
    return _LocalMesh()


def torch_mesh(group: dist.ProcessGroup) -> Mesh:
    return _TorchMesh(process_group=group)


class MockSymmetricMemoryHandle:

    def barrier(self, channel: int = 0) -> None:
        pass


_CURRENT_MESH = ContextVar[Mesh | None](__name__ + "._CURRENT_MESH", default=None)


def current_mesh() -> Mesh:
    mesh = _CURRENT_MESH.get()
    return mesh if mesh is not None else default_mesh()


@contextmanager
def mesh(mesh: Mesh) -> Iterator[Mesh]:
    # In Python 3.14, this won't be necessary, as the token returned by ContextVar.set
    # is already a context manager
    token = _CURRENT_MESH.set(mesh)
    try:
        yield mesh
    finally:
        _CURRENT_MESH.reset(token)


@dataclass
class _MemoryRegion:
    base: int
    size: int
    alignment: int


class SymmetricMemoryPool:
    _is_initialized: bool
    size: int
    bufs: tuple[torch.Tensor, ...]
    hdl: symm_mem._SymmetricMemory | MockSymmetricMemoryHandle
    regions: dict[str, _MemoryRegion]
    mesh: _TorchMesh

    def __init__(self, mesh: Mesh | None = None) -> None:
        if mesh is None:
            mesh = current_mesh()
        assert isinstance(mesh, _TorchMesh)
        self._is_initialized = False
        self.regions = {}
        self.mesh = mesh

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
            tensor.set_(storage, region_start // elem_size, torch.Size(shape))
            if clear and i == self.mesh.local_rank:
                tensor.zero_()
            tensors.append(tensor)
        if clear:
            self.hdl.barrier(channel=0)

        return tuple(tensors)

    def _initialize(
        self,
        device: torch.device,
    ) -> None:
        if self._is_initialized:
            return

        self.size = int(sum(region.size for region in self.regions.values()))
        if self.mesh.world_size > 1:
            buf = symm_mem.empty((self.size, ), dtype=torch.uint8, device=device)
            self.hdl = symm_mem.rendezvous(buf, group=self.mesh.process_group)
            self.bufs = tuple(self.hdl.get_buffer(r, buf.shape, buf.dtype) for r in range(self.mesh.world_size))
            self.hdl.barrier(channel=0)
        else:
            buf = torch.empty((self.size, ), dtype=torch.uint8, device=device)
            self.hdl = MockSymmetricMemoryHandle()
            self.bufs = (buf, )
        self._is_initialized = True

    def initialize_matmul(
        self,
        n_tokens_global: int,
        d_input: int,
        d_model: int,
        n_expts_act: int,
        n_expts_tot: int,
        dtype: torch.dtype,
        device: torch.device,
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
        n_bytes_ep_to_dp = ((n_tokens_global // self.mesh.world_size) * n_expts_act * d_model * elem_size)

        offset = self._reserve_region("topk_vals", n_bytes_topk_vals, 128, 0)
        offset = self._reserve_region("topk_y_indx", n_bytes_topk_indx, 128, offset)
        offset = self._reserve_region("topk_bitmatrix", n_bytes_topk_bitmatrix, 128, offset)
        offset = self._reserve_region("ep_to_dp", n_bytes_ep_to_dp, 128, offset)
        offset = self._reserve_region("dp_to_ep", n_bytes_dp_to_ep, 128, offset)
        self._initialize(device=device)
