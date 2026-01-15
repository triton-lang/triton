import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from dataclasses import dataclass
from typing import Tuple
from math import prod

# ------------------------------------------------------------
# Symmetric memory pool
# ------------------------------------------------------------


class Mesh:

    def __init__(self, process_group: dist.ProcessGroup):
        self.process_group = process_group
        self.world_size = dist.get_world_size(process_group)
        self.local_rank = dist.get_rank(process_group)


class MockSymmetricMemoryHandle:

    def barrier(self, channel: int = 0):
        pass


@dataclass
class _MemoryRegion:
    base: int
    size: int
    alignment: int


class SymmetricMemoryPool:

    def __init__(self, mesh: Mesh):
        self._is_initialized = False
        self.size = 0
        self.buf = None
        self.bufs = None
        self.hdl = None
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

    def make_empty(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        region: str,
        region_offset: int = 0,
        clear: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Allocate symmetric tensors from a reserved region.

        Args:
            shape: Shape of the tensor to allocate.
            dtype: Data type of the tensor to allocate.
            region: Name of the reserved region to allocate from.
            region_offset: Offset (in bytes) within the region to allocate from.
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
        if region_offset % elem_size != 0:
            raise ValueError(f"Region offset {region_offset} not aligned to element size {elem_size}")

        numel = prod(shape)
        nbytes = numel * elem_size
        region_start = region_info.base + region_offset
        region_end = region_info.base + region_info.size

        if region_start + nbytes > region_end:
            raise ValueError(
                f"Slice [{region_start}:{region_start + nbytes}) exceeds region {region} bounds [{region_info.base}:{region_end})"
            )

        tensors = []
        for buf in self.bufs:
            storage = buf.untyped_storage()
            total = storage.nbytes()
            if region_start + nbytes > total:
                raise ValueError(f"Slice [{region_start}:{region_start + nbytes}) exceeds storage size {total} bytes.")
            tensor = torch.empty(0, dtype=dtype, device=buf.device)
            tensor.set_(storage, region_start // elem_size, torch.Size(shape))
            if clear:
                tensor.zero_()
            tensors.append(tensor)

        return tuple(tensors)

    def _initialize(
        self,
        device: torch.device,
    ) -> None:
        if self._is_initialized:
            return

        self.size = int(sum(region.size for region in self.regions.values()))
        if self.mesh.world_size > 1:
            self.buf = symm_mem.empty((self.size, ), dtype=torch.uint8, device=device)
            self.hdl = symm_mem.rendezvous(self.buf, group=self.mesh.process_group)
            self.bufs = tuple(
                self.hdl.get_buffer(r, self.buf.shape, self.buf.dtype) for r in range(self.mesh.world_size))
            self.hdl.barrier(channel=0)
        else:
            self.buf = torch.empty((self.size, ), dtype=torch.uint8, device=device)
            self.hdl = MockSymmetricMemoryHandle()
            self.bufs = (self.buf, )
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
        n_bytes_topk = n_tokens_global * n_expts_act * 4  # topk logits (float32): pessimistic estimate
        n_bytes_topk += n_tokens_global * n_expts_act * 2  # topk indx (int16)
        cdiv = lambda x, y: (x + y - 1) // y
        num_blocks_m = cdiv(n_tokens_global, BLOCK_M)
        num_blocks_n = cdiv(n_expts_tot, BLOCK_N)
        n_bytes_topk += num_blocks_m * BLOCK_M * num_blocks_n * BLOCK_N // 32 * 4  # expt bitmatrix (int32)
        elem_size = torch.empty((), dtype=dtype).element_size()
        n_bytes_dp_to_ep = n_tokens_global * n_expts_act * d_input * elem_size
        n_bytes_ep_to_dp = (n_tokens_global // self.mesh.world_size) * n_expts_act * d_model * elem_size

        offset = self._reserve_region("topk", n_bytes_topk, 128, 0)
        offset = self._reserve_region("ep_to_dp", n_bytes_ep_to_dp, 128, offset)
        offset = self._reserve_region("dp_to_ep", n_bytes_dp_to_ep, 128, offset)
        self._initialize(device=device)
