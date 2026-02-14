from dataclasses import dataclass
from typing import Any, cast, override

import nvshmem.core as nvshmem
import nvshmem.core.interop.torch as nvshmem_torch
import torch
import torch.cuda
import torch.distributed as tdist
from cuda.core.experimental import Device  # type: ignore

from .shmem import Buffer, Shmem, Team


@dataclass
class NVTeam(Team):
    _h: Any

    @override
    def sync(self) -> None:
        s = nvshmem.NvshmemStream(torch.cuda.current_stream())
        nvshmem.sync(self._h, s)

    @override
    def barrier(self) -> None:
        s = nvshmem.NvshmemStream(torch.cuda.current_stream())
        nvshmem.barrier(self._h, s)


@dataclass
class NVBuffer(Buffer):
    _local_tensor: torch.Tensor

    @property
    @override
    def buffer(self) -> torch.Tensor:
        return self._local_tensor

    @override
    def peer_buffer(self, global_rank: int) -> torch.Tensor:
        if global_rank == nvshmem.my_pe():
            return self.buffer

        return cast(torch.Tensor, nvshmem.get_peer_tensor(self._local_tensor, global_rank))

    @override
    def free(self):
        # TODO(tudor): Do not free. There is a bug in nvshmem4py whose fix is awaiting release
        # as of 2/10/2026.
        # nvshmem_torch.free_tensor(self._local_tensor)
        pass


class NVShmem(Shmem):

    @override
    def create_team(self, process_group: tdist.ProcessGroup) -> Team | None:
        local_rank = tdist.get_rank(process_group)
        if local_rank == -1:
            # Not a member of the group; does not need to participate in team creation.
            return None

        uids = [nvshmem.get_team_unique_id() if local_rank == 0 else None]
        tdist.broadcast_object_list(uids, src=tdist.get_global_rank(process_group, 0), group=process_group)

        config = nvshmem.TeamConfig()
        config.version = 2
        config.num_contexts = 1
        config.uniqueid = uids[0]

        team = nvshmem.team_init(
            team_config=config,
            config_mask=0,
            npes=tdist.get_world_size(process_group),
            pe_idx_in_team=local_rank,
        )

        return NVTeam(
            process_group=process_group,
            _h=team,
        )

    @override
    def allocate(self, size: int) -> Buffer:
        # TODO: alignment? nvshmem_torch doesn't have alignment APIs
        t = cast(torch.Tensor, nvshmem_torch.bytetensor(shape=(size, )))
        return NVBuffer(_local_tensor=t)

    @override
    def sync(self) -> None:
        s = nvshmem.NvshmemStream(torch.cuda.current_stream())
        nvshmem.sync_all(s)

    @override
    def barrier(self) -> None:
        s = nvshmem.NvshmemStream(torch.cuda.current_stream())
        nvshmem.barrier_all(s)


def init_comms() -> NVShmem:
    if not tdist.is_initialized():
        raise ValueError("Call torch.distributed.init_process_group() first")

    rank = tdist.get_rank()

    # TODO: this assumes the same GPU count for every node.
    device_idx = rank % torch.cuda.device_count()
    torch.cuda.set_device(torch.device(type="cuda", index=device_idx))

    uids = [nvshmem.get_unique_id() if rank == 0 else None]
    tdist.broadcast_object_list(uids, src=0)

    nvshmem.init(
        device=Device(device_idx),
        uid=uids[0],
        rank=rank,
        nranks=tdist.get_world_size(),
        initializer_method="uid",
    )

    return NVShmem()
