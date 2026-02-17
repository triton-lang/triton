from __future__ import annotations

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from triton._internal_testing import is_cuda
from triton.experimental.gsan import symmetric_memory
from triton.experimental.gsan._allocator import get_reserve_pointer, get_reserve_size
from triton.experimental.gsan._testing_utils import shadow_region, uint8_cuda_tensor_from_ptr


def _get_free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _run_symmetric_memory_checks(rank: int, world_size: int) -> None:
    dev = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(dev)

    buf = symmetric_memory.empty((2048, ), dtype=torch.uint8, device=dev)
    buf.fill_(rank + 1)

    hdl = symmetric_memory.rendezvous(buf, group=dist.group.WORLD)
    assert hdl.rank == rank
    assert hdl.world_size == world_size
    assert symmetric_memory.rendezvous(buf, group=dist.group.WORLD) is hdl

    peer = (rank + 1) % world_size
    hdl.barrier(channel=0)
    peer_buf = hdl.get_buffer(peer, buf.shape, buf.dtype)
    assert torch.all(peer_buf == (peer + 1)).item()

    if rank == 0:
        peer_buf.fill_(17)
    hdl.barrier(channel=0)
    if rank == 1:
        assert torch.all(buf == 17).item()

    reserve_ptr = get_reserve_pointer()
    reserve_size = get_reserve_size()
    alloc_size = buf.untyped_storage().nbytes()
    local_shadow_ptr, local_shadow_size = shadow_region(
        buf.untyped_storage().data_ptr(),
        alloc_size,
        reserve_ptr,
        reserve_size,
    )
    peer_shadow_ptr, peer_shadow_size = shadow_region(
        peer_buf.untyped_storage().data_ptr(),
        alloc_size,
        reserve_ptr,
        reserve_size,
    )
    assert local_shadow_size == peer_shadow_size

    local_shadow = uint8_cuda_tensor_from_ptr(local_shadow_ptr, local_shadow_size, rank)
    peer_shadow = uint8_cuda_tensor_from_ptr(peer_shadow_ptr, peer_shadow_size, rank)

    local_shadow.fill_(rank + 3)
    hdl.barrier(channel=0)
    assert torch.all(peer_shadow == (peer + 3)).item()

    if rank == 0:
        peer_shadow.fill_(29)
    hdl.barrier(channel=0)
    if rank == 1:
        assert torch.all(local_shadow == 29).item()

    del peer_buf
    del local_shadow
    del peer_shadow
    torch.cuda.synchronize()

    dist.barrier()
    hdl.close()
    hdl.close()
    with pytest.raises(RuntimeError):
        hdl.get_buffer(rank, buf.shape, buf.dtype)

    dist.barrier()
    hdl2 = symmetric_memory.rendezvous(buf, group=dist.group.WORLD)
    assert hdl2 is not hdl
    peer_buf2 = hdl2.get_buffer(peer, buf.shape, buf.dtype)

    if rank == 0:
        peer_buf2.fill_(43)
    hdl2.barrier(channel=0)
    if rank == 1:
        assert torch.all(buf == 43).item()

    del peer_buf2
    torch.cuda.synchronize()
    dist.barrier()
    hdl2.close()
    hdl2.close()


def _run_subgroup_symmetric_memory_checks(rank: int) -> None:
    subgroup = dist.new_group(ranks=[1, 2], backend="nccl")
    try:
        if rank in (1, 2):
            dev = torch.device(f"cuda:{rank}")
            torch.cuda.set_device(dev)
            subgroup_rank = dist.get_rank(subgroup)
            subgroup_world_size = dist.get_world_size(subgroup)
            assert subgroup_world_size == 2

            buf = symmetric_memory.empty((2048, ), dtype=torch.uint8, device=dev)
            buf.fill_(subgroup_rank + 7)

            hdl = symmetric_memory.rendezvous(buf, group=subgroup)
            assert hdl.rank == subgroup_rank
            assert hdl.world_size == subgroup_world_size

            peer_rank = 1 - subgroup_rank
            hdl.barrier(channel=0)
            peer_buf = hdl.get_buffer(peer_rank, buf.shape, buf.dtype)
            assert torch.all(peer_buf == (peer_rank + 7)).item()

            if subgroup_rank == 0:
                peer_buf.fill_(61)
            hdl.barrier(channel=0)
            if subgroup_rank == 1:
                assert torch.all(buf == 61).item()

            hdl.close()
            hdl.close()
            with pytest.raises(RuntimeError):
                hdl.get_buffer(peer_rank, buf.shape, buf.dtype)

        dist.barrier()
    finally:
        if subgroup != dist.GroupMember.NON_GROUP_MEMBER:
            dist.destroy_process_group(subgroup)


def _distributed_worker(rank: int, world_size: int, master_port: int, run_subgroup_check: bool) -> None:
    dev = f"cuda:{rank}"
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size, device_id=torch.device(dev))
    try:
        if run_subgroup_check:
            _run_subgroup_symmetric_memory_checks(rank)
        else:
            _run_symmetric_memory_checks(rank, world_size)
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not is_cuda(), reason="requires CUDA backend")
def test_gsan_symmetric_memory_rendezvous():
    if torch.cuda.device_count() < 2:
        pytest.skip("requires 2 CUDA devices")

    world_size = 2
    master_port = _get_free_tcp_port()
    mp.spawn(
        _distributed_worker,
        args=(world_size, master_port, False),
        nprocs=world_size,
        join=True,
    )


@pytest.mark.skipif(not is_cuda(), reason="requires CUDA backend")
def test_gsan_symmetric_memory_rendezvous_subgroup_without_global_zero():
    if torch.cuda.device_count() < 3:
        pytest.skip("requires 3 CUDA devices")

    world_size = 3
    master_port = _get_free_tcp_port()
    mp.spawn(
        _distributed_worker,
        args=(world_size, master_port, True),
        nprocs=world_size,
        join=True,
    )
