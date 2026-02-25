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
from triton.experimental.gsan._utils import shadow_region, uint8_cuda_tensor_from_ptr


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

            dist.barrier(group=subgroup)
            hdl.close()
            hdl.close()
            with pytest.raises(RuntimeError):
                hdl.get_buffer(peer_rank, buf.shape, buf.dtype)

            dist.barrier(group=subgroup)
            hdl2 = symmetric_memory.rendezvous(buf, group=subgroup)
            assert hdl2 is not hdl
            peer_buf2 = hdl2.get_buffer(peer_rank, buf.shape, buf.dtype)

            if subgroup_rank == 0:
                peer_buf2.fill_(73)
            hdl2.barrier(channel=0)
            if subgroup_rank == 1:
                assert torch.all(buf == 73).item()

            del peer_buf2
            torch.cuda.synchronize()
            dist.barrier(group=subgroup)
            hdl2.close()
            hdl2.close()

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


def _run_triton_kernels_convert_dp_to_ep_with_gsan_pool(rank: int, world_size: int) -> None:
    from triton_kernels.distributed import (SymmetricMemoryPool, convert_dp_to_ep, make_expt_assignment,
                                            make_expt_dict_uniform)
    from triton_kernels.distributed_details.mesh import Mesh

    dev = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(dev)

    class _GSanSymmetricMemoryPool(SymmetricMemoryPool):

        def _initialize(self, device: torch.device) -> None:
            if self._is_initialized:
                return
            self.size = int(sum(region.size for region in self.regions.values()))
            self.buf = symmetric_memory.empty((self.size, ), dtype=torch.uint8, device=device)
            self.hdl = symmetric_memory.rendezvous(self.buf, group=self.mesh.process_group)
            self.bufs = tuple(
                self.hdl.get_buffer(r, self.buf.shape, self.buf.dtype) for r in range(self.mesh.world_size))
            self.hdl.barrier(channel=0)
            self._is_initialized = True

    n_tokens_local = 4
    d_model = 16
    n_expts_tot = 4
    n_expts_act = 2
    n_tokens_global = n_tokens_local * world_size

    symm_mem_pool = None
    x_local = None
    x_global = None
    dst_local = None
    expected = None
    try:
        x_local = (
            torch.arange(n_tokens_local * d_model, device=dev, dtype=torch.float32).reshape(n_tokens_local, d_model) +
            rank * 1000.0)
        x_global = torch.empty((n_tokens_global, d_model), dtype=x_local.dtype, device=dev)
        dist.all_gather_into_tensor(x_global, x_local)

        expt_dict = make_expt_dict_uniform(world_size, n_expts_tot)
        expt_assignment = make_expt_assignment(world_size, n_expts_tot, expt_dict, device=dev)
        expt_indx = torch.empty((n_tokens_global, n_expts_act), dtype=torch.int32, device=dev)
        expt_indx[:, 0] = 0
        expt_indx[:, 1] = n_expts_tot // world_size
        gate_indx = torch.arange(n_tokens_global * n_expts_act, device=dev,
                                 dtype=torch.int32).reshape(n_tokens_global, n_expts_act)

        symm_mem_pool = _GSanSymmetricMemoryPool(Mesh(dist.group.WORLD))
        symm_mem_pool.initialize_matmul(
            n_tokens_global=n_tokens_global,
            d_input=d_model,
            d_model=d_model,
            n_expts_act=n_expts_act,
            n_expts_tot=n_expts_tot,
            dtype=x_local.dtype,
            device=dev,
        )

        symm_mem_pool.make_empty(
            shape=(n_tokens_global * n_expts_act, d_model),
            dtype=x_local.dtype,
            region="dp_to_ep",
            clear=True,
        )

        dst_local = convert_dp_to_ep(x_local, expt_assignment, expt_indx, gate_indx, symm_mem_pool)
        expected = torch.zeros_like(dst_local)
        for global_token in range(n_tokens_global):
            for expt_slot in range(n_expts_act):
                expt_id = int(expt_indx[global_token, expt_slot].item())
                dst_rank = int(torch.nonzero(expt_assignment.expt_boolmask[:, expt_id], as_tuple=False)[0].item())
                if dst_rank == rank:
                    dst_row = int(gate_indx[global_token, expt_slot].item())
                    expected[dst_row] = x_global[global_token]

        assert torch.equal(dst_local, expected)
    finally:
        del dst_local
        del expected
        del x_global
        del x_local
        torch.cuda.synchronize()
        dist.barrier()
        dist.barrier()


def _distributed_worker_triton_kernels_convert_dp_to_ep(rank: int, world_size: int, master_port: int) -> None:
    dev = f"cuda:{rank}"
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    torch.cuda.set_device(dev)
    try:
        _run_triton_kernels_convert_dp_to_ep_with_gsan_pool(rank, world_size)
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not is_cuda(), reason="requires CUDA backend")
def test_gsan_symmetric_memory_with_triton_kernels_convert_dp_to_ep():
    pytest.importorskip("triton_kernels.distributed")
    if torch.cuda.device_count() < 2:
        pytest.skip("requires 2 CUDA devices")

    world_size = 2
    master_port = _get_free_tcp_port()
    mp.spawn(
        _distributed_worker_triton_kernels_convert_dp_to_ep,
        args=(world_size, master_port),
        nprocs=world_size,
        join=True,
    )
