from __future__ import annotations

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import triton
import triton.language as tl

from triton._internal_testing import is_cuda, run_in_process
from triton.experimental.gsan import symmetric_memory
from triton.experimental.gsan._allocator import get_runtime_state_layout
from triton.experimental.gsan._testing_utils import atomic_poll, shadow_tensor_for
from triton.experimental.gsan._utils import uint8_cuda_tensor_from_ptr


def _get_free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _local_vector_clocks(device_index: int) -> tuple[torch.Tensor, dict[str, int]]:
    layout = get_runtime_state_layout(device_index)
    region_size = layout["thread_state_stride_bytes"] * layout["num_sms"]
    region = uint8_cuda_tensor_from_ptr(layout["thread_state_base_ptr"], region_size, device_index)
    clocks = torch.as_strided(
        region.view(torch.uint16)[layout["thread_state_header_size_bytes"] // 2:],
        size=(layout["num_sms"], layout["num_threads"]),
        stride=(layout["thread_state_stride_bytes"] // 2, 1),
    )
    return clocks, layout


@triton.jit
def _single_cta_atomic_sync_kernel(counter_ptr, payload_ptr, peer_payload_ptr, num_ready_ptr, seen_peer_ptr,
                                   payload_value, num_gpus):
    tl.store(payload_ptr, payload_value)

    num_ready = tl.atomic_add(counter_ptr, 1, sem="acq_rel", scope="sys")
    if num_ready != num_gpus - 1:
        atomic_poll(counter_ptr, num_gpus, sem="acquire", scope="sys")

    seen_peer = tl.load(peer_payload_ptr)
    tl.store(num_ready_ptr, num_ready)
    tl.store(seen_peer_ptr, seen_peer)


@triton.jit
def _single_cta_no_atomic_sync_kernel(payload_ptr, peer_payload_ptr, seen_peer_ptr, payload_value):
    tl.store(payload_ptr, payload_value)
    seen_peer = tl.load(peer_payload_ptr)
    tl.store(seen_peer_ptr, seen_peer)


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

    local_shadow = shadow_tensor_for(buf)
    peer_shadow = shadow_tensor_for(peer_buf)
    assert local_shadow.numel() == peer_shadow.numel()

    local_shadow.fill_(rank + 3)
    hdl.barrier(channel=0)
    assert torch.all(peer_shadow == (peer + 3)).item()

    if rank == 0:
        peer_shadow.fill_(29)
    hdl.barrier(channel=0)
    if rank == 1:
        assert torch.all(local_shadow == 29).item()

    local_clocks, layout = _local_vector_clocks(rank)
    local_clocks.zero_()
    local_tid = rank * layout["num_sms"]
    peer_tid = peer * layout["num_sms"]
    local_clocks[0, local_tid] = rank + 11
    hdl.barrier(channel=0)
    synced_clocks, _ = _local_vector_clocks(rank)
    assert torch.all(synced_clocks[:, local_tid] == (rank + 11)).item()
    assert torch.all(synced_clocks[:, peer_tid] == (peer + 11)).item()

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


def _run_single_cta_atomic_sync_check(rank: int, world_size: int) -> None:
    dev = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(dev)

    peer = (rank + 1) % world_size
    state = symmetric_memory.empty((2, ), dtype=torch.int32, device=dev)
    state.zero_()

    hdl = symmetric_memory.rendezvous(state, group=dist.group.WORLD)
    counter = hdl.get_buffer(0, (1, ), state.dtype, storage_offset=0)
    peer_payload = hdl.get_buffer(peer, (1, ), state.dtype, storage_offset=1)
    local_payload = state[1:]
    num_ready = torch.full((1, ), -1, dtype=torch.int32, device=dev)
    seen_peer = torch.full((1, ), -1, dtype=torch.int32, device=dev)

    hdl.barrier(channel=0)
    _single_cta_atomic_sync_kernel[(1, )](
        counter,
        local_payload,
        peer_payload,
        num_ready,
        seen_peer,
        rank + 1,
        world_size,
        num_warps=1,
    )
    torch.cuda.synchronize()

    assert 0 <= int(num_ready.item()) < world_size
    assert int(seen_peer.item()) == peer + 1

    all_num_ready = [None] * world_size
    dist.all_gather_object(all_num_ready, int(num_ready.item()))
    if rank == 0:
        assert sorted(all_num_ready) == list(range(world_size))
        assert int(state[0].item()) == world_size

    dist.barrier()
    torch.cuda.synchronize()
    hdl.close()
    hdl.close()


def _run_single_cta_no_atomic_sync_check(rank: int, world_size: int) -> None:
    triton.knobs.compilation.instrumentation_mode = "gsan"

    dev = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(dev)

    peer = (rank + 1) % world_size
    payload = symmetric_memory.empty((1, ), dtype=torch.int32, device=dev)
    payload.zero_()

    hdl = symmetric_memory.rendezvous(payload, group=dist.group.WORLD)
    peer_payload = hdl.get_buffer(peer, payload.shape, payload.dtype)
    seen_peer = torch.full((1, ), -1, dtype=torch.int32, device=dev)

    hdl.barrier(channel=0)
    _single_cta_no_atomic_sync_kernel[(1, )](
        payload,
        peer_payload,
        seen_peer,
        rank + 1,
        num_warps=1,
    )
    torch.cuda.synchronize()


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


def _distributed_worker_single_cta_atomic_sync(rank: int, world_size: int, master_port: int) -> None:
    dev = f"cuda:{rank}"
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size, device_id=torch.device(dev))
    try:
        _run_single_cta_atomic_sync_check(rank, world_size)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _distributed_worker_single_cta_no_atomic_sync(rank: int, world_size: int, master_port: int) -> None:
    dev = f"cuda:{rank}"
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    torch.cuda.set_device(dev)
    try:
        _run_single_cta_no_atomic_sync_check(rank, world_size)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _run_single_cta_no_atomic_sync_failure_case() -> None:
    world_size = 2
    master_port = _get_free_tcp_port()
    mp.spawn(
        _distributed_worker_single_cta_no_atomic_sync,
        args=(world_size, master_port),
        nprocs=world_size,
        join=True,
    )


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


@pytest.mark.skipif(not is_cuda(), reason="requires CUDA backend")
def test_gsan_symmetric_memory_single_cta_atomic_sync():
    if torch.cuda.device_count() < 2:
        pytest.skip("requires 2 CUDA devices")

    world_size = 2
    master_port = _get_free_tcp_port()
    mp.spawn(
        _distributed_worker_single_cta_atomic_sync,
        args=(world_size, master_port),
        nprocs=world_size,
        join=True,
    )


@pytest.mark.skipif(not is_cuda(), reason="requires CUDA backend")
def test_gsan_symmetric_memory_single_cta_no_atomic_sync_fails():
    if torch.cuda.device_count() < 2:
        pytest.skip("requires 2 CUDA devices")

    result = run_in_process(_run_single_cta_no_atomic_sync_failure_case, env={"CUDA_LAUNCH_BLOCKING": "1"})
    assert result.exc is not None
    assert "race detected" in result.driver_stderr_output


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
