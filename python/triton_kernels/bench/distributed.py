import os
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from copy import deepcopy
from dataclasses import dataclass
from typing import Tuple

import triton
import triton.language as tl
import triton_kernels
import triton_kernels.routing
import triton_kernels.swiglu
from triton_kernels.routing import (
    RoutingData,
    GatherIndx,
    ScatterIndx,
    compute_expt_data_torch,
    topk_torch,
    prune_routing,
    routing_from_bitmatrix,
)
from triton_kernels.topk import topk
from triton_kernels.matmul_ogs import matmul_ogs, PrecisionConfig, FlexCtx, FnSpecs, FusedActivation
from triton_kernels.routing_details._routing_compute import _routing_clear_bitmatrix
from triton_kernels.target_info import get_cdna_version, is_hip, is_cuda, cuda_capability_geq
from triton_kernels.tensor_details import layout
from triton_kernels.tensor import Bitmatrix

from bench_utils import quantize_weight


@dataclass
class ReduceScatterMetadata:
    input_split_sizes: list[int]
    ep_indx: torch.Tensor
    EP: int = 1
    TP: int = 1


def _is_distributed_launch() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup() -> Tuple[int, int]:
    if _is_distributed_launch():
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", world_size=world_size, device_id=torch.device(local_rank))
    else:
        world_size = 1
        local_rank = 0
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return local_rank, world_size


def cleanup():
    if _is_distributed_launch():
        dist.barrier()
        dist.destroy_process_group()
    else:
        pass


def broadcast(x: torch.Tensor, src: int = 0, groups: list = None, group_idx: int = None) -> torch.Tensor:
    if _is_distributed_launch():
        group = None
        if groups:
            groups = [dist.new_group(group) for group in groups]
            dist.barrier()
            group = groups[group_idx]
        dist.broadcast(x, src=src, group=group)
        return x
    else:
        return x


def all_gather(x: torch.Tensor, dim=0) -> torch.Tensor:
    if _is_distributed_launch():
        world_size = dist.get_world_size()
        x_list = [torch.empty_like(x) for _ in range(world_size)]
        dist.all_gather(x_list, x)
        return torch.cat(x_list, dim=dim)
    else:
        return x


def reduce_scatter(
    input_tensor: torch.Tensor,
    metadata: ReduceScatterMetadata = None,
    dim: int = 0,
    op: dist.ReduceOp.RedOpType = dist.ReduceOp.SUM,
) -> torch.Tensor:
    if _is_distributed_launch():

        def dtype_cast(dtype: torch.dtype) -> torch.dtype:
            # check if dtype is fp8, then convert it to float16 before reducing
            if dtype in [torch.float16, torch.bfloat16, torch.float32]:
                return dtype
            else:
                return torch.float16

        world_size = dist.get_world_size()
        original_dtype = input_tensor.dtype
        intermediate_dtype = dtype_cast(original_dtype)
        if metadata and metadata.input_split_sizes:
            assert dim == 0, "metadata only works with dim=0"
            input_list = list(input_tensor.split(metadata.input_split_sizes, dim=0))
            output_list = all_to_all(input_list, dim=0)
            n_tokens = metadata.ep_indx.size(dim)
            other_dims = input_tensor.shape[1:]
            output_tensor = input_tensor.new_zeros((n_tokens, ) + other_dims, dtype=intermediate_dtype)
            for i in range(world_size):
                ep_rank = i // metadata.TP
                mask = torch.any(metadata.ep_indx == ep_rank, dim=1)
                if op == dist.ReduceOp.SUM:
                    output_tensor[mask] += output_list[i].to(intermediate_dtype)
                else:
                    raise NotImplementedError(f"Reduce operation {op} is not implemented.")
            return output_tensor.to(original_dtype)
        else:
            input_list = list(input_tensor.chunk(world_size, dim=dim))
            shape = input_list[0].shape
            input_list = [x.to(intermediate_dtype) for x in input_list]
            output_tensor = input_tensor.new_empty(shape, dtype=intermediate_dtype)
            dist.reduce_scatter(output_tensor, input_list, op=op)
            return output_tensor.to(original_dtype)
    else:
        return input_tensor


def all_to_all(input_list: list[torch.Tensor], dim: int = 0) -> list[torch.Tensor]:
    if _is_distributed_launch():
        # Check if all tensors have only one dimension with different sizes
        for t in input_list:
            for d in range(t.dim()):
                if d != dim and t.size(d) != input_list[0].size(d):
                    raise ValueError("All tensors must have the same size in all dimensions except the specified one.")
        input_sizes = [t.size(dim) for t in input_list]
        input_sizes = torch.tensor(input_sizes, device=input_list[0].device).unsqueeze(0)
        input_sizes = all_gather(input_sizes, dim=0)
        output_split_sizes = input_sizes[:, dist.get_rank()].tolist()
        other_dims = list(input_list[0].shape[:dim] + input_list[0].shape[dim + 1:])
        output_list = [
            torch.empty([size] + other_dims, dtype=input_list[0].dtype, device=input_list[0].device)
            for size in output_split_sizes
        ]
        dist.all_to_all(output_list, input_list)
        return output_list
    else:
        return input_list


def _apply_parallelism(
    expt_scal: torch.Tensor,
    expt_indx: torch.Tensor,
    x: torch.Tensor,
    chunk_size: int,
    EP: int = 1,
    TP: int = 1,
):
    if EP > 1:
        # Distributed Expert Parallelism
        ep_indx = expt_indx // chunk_size

        # Partition tokens by expert parallelism rank
        expt_scal_list = []
        expt_indx_list = []
        x_list = []

        for i in range(EP):
            mask = torch.any(ep_indx == i, dim=1)
            expt_scal_masked = expt_scal[mask]
            expt_indx_masked = expt_indx[mask]
            x_masked = x[mask]

            for _ in range(TP):
                expt_scal_list.append(expt_scal_masked)
                expt_indx_list.append(expt_indx_masked)
                x_list.append(x_masked)

        # Exchange data across processes
        expt_scal_list = all_to_all(expt_scal_list, dim=0)
        expt_indx_list = all_to_all(expt_indx_list, dim=0)
        x_list = all_to_all(x_list, dim=0)

        output_split_sizes = [x.size(0) for x in expt_scal_list]
        expt_scal = torch.cat(expt_scal_list, dim=0)
        expt_indx = torch.cat(expt_indx_list, dim=0)
        x = torch.cat(x_list, dim=0)

        # Filter for local experts only
        ep_rank = dist.get_rank() // TP
        mask = (expt_indx // chunk_size) == ep_rank
        expt_indx -= ep_rank * chunk_size
        expt_scal = expt_scal.masked_fill(~mask, 0)
        expt_indx = expt_indx.masked_fill(~mask, chunk_size)
    else:
        # Distributed Data Parallelism
        ep_indx = None
        output_split_sizes = None
        x = all_gather(x, dim=0)
        expt_scal = all_gather(expt_scal, dim=0)
        expt_indx = all_gather(expt_indx, dim=0)

    return expt_scal, expt_indx, ep_indx, x, output_split_sizes


def routing_torch(x, logits, n_expts_act, sm_first=False, expt_indx=None, n_rows=None, EP=1, TP=1):
    _, n_expts_tot = logits.shape

    if n_rows:
        logits = logits[:n_rows]
    if sm_first:
        logits = torch.softmax(logits, dim=-1)

    expt_scal, expt_indx = topk_torch(logits, n_expts_act, expt_indx, has_user_provided_indx=expt_indx is not None)
    expt_indx = expt_indx.int()
    if not sm_first:
        expt_scal = torch.softmax(expt_scal, dim=-1)

    # Sort each token's selections by expert
    expt_indx, sort_indices = torch.sort(expt_indx, dim=1, stable=True)
    expt_scal = torch.gather(expt_scal, 1, sort_indices)

    chunk_size = n_expts_tot // EP

    expt_scal, expt_indx, ep_indx, x, output_split_sizes = _apply_parallelism(expt_scal, expt_indx, x, chunk_size,
                                                                              EP=EP, TP=TP)

    # Flatten topk data
    expt_scal = expt_scal.reshape(-1)
    expt_indx = expt_indx.reshape(-1).to(torch.int32)

    # Sort by expert_id for contiguous experts in matmul
    expt_indx, topk_indx = torch.sort(expt_indx, stable=True)
    gate_indx = torch.argsort(topk_indx, stable=True)

    mask = expt_indx != chunk_size
    topk_indx[~mask] = -1
    gate_indx[gate_indx >= mask.sum()] = -1
    gate_scal = expt_scal[topk_indx]
    hist = torch.histc(expt_indx[mask], bins=chunk_size, min=0, max=chunk_size - 1)

    # Pack the matmul data structures
    gather_indx = GatherIndx(src_indx=topk_indx.int(), dst_indx=gate_indx.int())
    scatter_indx = ScatterIndx(src_indx=gate_indx.int(), dst_indx=topk_indx.int())
    n_gates = mask.sum().item()
    expt_data = compute_expt_data_torch(hist, chunk_size, n_gates)

    return (
        x,
        RoutingData(gate_scal, hist, chunk_size, n_expts_act, expt_data=expt_data),
        gather_indx,
        scatter_indx,
        ReduceScatterMetadata(input_split_sizes=output_split_sizes, ep_indx=ep_indx, EP=EP, TP=TP),
    )


@triton.jit
def pack_bitmatrix(
    bitmatrix,
    expt_indx,
    n_rows,
    n_cols,
    n_expts_act,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    sentinel: tl.constexpr,
):
    """
    Packs expt_indx into a bitmatrix.
    """
    pid_m = tl.program_id(0)
    offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_k = tl.arange(0, BLOCK_SIZE_K)
    offsets = offsets_m[:, None] * n_expts_act + offsets_k[None, :]
    mask = (offsets_m < n_rows)[:, None] & (offsets_k < n_expts_act)[None, :]
    indices = tl.load(expt_indx + offsets, mask=mask, other=sentinel)
    div = indices // 32
    rem = indices % 32
    iters = tl.cdiv(sentinel, BLOCK_SIZE_K)
    for i in range(iters):
        offs = tl.arange(0, BLOCK_SIZE_K // 32) + i * (BLOCK_SIZE_K // 32)
        x = tl.where(div[:, :, None] == offs[None, None, :], (1 << rem)[:, :, None], 0)
        y = tl.reduce_or(x, axis=1)
        bitmatrix_ptrs = bitmatrix + offsets_m[:, None] * n_cols + offs[None, :]
        tl.store(bitmatrix_ptrs, y, mask=offsets_m[:, None] < n_rows)


def routing_triton(x, logits, n_expts_act, sm_first=False, expt_indx=None, n_rows=None, EP=1, TP=1):
    _, n_expts_tot = logits.shape

    if sm_first:
        logits = torch.softmax(logits, dim=-1)

    expt_scal, expt_indx, _ = topk(logits, n_expts_act, apply_softmax=not sm_first, y_indx=expt_indx, n_rows=n_rows)
    expt_indx = expt_indx.int()

    chunk_size = n_expts_tot // EP

    expt_scal, expt_indx, ep_indx, x, output_split_sizes = _apply_parallelism(expt_scal, expt_indx, x, chunk_size,
                                                                              EP=EP, TP=TP)

    # TODO: Skip all the following if `EP == 1`
    # Recover bitmatrix for local experts
    BLOCK_SIZE_M = 512
    BLOCK_SIZE_K = 32
    # The sentinel value is chunk_size + 1 instead of chunk_size to ensure the bitmatrix buffer
    # doesn't overflow. For example, cdiv(32, 32) is 1, while the 32th bit is on the second column.
    sentinel = chunk_size + 1
    n_cols = triton.cdiv(sentinel, BLOCK_SIZE_K)
    n_rows = expt_indx.size(0)
    bitmatrix = torch.zeros((n_rows, n_cols), dtype=torch.uint32, device=expt_indx.device)
    grid = (triton.cdiv(n_rows, BLOCK_SIZE_M), )

    pack_bitmatrix[grid](
        bitmatrix,
        expt_indx,
        n_rows,
        n_cols,
        n_expts_act,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        sentinel=sentinel,
    )
    bitmatrix_shape = [n_rows, triton.cdiv(chunk_size, BLOCK_SIZE_K) * 32]
    bitmatrix_shape_max = [n_rows, None]
    bitmatrix = Bitmatrix(bitmatrix, shape=bitmatrix_shape, shape_max=bitmatrix_shape_max, scratchpad=None)
    expt_scal, expt_indx, bitmatrix = prune_routing(expt_scal, expt_indx, bitmatrix, n_expts_tot, EP)
    routing_data, gather_indx, scatter_indx = routing_from_bitmatrix(bitmatrix, expt_scal, expt_indx, n_expts_tot // EP,
                                                                     n_expts_act)

    return (
        x,
        routing_data,
        gather_indx,
        scatter_indx,
        ReduceScatterMetadata(input_split_sizes=output_split_sizes, ep_indx=ep_indx, EP=EP, TP=TP),
    )


def routing(x, logits, n_expts_act, sm_first=False, expt_indx=None, n_rows=None, EP=1, TP=1,
            backend="triton") -> Tuple[RoutingData, GatherIndx, ScatterIndx, ReduceScatterMetadata]:
    if _is_distributed_launch():
        assert backend in ["torch", "triton"], "backend must be either 'torch' or 'triton'"
        if backend == "torch":
            return routing_torch(x, logits, n_expts_act, sm_first, expt_indx, n_rows, EP, TP)
        elif backend == "triton":
            return routing_triton(x, logits, n_expts_act, sm_first, expt_indx, n_rows, EP, TP)
        else:
            raise ValueError(f"Unknown backend: {backend}")
    else:
        return x, *triton_kernels.routing.routing(logits, n_expts_act, sm_first, expt_indx, EP, n_rows), None


# The following dummy methods simulate the behavior of distributed operations
# in a non-distributed environment for testing purposes.
# Assuming each rank has the same data for simplicity.


def dummy_all_gather(out, x):
    out[0].copy_(x)
    out[1].copy_(x)


def dummy_all_to_all(output_list, input_list):
    output_list[0].copy_(input_list[0])
    output_list[1].copy_(input_list[0])


def dummy_reduce_scatter(out, x_list, op):
    out.copy_(x_list[0] * 2)


def test_all_gather_non_distributed(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    x = torch.randn(4, 5)
    result = all_gather(x, dim=0)
    torch.testing.assert_close(result, x)


@pytest.mark.parametrize("dim", [0, 1])
def test_all_gather_distributed(monkeypatch, dim):
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "all_gather", dummy_all_gather)

    x = torch.randn(4, 4)
    result = all_gather(x, dim=dim)
    expected = torch.cat([x, x], dim=dim)
    torch.testing.assert_close(result, expected)


def test_reduce_scatter_non_distributed(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    x = torch.randn(4, 6)
    result = reduce_scatter(x, dim=0)
    torch.testing.assert_close(result, x)


def test_reduce_scatter_distributed(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "reduce_scatter", dummy_reduce_scatter)

    x = torch.randn(4, 6)
    expected = x.chunk(2, dim=0)[0] * 2

    result = reduce_scatter(x, dim=0)
    torch.testing.assert_close(result, expected)


def test_reduce_scatter_distributed_with_metadata(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dist, "all_to_all", dummy_all_to_all)
    monkeypatch.setattr(dist, "all_gather", dummy_all_gather)

    input_split_sizes = [1, 1]
    ep_indx = torch.tensor([[0], [1]])
    metadata = ReduceScatterMetadata(input_split_sizes=input_split_sizes, ep_indx=ep_indx, EP=2)
    # Assume the current ep rank is 0.
    # [1, 2] comes from rank 0
    # [3, 4] comes from rank 1.
    x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)

    result = reduce_scatter(x, metadata=metadata, dim=0)
    torch.testing.assert_close(result, torch.tensor([[1, 2], [1, 2]], dtype=torch.float32))


def test_routing_distributed_EP(monkeypatch):
    # Test distributed routing with EP=1 (token_mask should be None)
    monkeypatch.setenv("WORLD_SIZE", "2")
    # Set environment for local rank and distributed group
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dist, "all_gather", dummy_all_gather)
    monkeypatch.setattr(dist, "all_to_all", dummy_all_to_all)

    # NOTE: must set `device="cuda"` since `routing` expects CUDA tensors.
    logits = torch.tensor([[0.1, 0.2, 0.4, 0.3], [0.5, 0.4, 0.3, 0.1]], device="cuda", dtype=torch.float16)
    x = torch.randn_like(logits, device="cuda", dtype=torch.float16)
    n_expts_act = 2
    EP = 2
    expt_indx = torch.tensor([[0, 1], [0, 1]], device="cuda").reshape(-1)
    topk_indx = torch.argsort(expt_indx, stable=True)
    gate_indx = torch.argsort(topk_indx, stable=True)
    _, rdata, gather_indx, scatter_indx, metadata = routing(x, logits, n_expts_act, EP=EP)
    assert torch.equal(gather_indx.src_indx, topk_indx.int())
    assert torch.equal(gather_indx.dst_indx, gate_indx.int())
    assert torch.equal(scatter_indx.src_indx, gate_indx.int())
    assert torch.equal(scatter_indx.dst_indx, topk_indx.int())


def test_all_to_all(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dist, "all_to_all", dummy_all_to_all)
    monkeypatch.setattr(dist, "all_gather", dummy_all_gather)

    input_list = [torch.tensor([1, 2], dtype=torch.float32), torch.tensor([3, 4], dtype=torch.float32)]
    output_list = all_to_all(input_list)
    assert torch.equal(output_list[0], torch.tensor([1, 2], dtype=torch.float32))
    assert torch.equal(output_list[1], torch.tensor([1, 2], dtype=torch.float32))
    assert len(output_list) == 2


def test_pack_bitmatrix():
    # Test parameters
    n_rows, n_expts_act = 4, 3
    sentinel = 63  # We have experts 0-62, and 63 is a dummy value

    expt_indx = torch.tensor([[0, 33, 63], [31, 32, 33], [5, 10, 15], [0, 62, 63]], dtype=torch.int32, device="cuda")
    n_cols = triton.cdiv(sentinel, 32)
    bitmatrix = torch.zeros((n_rows, n_cols), dtype=torch.uint32, device="cuda")

    BLOCK_SIZE_M = 128
    BLOCK_SIZE_K = 32
    grid = (triton.cdiv(n_rows, BLOCK_SIZE_M), )

    pack_bitmatrix[grid](
        bitmatrix,
        expt_indx,
        n_rows,
        n_cols,
        n_expts_act,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        sentinel=sentinel,
    )
    # Prune the bitmatrix to remove dummy values
    _routing_clear_bitmatrix[(n_rows, )](
        bitmatrix,
        bitmatrix.stride(0),
        bitmatrix.stride(1),
        bitmatrix.shape[1],
        sentinel,
        BLOCK_N=128,
    )

    # Old pytorch version do not have "bitwise_and_cpu" not implemented for 'UInt32'
    bitmatrix = bitmatrix.cpu().numpy()

    # Verify bit packing
    def is_bit_set(row, expert_id):
        word_idx, bit_idx = expert_id // 32, expert_id % 32
        return (bitmatrix[row, word_idx] & (1 << bit_idx)) != 0

    # Check specific cases
    assert is_bit_set(0, 0) and is_bit_set(0, 33) and not is_bit_set(0, 63)  # Token 0
    assert is_bit_set(1, 31) and is_bit_set(1, 32) and is_bit_set(1, 33)  # Token 1
    assert is_bit_set(2, 5) and is_bit_set(2, 10) and is_bit_set(2, 15)  # Token 2
    assert is_bit_set(3, 0) and not is_bit_set(3, 63) and is_bit_set(3, 62)  # Token 3


def gather_ep(rank, world_size, param, TP, EP):
    gathered = None
    group = dist.new_group(list(range(0, world_size, TP)))
    dist.barrier()
    if rank % TP == 0:
        gathered = []
        if rank == 0:
            gathered = [torch.zeros_like(param) for _ in range(EP)]
        dist.gather(param, gathered, dst=0, group=group)
        if rank == 0:
            gathered = torch.cat(gathered, dim=0)
    return gathered


def gather_full(rank, world_size, param, TP, EP, concat_dim_inside, concat_dim_outside):
    gathered = []
    if rank == 0:
        gathered = [torch.zeros_like(param) for _ in range(world_size)]
    dist.gather(param, gathered, dst=0)
    if rank == 0:
        rows = [torch.cat(gathered[e * TP:(e + 1) * TP], dim=concat_dim_inside) for e in range(EP)]
        return torch.cat(rows, dim=concat_dim_outside)
    return None


# We compare the distributed and single-GPU versions of the model to verify correctness.
def distributed_run(rank, world_size, batch, dim1, dim2, n_expts_tot, n_expts_act, x_dtype, w_dtype, TP, EP):
    # init
    dev = f"cuda:{rank}"
    # Though we specify backend="nccl", it will be mapped to "rccl" for HIP.
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size, device_id=torch.device(dev))
    torch.cuda.set_device(rank)

    # weights & biases
    wg = torch.randn((dim1, n_expts_tot), device=dev)
    dist.broadcast(wg, src=0)

    bg = torch.randn((n_expts_tot, ), device=dev)
    dist.broadcast(bg, src=0)

    b2 = torch.randn((n_expts_tot // EP, dim1), device=dev)
    ep_rank = rank // TP
    groups = [dist.new_group(list(range(ep * TP, (ep + 1) * TP))) for ep in range(EP)]
    dist.barrier()
    group = groups[ep_rank]
    dist.broadcast(b2, src=ep_rank * TP, group=group)

    w1 = torch.randn((n_expts_tot // EP, dim1, dim2 // TP), device=dev)
    w2 = torch.randn((n_expts_tot // EP, dim2 // TP // 2, dim1), device=dev)
    b1 = torch.randn((n_expts_tot // EP, dim2 // TP), device=dev)

    w1_full = gather_full(rank, world_size, w1, TP, EP, concat_dim_inside=2, concat_dim_outside=0)
    w2_full = gather_full(rank, world_size, w2, TP, EP, concat_dim_inside=1, concat_dim_outside=0)
    b1_full = gather_full(rank, world_size, b1, TP, EP, concat_dim_inside=1, concat_dim_outside=0)
    b2_full = gather_ep(rank, world_size, b2, TP, EP)

    # quantization
    opt1 = dict()
    opt2 = dict()
    if w_dtype == "mx4" and not is_hip():
        num_warps = 4 if batch <= 512 else 8
        value_layout, value_layout_opts = layout.make_default_matmul_mxfp4_w_layout(mx_axis=1)
        scale_layout, scale_layout_opts = layout.make_default_matmul_mxfp4_w_scale_layout(
            mx_axis=1, num_warps=num_warps)
        opt1 = {
            "value_layout": value_layout,
            "value_layout_opts": value_layout_opts,
            "scale_layout": scale_layout,
            "scale_layout_opts": scale_layout_opts,
        }
        opt2 = deepcopy(opt1)
    wg, wg_flex, wg_scale = quantize_weight(wg, "bf16")
    w1, w1_flex, w1_scale = quantize_weight(w1, w_dtype, **opt1)
    w2, w2_flex, w2_scale = quantize_weight(w2, w_dtype, **opt2)
    if rank == 0:
        w1_full, w1_flex_full, w1_scale_full = quantize_weight(w1_full, w_dtype, **opt1)
        w2_full, w2_flex_full, w2_scale_full = quantize_weight(w2_full, w_dtype, **opt2)
    else:
        w1_full = w2_full = w1_flex_full = w2_flex_full = w1_scale_full = w2_scale_full = None

    # precision configs
    pcg = PrecisionConfig(flex_ctx=FlexCtx(rhs_data=wg_flex), weight_scale=wg_scale)
    act = FusedActivation(FnSpecs("swiglu", triton_kernels.swiglu.swiglu_fn, ("alpha", "limit")), (1.0, 1.0), 2)
    pc1 = PrecisionConfig(flex_ctx=FlexCtx(rhs_data=w1_flex), weight_scale=w1_scale)
    pc2 = PrecisionConfig(flex_ctx=FlexCtx(rhs_data=w2_flex), weight_scale=w2_scale)
    if rank == 0:
        pc1_full = PrecisionConfig(flex_ctx=FlexCtx(rhs_data=w1_flex_full), weight_scale=w1_scale_full)
        pc2_full = PrecisionConfig(flex_ctx=FlexCtx(rhs_data=w2_flex_full), weight_scale=w2_scale_full)
    else:
        pc1_full = pc2_full = None

    # inputs
    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp8": torch.float8_e4m3fnuz if get_cdna_version() == 3 else torch.float8_e4m3fn,
    }
    xd = torch.randn((batch // world_size, dim1), device=dev).to(dtype_map[x_dtype])
    x0 = all_gather(xd, dim=0)

    # single-GPU pass
    def single(x):
        xg = x.to(wg.dtype if n_expts_tot > 1 else x.dtype)
        if n_expts_tot > 1:
            logits = matmul_ogs(xg, wg, bg, precision_config=pcg)
            rdata, gi, si = triton_kernels.routing.routing(logits, n_expts_act)
        else:
            rdata = gi = si = None
        x = matmul_ogs(x, w1_full, b1_full, rdata, gather_indx=gi, precision_config=pc1_full, fused_activation=act)
        return matmul_ogs(x, w2_full, b2_full, rdata, scatter_indx=si, precision_config=pc2_full)

    # distributed pass
    def distributed(x):
        xg = x.to(wg.dtype if n_expts_tot > 1 else x.dtype)
        if n_expts_tot > 1:  # sparse
            logits = matmul_ogs(xg, wg, bg, precision_config=pcg)
            x, rdata, gi, si, metadata = routing(x, logits, n_expts_act, EP=EP, TP=TP)
        else:  # dense
            x = all_gather(x, dim=0)
            rdata = gi = si = metadata = None
        x = matmul_ogs(x, w1, b1, rdata, gather_indx=gi, precision_config=pc1, fused_activation=act)
        x = matmul_ogs(x, w2, b2 if rank % TP == 0 else None, rdata, scatter_indx=si, precision_config=pc2)
        x = reduce_scatter(x, metadata=metadata, dim=0)
        # gather the result from all GPUs, just for verification
        return all_gather(x, dim=0)

    distributed_result = distributed(xd)
    if rank == 0:
        single_result = single(x0)
        torch.testing.assert_close(distributed_result.to(torch.float16), single_result.to(torch.float16), rtol=1e-2,
                                   atol=1.0, equal_nan=True)

    dist.barrier()
    dist.destroy_process_group()


has_native_mx4 = torch.cuda.get_device_capability(0)[0] >= 10 or get_cdna_version() == 4


@pytest.mark.parametrize(
    "batch, dim1, dim2, n_expts_tot, n_expts_act, x_dtype, w_dtype, TP, EP",
    [
        # dense cases - test parallelism
        (1024, 1024, 1024, 1, 1, "bf16", "bf16", 1, 1),
        (1024, 1024, 1024, 1, 1, "bf16", "bf16", 4, 1),
    ] +
    # dense cases - test precision
    [(1024, 1024, 1024, 1, 1, "fp8", "fp8", 1, 1), (1024, 1024, 1024, 1, 1, "fp8", "fp8", 4, 1)]
    # moe cases - test parallelism
    + [
        (1024, 1024, 1024, 128, 2, "bf16", "bf16", 1, 1),
        (1024, 1024, 1024, 128, 2, "bf16", "bf16", 4, 1),
        (1024, 1024, 1024, 128, 2, "bf16", "bf16", 1, 4),
        (1024, 1024, 1024, 128, 2, "bf16", "bf16", 2, 2),
    ] +
    # moe cases - test precision
    ([
        (1024, 1024, 1024, 128, 2, "fp8", "mx4", 1, 1),
        (1024, 1024, 1024, 128, 2, "fp8", "mx4", 4, 1),
        (1024, 1024, 1024, 128, 2, "fp8", "mx4", 1, 4),
        (1024, 1024, 1024, 128, 2, "fp8", "mx4", 2, 2),
    ] if has_native_mx4 else [
        (1024, 1024, 1024, 128, 2, "bf16", "mx4", 1, 1),
        (1024, 1024, 1024, 128, 2, "bf16", "mx4", 4, 1),
        (1024, 1024, 1024, 128, 2, "bf16", "mx4", 1, 4),
        (1024, 1024, 1024, 128, 2, "bf16", "mx4", 2, 2),
    ]),
)
def test_mlp_mp(batch, dim1, dim2, n_expts_tot, n_expts_act, x_dtype, w_dtype, TP, EP, monkeypatch):
    parallelism = TP * EP
    if torch.cuda.device_count() < parallelism:
        pytest.skip(f"Test requires at least {parallelism} GPUs.")
    if is_cuda() and not cuda_capability_geq(9, 0):
        pytest.skip("Test requires CUDA compute capability >= 9.0.")
    if is_hip() and get_cdna_version() == 4 and EP > 1:
        pytest.skip("[TODO] Unknown issue with CDNA 4 and EP > 1")
    if TP > 1 and x_dtype == "fp8":
        pytest.skip("[TODO] Testing FP8 is not supported for TP > 1.")

    monkeypatch.setenv("WORLD_SIZE", f"{parallelism}")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "12355")
    mp.spawn(
        distributed_run,
        args=(parallelism, batch, dim1, dim2, n_expts_tot, n_expts_act, x_dtype, w_dtype, TP, EP),
        nprocs=parallelism,
        join=True,
    )
