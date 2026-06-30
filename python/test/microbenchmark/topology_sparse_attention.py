from __future__ import annotations

import argparse
import math
import statistics

import torch
import torch.nn.functional as F

import triton
import triton.language as tl


def zero_dim_persistence_salience(centroids: torch.Tensor) -> torch.Tensor:
    if centroids.ndim != 2:
        raise ValueError("centroids must have shape [num_blocks, dim]")
    num_blocks = centroids.shape[0]
    if num_blocks == 0:
        raise ValueError("at least one block is required")
    if num_blocks == 1:
        return torch.ones((1, ), dtype=centroids.dtype, device=centroids.device)

    cpu_centroids = centroids.detach().to("cpu", torch.float64)
    distances = torch.cdist(cpu_centroids, cpu_centroids)
    edges = []
    for i in range(num_blocks):
        for j in range(i + 1, num_blocks):
            edges.append((float(distances[i, j]), i, j))
    edges.sort(key=lambda item: item[0])

    parent = list(range(num_blocks))
    members = {i: {i} for i in range(num_blocks)}
    salience = torch.zeros((num_blocks, ), dtype=torch.float64)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for distance, left, right in edges:
        root_left = find(left)
        root_right = find(right)
        if root_left == root_right:
            continue
        if len(members[root_left]) > len(members[root_right]):
            root_left, root_right = root_right, root_left

        for block in members[root_left]:
            salience[block] = distance
        parent[root_left] = root_right
        members[root_right].update(members[root_left])
        del members[root_left]

    return salience.to(device=centroids.device, dtype=centroids.dtype)


def build_topology_block_schedule(
    keys: torch.Tensor,
    block_size: int,
    local_radius_blocks: int,
    sink_blocks: int,
    topk_topology_blocks: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if keys.ndim != 2:
        raise ValueError("keys must have shape [seq, dim]")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if keys.shape[0] % block_size != 0:
        raise ValueError("sequence length must be divisible by block_size")

    num_blocks = keys.shape[0] // block_size
    centroids = keys.reshape(num_blocks, block_size, keys.shape[1]).mean(dim=1)
    salience = zero_dim_persistence_salience(centroids)
    topk = min(max(topk_topology_blocks, 0), num_blocks)
    topology_blocks = set(torch.topk(salience, k=topk).indices.tolist()) if topk else set()

    offsets = [0]
    indices = []
    for q_block in range(num_blocks):
        allowed = set(range(min(sink_blocks, num_blocks)))
        allowed.update(range(max(0, q_block - local_radius_blocks), q_block + 1))
        allowed.update(block for block in topology_blocks if block <= q_block)
        allowed = {block for block in allowed if block <= q_block}
        if not allowed:
            allowed.add(q_block)

        indices.extend(sorted(allowed))
        offsets.append(len(indices))

    return torch.tensor(offsets, dtype=torch.int64), torch.tensor(indices, dtype=torch.int64)


def build_dense_causal_block_schedule(num_blocks: int) -> tuple[torch.Tensor, torch.Tensor]:
    offsets = [0]
    indices = []
    for q_block in range(num_blocks):
        indices.extend(range(q_block + 1))
        offsets.append(len(indices))
    return torch.tensor(offsets, dtype=torch.int64), torch.tensor(indices, dtype=torch.int64)


def dense_masked_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    offsets: torch.Tensor,
    indices: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    num_blocks = q.shape[0] // block_size
    allow = torch.zeros((q.shape[0], k.shape[0]), dtype=torch.bool, device=q.device)
    for q_block in range(num_blocks):
        q_slice = slice(q_block * block_size, (q_block + 1) * block_size)
        for block in indices[int(offsets[q_block].item()):int(offsets[q_block + 1].item())]:
            k_slice = slice(int(block.item()) * block_size, (int(block.item()) + 1) * block_size)
            allow[q_slice, k_slice] = True

    positions = torch.arange(q.shape[0], device=q.device)
    allow &= positions[None, :] <= positions[:, None]
    logits = (q @ k.T) / math.sqrt(q.shape[1])
    logits = logits.masked_fill(~allow, float("-inf"))
    return torch.softmax(logits, dim=-1) @ v


@triton.jit
def _scheduled_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    offsets_ptr,
    indices_ptr,
    out_ptr,
    seq: tl.constexpr,
    stride_qm: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kn: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_vn: tl.constexpr,
    stride_vd: tl.constexpr,
    stride_om: tl.constexpr,
    stride_od: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    q_block = tl.program_id(0)
    offs_m = q_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q = tl.load(
        q_ptr + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
        mask=offs_m[:, None] < seq,
        other=0.0,
    )

    m_i = tl.full((BLOCK_M, ), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M, ), tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)

    schedule_ptr = tl.load(offsets_ptr + q_block)
    schedule_end = tl.load(offsets_ptr + q_block + 1)
    while schedule_ptr < schedule_end:
        k_block = tl.load(indices_ptr + schedule_ptr)
        k_pos = k_block * BLOCK_N + offs_n

        k_tile = tl.load(
            k_ptr + k_pos[:, None] * stride_kn + offs_d[None, :] * stride_kd,
            mask=k_pos[:, None] < seq,
            other=0.0,
        )
        scores = tl.dot(q, tl.trans(k_tile)) * scale
        valid = (offs_m[:, None] < seq) & (k_pos[None, :] < seq) & (k_pos[None, :] <= offs_m[:, None])
        scores = tl.where(valid, scores, -float("inf"))

        m_ij = tl.maximum(m_i, tl.max(scores, 1))
        p = tl.exp(scores - m_ij[:, None])
        alpha = tl.exp(m_i - m_ij)

        v_tile = tl.load(
            v_ptr + k_pos[:, None] * stride_vn + offs_d[None, :] * stride_vd,
            mask=k_pos[:, None] < seq,
            other=0.0,
        )
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float32), v_tile.to(tl.float32))
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_ij
        schedule_ptr += 1

    out = acc / l_i[:, None]
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
        out,
        mask=offs_m[:, None] < seq,
    )


def triton_scheduled_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    offsets: torch.Tensor,
    indices: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    offsets = offsets.contiguous()
    indices = indices.contiguous()
    seq, head_dim = q.shape
    out = torch.empty((seq, head_dim), device=q.device, dtype=torch.float32)
    _scheduled_attention_kernel[(seq // block_size, )](
        q,
        k,
        v,
        offsets,
        indices,
        out,
        seq,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        out.stride(0),
        out.stride(1),
        1.0 / math.sqrt(head_dim),
        BLOCK_M=block_size,
        BLOCK_N=block_size,
        HEAD_DIM=head_dim,
        num_warps=4,
    )
    return out


def timed_cuda(fn, *, rounds: int) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def run_case(seq: int, dim: int, block_size: int, rounds: int) -> dict[str, float]:
    torch.manual_seed(20260629 + seq)
    q = torch.randn((seq, dim), device="cuda", dtype=torch.float16)
    k = torch.randn((seq, dim), device="cuda", dtype=torch.float16)
    v = torch.randn((seq, dim), device="cuda", dtype=torch.float16)

    offsets, indices = build_topology_block_schedule(
        k.float(),
        block_size=block_size,
        local_radius_blocks=1,
        sink_blocks=1,
        topk_topology_blocks=max(1, seq // block_size // 8),
    )
    offsets_cuda = offsets.to(device="cuda")
    indices_cuda = indices.to(device="cuda")
    dense_offsets, dense_indices = build_dense_causal_block_schedule(seq // block_size)
    dense_offsets_cuda = dense_offsets.to(device="cuda")
    dense_indices_cuda = dense_indices.to(device="cuda")

    expected = dense_masked_attention(q.float(), k.float(), v.float(), offsets, indices, block_size)
    actual = triton_scheduled_attention(q, k, v, offsets_cuda, indices_cuda, block_size)
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)
    dense_csr_expected = dense_masked_attention(q.float(), k.float(), v.float(), dense_offsets, dense_indices,
                                                block_size)
    dense_csr_actual = triton_scheduled_attention(q, k, v, dense_offsets_cuda, dense_indices_cuda, block_size)
    torch.testing.assert_close(dense_csr_actual, dense_csr_expected, rtol=3e-2, atol=3e-2)

    dense_causal_blocks = (seq // block_size) * (seq // block_size + 1) // 2
    scheduled_blocks = indices.numel()
    dense_masked_ms = timed_cuda(
        lambda: dense_masked_attention(q.float(), k.float(), v.float(), offsets, indices, block_size),
        rounds=rounds,
    )
    sdpa_full_causal_ms = timed_cuda(
        lambda: F.scaled_dot_product_attention(
            q[None, None, :, :],
            k[None, None, :, :],
            v[None, None, :, :],
            is_causal=True,
        ),
        rounds=rounds,
    )
    triton_scheduled_ms = timed_cuda(
        lambda: triton_scheduled_attention(q, k, v, offsets_cuda, indices_cuda, block_size),
        rounds=rounds,
    )
    triton_dense_csr_ms = timed_cuda(
        lambda: triton_scheduled_attention(q, k, v, dense_offsets_cuda, dense_indices_cuda, block_size),
        rounds=rounds,
    )

    return {
        "seq": seq,
        "scheduled_blocks": scheduled_blocks,
        "dense_csr_blocks": dense_indices.numel(),
        "dense_causal_blocks": dense_causal_blocks,
        "block_reduction": 1.0 - scheduled_blocks / dense_causal_blocks,
        "dense_masked_ms": dense_masked_ms,
        "sdpa_full_causal_ms": sdpa_full_causal_ms,
        "triton_scheduled_ms": triton_scheduled_ms,
        "triton_dense_csr_ms": triton_dense_csr_ms,
        "triton_vs_dense_masked": dense_masked_ms / triton_scheduled_ms,
        "triton_vs_sdpa_full_causal": sdpa_full_causal_ms / triton_scheduled_ms,
        "triton_sparse_vs_triton_dense_csr": triton_dense_csr_ms / triton_scheduled_ms,
        "max_abs_error": (actual - expected).abs().max().item(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark topology-derived CSR scheduled attention.")
    parser.add_argument("--seq", type=int, nargs="*", default=[1024, 2048, 4096])
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=10)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Triton: {triton.__version__}")
    print()
    print("| seq | scheduled / dense blocks | block reduction | dense masked ms | full causal SDPA ms | "
          "Triton dense CSR ms | Triton scheduled ms | Triton sparse vs dense CSR | Triton vs SDPA | "
          "max abs error |")
    print("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for seq in args.seq:
        result = run_case(seq=seq, dim=args.dim, block_size=args.block_size, rounds=args.rounds)
        print("| {seq} | {scheduled_blocks} / {dense_csr_blocks} | {block_reduction:.1%} | "
              "{dense_masked_ms:.3f} | {sdpa_full_causal_ms:.3f} | {triton_dense_csr_ms:.3f} | "
              "{triton_scheduled_ms:.3f} | {triton_sparse_vs_triton_dense_csr:.2f}x | "
              "{triton_vs_sdpa_full_causal:.2f}x | {max_abs_error:.4f} |".format(**result))


if __name__ == "__main__":
    main()
