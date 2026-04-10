# Reduce-Scatter Runtime Model

## Confirmed Facts

### Algorithm: Tensor-Parallel Reduce-Scatter

In column-parallel linear layers (e.g., Megatron-LM style tensor parallelism):

- Input X: shape (batch*seq, H), same on all GPUs
- Weight W: sharded along output dimension: W_r = W[:, r*H_out//W : (r+1)*H_out//W]
- Each GPU computes: P_r = X @ W_r, shape (batch*seq, H_out//W)
- No reduction needed across GPUs for this shape (each GPU's output is independent)

For row-parallel linear with reduce-scatter:
- Input X: sharded along hidden dim: X_r = X[:, r*H//W : (r+1)*H//W]
- Weight W: shared W, shape (H//W, H_out)
- Each GPU computes: P_r = X_r @ W, shape (batch*seq, H_out) — partial sum
- Reduce-scatter: GPU r ends up with rows [r*BS//W : (r+1)*BS//W] of sum_r P_r

### Our Kernel's Runtime Execution Model

**Two-kernel + host-barrier design:**

```
GPU 0                          GPU 1 (etc.)
┌─────────────────────────┐   ┌─────────────────────────┐
│ gemm_scatter_kernel      │   │ gemm_scatter_kernel      │
│  ├─ GEMM inner loop      │   │  ├─ GEMM inner loop      │
│  │   (pipelined loads)   │   │  │   (pipelined loads)   │
│  └─ scatter store        │   │  └─ scatter store        │
│      → peer buf[1]       │   │      → peer buf[0]       │
└─────────────────────────┘   └─────────────────────────┘
         ↕ NVLink/UVA peer writes
┌─────────────────────────────────────────────────────────┐
│  hdl.barrier(channel=0)  ← host-side sync (CPU barrier  │
│                              + CUDA stream sync)         │
└─────────────────────────────────────────────────────────┘
┌─────────────────────────┐   ┌─────────────────────────┐
│ reduce_kernel            │   │ reduce_kernel            │
│  ├─ read P_0 from buf    │   │  ├─ read P_0 from buf    │
│  ├─ read P_1 from buf    │   │  ├─ read P_1 from buf    │
│  └─ store output shard   │   │  └─ store output shard   │
└─────────────────────────┘   └─────────────────────────┘
```

### Why the Host Barrier Creates an Epoch Boundary

The `hdl.barrier(channel=0)` call:
1. `dist.barrier()` — synchronizes all CPU process threads
2. `torch.cuda.synchronize()` — waits for all GPU streams to complete
3. Signals all other ranks that CUDA work is done

After the barrier, when `reduce_kernel` is launched, all scatter writes are guaranteed visible by CUDA memory model. The GPU is idle at the barrier point.

**Implication for overlap:** There is no possibility of overlapping the scatter writes (end of `gemm_scatter_kernel`) with the reduction reads (start of `reduce_kernel`) at the kernel level. The barrier enforces a happens-before relationship that the compiler cannot collapse.

### Potential for Intra-Kernel Overlap (Why It's Hard)

For true fused GEMM + reduce-scatter overlap, both phases would need to be in a single kernel with GPU-side synchronization:

```
CTA (pid_m=0) computes GEMM tiles for rows 0..127
  → atomically writes to peer_buf for these rows
  → signals a semaphore: "rows 0..127 ready"

Meanwhile on peer GPU:
  CTA for reduce of rows 0..127 spins on the semaphore
  → reads and accumulates when signaled

CTA (pid_m=1) computes rows 128..255
  → ... (overlap with peer reduce of rows 0..127)
```

This requires:
1. GPU-side semaphores (not available in Triton without `tl.inline_asm_elementwise`)
2. Ordering guarantees for NVLink writes (CUDA MemOps flags)
3. A cross-GPU memory fence (`__threadfence_system()` in CUDA)
4. Multi-CTA blocks (clusters) or separate kernel launches with dependency tracking

None of these are currently modeled in Triton's IR or supported by the software pipeline pass.

## Observed Behavior

- Single-GPU simulation (world_size=1): the "scatter" is a local store; the "reduce" reads back the same data. Functionally equivalent to a regular GEMM.
- The host barrier is the fundamental obstacle to pipelining: it is a synchronization point that the compiler cannot see through.

## Hypotheses

- NCCL's pipelined all-reduce uses a ring algorithm that achieves ~50% overlap by interleaving communication chunks with the next chunk's computation. A similar approach in Triton would require split-K with streaming partial stores — fundamentally changing the algorithm.
- Warp specialization (separate load/compute/communication warp groups) could provide intra-kernel overlap on a single GPU, but cross-GPU synchronization still requires the barrier.

## Open Questions

- Is `__threadfence_system()` accessible via Triton's `tl.inline_asm_elementwise`?
- Could `mbarrier` (available via Gluon) model cross-GPU synchronization if combined with NVLink CTAs in a CGA (Cooperative Thread Array Cluster)?
- Does CUDA cooperative groups provide cross-SM barriers sufficient for single-kernel reduce-scatter?

## Next Actions

- Determine whether Hopper's CGA (SM90) can coordinate across GPUs (answer: no, CGA is intra-device)
- Document the split-K streaming approach as the theoretical basis for true overlap
