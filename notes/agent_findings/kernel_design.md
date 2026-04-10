# Kernel Design Notes

## Confirmed Facts

### Problem Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| M | 1024 | Rows of input A / output |
| N | 1024 | Columns of B / output |
| K | 1024 | Reduction dimension |
| BLOCK_M | 128 | Standard tile size for fp16 GEMM |
| BLOCK_N | 128 | Standard tile size for fp16 GEMM |
| BLOCK_K | 64 | Standard tile size for fp16 GEMM |
| NUM_STAGES | 3 | Pipeline depth (2 in-flight loads + 1 executing) |
| dtype | float16 | Common for DL workloads; uses tensor cores |

### Two-Kernel Design

The design separates two semantically distinct phases:

**Kernel 1: `gemm_scatter_kernel`**
- Computes one (BLOCK_M, BLOCK_N) output tile via the inner k-loop
- Writes the completed tile to the destination rank's symmetric memory buffer
- The `num_stages=NUM_STAGES` annotation on `tl.range` is the pipeline request

**Kernel 2: `reduce_kernel`**  
- Reads WORLD_SIZE partial tiles from local symm mem buffer
- Sums them using `tl.static_range(WORLD_SIZE)` (unrolled at compile time)
- Stores final reduced tile to output tensor

**Host barrier** between kernels: `hdl.barrier(channel=0)`

### Why Two Kernels Instead of One

A single fused kernel would require intra-kernel cross-GPU synchronization:
- After each CTA finishes its GEMM tile, it would need to signal to the peer GPU
- The peer GPU's CTA would need to spin-wait until all contributing ranks signal
- Triton provides no GPU-side inter-GPU semaphore primitive

The two-kernel design sacrifices potential overlap for correctness guarantees.
The host barrier provides the required ordering.

### Scatter Mechanism

For world_size=1 (single-GPU simulation and IR capture):
- The "peer" buffer is the local output buffer
- `scatter_offset = RANK * M_SHARD * N = 0` (rank 0, slot 0)
- `tl.store` writes to the same device — no actual NVLink/peer access

For world_size=N (multi-GPU):
- The calling code computes which destination rank owns each row range
- `gemm_scatter_kernel` is launched once per destination rank
  (In a production implementation, all WORLD_SIZE peer pointers would be
  passed as a variadic argument and selected via `tl.static_range`, following
  the pattern in `python/triton_kernels/triton_kernels/distributed.py`)
- `out_ptr` points to the appropriate peer's symm mem buffer
- `scatter_offset = RANK * M_SHARD * N` encodes which slot this rank writes to

### Buffer Layout

The symmetric memory buffer has layout `[WORLD_SIZE, M_SHARD, N]` (flat):
```
offset = src_rank * M_SHARD * N + local_row * N + col
```

GPU `r` (as destination) allocates one buffer of size `WORLD_SIZE * M_SHARD * N`.
GPU `s` (as source) writes to slot `s * M_SHARD * N + ...` in GPU `r`'s buffer.
After all sources have written, GPU `r`'s `reduce_kernel` sums slots 0..W-1.

### IR Analysis Target

The key IR structures to observe:

1. `scf.for %k = 0 to K//BLOCK_K step 1 {tt.num_stages = 3}` — the pipeline request
2. Inside the loop: `tt.load %a_ptrs`, `tt.load %b_ptrs`, `tt.dot %a, %b, %acc`
3. Outside the loop: `tt.store %out_ptrs, %result`

After pipelining:
- Items 2 (loads) → `ttg.async_copy_global_to_local` + commit/wait groups
- Items 2 (dot) → unchanged, but now overlapped with loads of next iteration
- Item 3 → **unchanged** (synchronous store to peer buffer)

This contrast is the central finding of the IR analysis.

## Observed Behavior

- `tl.range(0, N, num_stages=K)` compiles to `scf.for` with `{tt.num_stages = K}` attribute (verified from TTIR dumps in the test suite, e.g., `test/TritonGPU/loop-pipeline-cuda.mlir`)
- The `tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)` generates an initial zero tensor used as the `acc` argument to the first `tt.dot` — this is a loop-carried value (iter_arg) in the MLIR IR

## Hypotheses

- With Hopper TMA, the `tt.load` ops could be replaced by `tt.DescriptorLoadOp` and the pipeliner would use `mbarrier` instead of `cp.async`. The scatter store analysis would be identical.
- With warp specialization, load warps could be separated from compute warps. However, the scatter store would still need to wait for the compute warp group's accumulator — the same dependency holds.

## Open Questions

- Should `BLOCK_K` be 32 or 64 for best pipelining efficiency on the test GPU?
- Does passing `other=0.0` to `tl.load` prevent async conversion (per `AssignLatencies.cpp` — loads with non-zero "other" cannot become async copies)?

## Next Actions

- Verify: if `other=0.0` prevents async, change to boundary-tile masking via separate loop split
- Check BLOCK_K sizing against the hardware: 64 × 128 × 2 bytes = 16KB per double buffer = borderline for shared memory
