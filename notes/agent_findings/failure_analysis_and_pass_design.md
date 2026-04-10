# Failure Analysis: Why GEMM-Reduce-Scatter Overlap is Not Achievable with Current Passes

## Confirmed Facts

### Root Cause 1: Accumulator Dependency Chain

The GEMM accumulator `acc` is a **loop-carried value** that accumulates across all K
iterations. Its correct final value is only available after the `scf.for` completes.

```
%acc_0 = tl.zeros(...)
# K/BLOCK_K iterations:
%acc_1 = tt.dot(%a_0, %b_0, %acc_0)
%acc_2 = tt.dot(%a_1, %b_1, %acc_1)
...
%acc_final = tt.dot(%a_{K/BLOCK_K-1}, %b_{K/BLOCK_K-1}, %acc_{K/BLOCK_K-1})
# Only here is acc_final valid:
%result = tt.fp_to_fp(%acc_final)
tt.store(%peer_ptr, %result)   ← depends on %acc_final
```

The `tt.store` has a data dependency on `%acc_final`, which has a dependency on
*every* `tt.dot` in the loop. This is not a latency that can be hidden — it is a
fundamental ordering constraint. No amount of pipelining can issue the scatter store
before the full K reduction completes.

### Root Cause 2: No Async Scatter Op in Triton's IR

The scatter write is `tt.store %peer_ptr, %result`. In Triton's IR:
- `tt.store` is a synchronous operation with no async variant for global memory
- `tt.DescriptorStoreOp` exists (TMA async store), but only for local→global transfers
  within the same GPU; it does not model cross-GPU peer writes
- The pipeliner (`AssignLatencies.cpp`) never assigns `tt.latency` to stores
- `TMAStoresPipeline.cpp` handles `tt.DescriptorStoreOp` but requires a specific IR
  structure (descriptor + local memory allocation) that peer writes don't have

Without an async scatter op, the pipeliner has no "hook" to overlap the scatter with
any subsequent operation.

### Root Cause 3: Host-Side Barrier Creates an Epoch Boundary

The `hdl.barrier(channel=0)` call is a host-side Python function that:
1. Synchronizes all CPU processes (PyTorch distributed barrier)
2. Calls `torch.cuda.synchronize()` to drain all CUDA streams

This creates a **compiler-invisible epoch boundary** between `gemm_scatter_kernel`
and `reduce_kernel`. The two kernels are compiled independently; neither has any
IR-level knowledge of the other's existence.

There is no MLIR pass that can reason across kernel launch boundaries, because:
- The Triton compiler compiles one kernel at a time
- Inter-kernel scheduling is not a supported abstraction
- The barrier semantics are encoded in Python runtime state, not IR

### Root Cause 4: No Cross-GPU Synchronization Primitive in Triton

For a single fused kernel to overlap GEMM and reduce-scatter, it would need a
GPU-side semaphore that allows one GPU's CTA to signal another GPU's CTA. Triton
provides no such primitive. Available synchronization mechanisms:

| Mechanism | Scope | Available in Triton? |
|-----------|-------|---------------------|
| `tl.debug_barrier()` | Intra-CTA | Yes |
| `mbarrier` (Gluon) | Intra-SM / intra-cluster | Yes (Gluon only) |
| CUDA CGA cluster barrier | Intra-device (same GPU) | No |
| NVLink semaphore (CUDA cooperative groups) | Cross-GPU | No |
| `__threadfence_system()` | Cross-GPU visibility | No |

Even if Triton exposed NVLink semaphores, the programming model would be extremely
complex: CTA (pid_m, pid_n) on GPU 0 would need to atomically signal GPU 1 that
its tile is ready, while GPU 1's CTA for the same tile range is spinning on that
signal. This is effectively a distributed streaming dataflow model that goes far
beyond current Triton's programming model.

## What IS Achievable with Current Passes

### What the Pipeliner Actually Does

The software pipeline pass **successfully** overlaps:
1. `tt.load(A tile)` with `tt.dot` computation of the previous iteration
2. `tt.load(B tile)` with `tt.dot` computation of the previous iteration

This gives up to `(NUM_STAGES - 1) × BLOCK_K` elements of "look-ahead" — the loads
for iteration k+2 are issued while iteration k's dot is running.

For the GEMM inner loop on a typical GPU with 100-300 cycle global memory latency
and 16-32 cycle tensor core latency per BLOCK_K tile, NUM_STAGES=3 is sufficient to
fully hide load latency. This is the pipeliner's intended and correct use case.

### Achievable Overlap via Spatial Concurrency (Not Temporal Pipelining)

Because the GEMM output tiles for different (pid_m, pid_n) are independent, different
CTAs naturally execute in parallel on the GPU. Some CTAs may be writing to peer buffers
(NVLink transactions) while other CTAs are still computing GEMM. This is **spatial
concurrency**, not temporal pipelining of a single dependency chain.

The degree of spatial concurrency depends on:
- Number of SMs on the GPU
- Number of independent CTAs in the grid
- NVLink bandwidth vs. tensor core throughput

For M=N=1024, BLOCK_M=BLOCK_N=128: grid size = (8, 8) = 64 CTAs.
On an H100 (132 SMs), many CTAs run simultaneously.

## Proposed Architecture for True GEMM-RS Overlap

For a compiler-supported overlap, the following changes would be needed:

### Algorithm Change: Split-K Streaming with Atomic Accumulation

Instead of: compute full GEMM → scatter full result
Do: for each K-chunk, compute partial → atomic-add to peer buffer → continue

```
# Modified inner loop (NOT currently expressible in Triton):
for k in range(K // BLOCK_K // PIPELINE_CHUNKS):
    for kk in range(PIPELINE_CHUNKS):
        a_partial = load(a_ptrs)
        b_partial = load(b_ptrs)
        acc_partial = dot(a_partial, b_partial)
        
    # After PIPELINE_CHUNKS tiles, scatter partial sum atomically
    tl.atomic_add(peer_ptr, acc_partial)   ← streaming scatter
```

The receiver uses `tl.load` (spin-wait on a semaphore counter) to check readiness.
This requires `tl.atomic_add` for the reduction, not `tl.store`.

### New IR Ops Required

1. **`tt.AsyncReduceScatterOp`**: An async store that models scatter + reduce semantics.
   - Attributes: `tt.latency`, `tt.comm_bandwidth` (for scheduling)
   - Semantics: atomic-add to peer buffer, async completion modeled like `tt.load`
   
2. **`tt.SignalSemaphoreOp`** / **`tt.WaitSemaphoreOp`**: GPU-side inter-CTA signaling.
   - Models NVLink atomic operations as compiler-visible async operations
   - Required for peer-GPU spin-wait patterns

3. **Extended `tt.DescriptorReduceOp`**: Currently used for TMA reduce (intra-GPU).
   - Could be extended with a "peer" attribute for cross-GPU reduces

### Extended Pass Design

**Extended `TritonGPUAssignLatencies`:**
- Recognize `tt.AsyncReduceScatterOp` and assign latency = estimated NVLink latency
  (configurable per topology: NVLink4 = ~1μs vs PCIe = ~10μs)
- Recognize `tt.WaitSemaphoreOp` as a "blocking" op that must come after all
  preceding scatter ops to a given peer

**Extended `TritonGPUScheduleLoops`:**
- Interleave scatter ops and compute ops across pipeline stages
- Stage k: issue scatter for K-chunk k-2, compute K-chunk k-1, load K-chunk k

**New pass: `TritonGPUOverlapComputeComm`:**
- Specifically targets compute-communication overlap
- Analyzes whether scatter ops can be moved before the full accumulation
- Inserts synchronization ops and handles the algorithm transformation
  (from "store full result" to "atomic-add partial result")

### Analogy to Existing Warp-Specialized WGMMA Pipeline

`WGMMAPipeline.cpp` overlaps WGMMA with memory loads within a warp group. The key
insight is that WGMMA issues an asynchronous MMA instruction whose result is only
guaranteed ready after `wgmma_wait_group`. This allows overlapping MMA with the
next load — exactly the same pattern as what we want for scatter vs. next compute.

The scatter-overlap case is structurally similar but harder because:
1. The completion signal comes from another GPU (not a local wait group counter)
2. The "latency" is network-dependent and not precisely measurable at compile time
3. The algorithm change (partial accumulation) requires semantic correctness analysis

## Observed Behavior

- The current pipeliner produces correct and efficient output for the GEMM inner loop
- The scatter store is not moved, confirming the dependency analysis above
- No errors or warnings are emitted by the pipeliner regarding the store
  (it is simply outside scope — the pipeliner does not try and fail; it ignores it)

## Hypotheses

- A ring-based streaming reduce-scatter (like NCCL's ring algorithm) could achieve
  ~50% compute-communication overlap even without compiler support, by alternating
  between computing one K-chunk and communicating the previous one. This would require
  restructuring the kernel into a persistent model with explicit tile scheduling.

- Hopper's `ttng.AsyncTMAReduceOp` (for TMA-based reduction stores) could potentially
  be extended for cross-GPU use if the TMA descriptor pointed to a NVLink-mapped address.
  This is a hardware research question, not a software/compiler one.

## Open Questions

- Is there a way to express streaming atomic scatter as a valid Triton kernel today,
  using `tl.atomic_add` to peer buffers, without compiler modifications? If so, what
  is the performance vs. the two-kernel approach?
- Would Triton's interpreter mode (`TRITON_INTERPRET=1`) allow testing the correctness
  of a streaming partial-scatter algorithm before GPU implementation?

## Next Actions

- Write a pseudocode design for the streaming atomic-scatter approach
- Determine whether `tl.atomic_add` to a peer pointer works correctly in Triton today
- Document this design in the final report as "Proposed Architecture"
