# Triton Software Pipelining: Compiler Pass Inspection

## Confirmed Facts

### Pass Sequence
Three sequential MLIR passes, invoked via:
```
triton-opt -tritongpu-assign-latencies -tritongpu-schedule-loops -tritongpu-pipeline
```

Source locations:
- `lib/Dialect/TritonGPU/Transforms/Pipeliner/AssignLatencies.cpp`
- `lib/Dialect/TritonGPU/Transforms/Pipeliner/ScheduleLoops.cpp`
- `lib/Dialect/TritonGPU/Transforms/Pipeliner/SoftwarePipeliner.cpp` (orchestrates LowerLoops + PipelineExpander)

### Pass 1: `TritonGPUAssignLatencies`

**What it does:** Discovers pipelineable operations and assigns `tt.latency` attributes.

**Ops that receive latency assignments** (`AssignLatencies.cpp:59-243`):
- `tt.LoadOp` (global memory loads) — latency = `(numStages - 1) / (maxIndirectionLevel + 1)`
- `tt.DescriptorLoadOp` (TMA loads)
- `tt.DescriptorGatherOp` (TMA gathers)
- `tt.DotOp` / `ttng.MMAv5OpInterface` / `ttng.WarpGroupDotOp`

**Ops that do NOT receive latency assignments:**
- `tt.StoreOp` (global memory stores) — including scatter writes to peer buffers
- `tt.DescriptorStoreOp` gets handled separately in `TMAStoresPipeline.cpp`, not in `AssignLatencies`
- Any op outside the innermost loop body

**Key constraint (`AssignLatencies.cpp:29-38`):**
```cpp
// Loads must be ≥ 4 bytes for async (cp.async minimum transfer size)
// Loads with non-zero "other" value cannot become async
// Loads with incompatible dot encodings across users are skipped
```

### Pass 2: `TritonGPUScheduleLoops`

**What it does:** Creates a coarse schedule mapping each op to `(stage, cluster)`.

**Preconditions that abort pipelining** (`ScheduleLoops.cpp:36-48`):
- Loop-carried dependency distance > 1
- Is an outer loop (only innermost loops are pipelined)
- Loop body contains `ttg.BarrierOp`
- Loop body contains `tt.AssertOp`
- Loop body contains `tt.PrintOp`

**Scheduling algorithm:**
1. Place ops with `tt.latency` attribute in earlier stages (stage = last - latency)
2. Move distance-1 dependencies to the next stage
3. Place remaining ops in the last stage (topological order preserved)

**Output:** `loop.stage` and `loop.cluster` attributes on each op.

### Pass 3: `TritonGPUPipeline` (SoftwarePipeliner)

**What it does:** Transforms the loop using the schedule to emit prologue/epilogue.

**Two-phase implementation:**
1. `LowerLoops` (`LowerLoops.cpp`): converts `tt.load` to `ttg.AsyncCopyGlobalToLocalOp`, creates multi-buffered shared memory allocations and barrier allocations
2. `ExpandLoops` (`PipelineExpander.cpp`): calls `triton::pipelineForLoop()` — generates prologue (stages 0..maxStage-2), pipelined kernel, epilogue

**Generated ops after pipelining:**
- `ttg.async_copy_global_to_local` — replaces `tt.load` inside the pipeline loop
- `ttng.async_commit_group` — commits pending async copies into a group
- `ttng.async_wait {num = N}` — waits for N async groups to complete
- `ttg.local_load` — loads from shared memory after wait
- Prologue iterations: first `(numStages - 1)` iterations unrolled before the main loop

### Key Constraint for Reduce-Scatter

The scatter `tt.store` (writing GEMM result to peer symm mem buffer) is:

1. **Outside the inner k-loop** — the pipeliner only pipelines ops *inside* the loop it targets. The store is executed once per CTA, after the loop completes.

2. **Dependent on the fully-accumulated `acc`** — `acc` is a loop-carried value that accumulates across all k iterations. Its final value is only available after the last iteration. There is no "partial scatter" because the receiver would see an incorrect (partial) value.

3. **Has no `tt.latency` attribute** — `AssignLatencies` never annotates store ops. Even if the store were inside the loop, it would not be given a stage assignment.

4. **Not modeled as a communication op** — the peer buffer write is simply a `tt.store` to a pointer that happens to point to peer GPU memory. The pipeliner has no semantic knowledge of its communication nature.

## Observed Behavior

From IR dumps (captured with `MLIR_ENABLE_DUMP=1`):

**Before `TritonGPUPipeline`** (after `TritonGPUScheduleLoops`):
- The `scf.for` loop has `{tt.num_stages = 3}` attribute
- The two `tt.load` ops have `{loop.stage = 0, loop.cluster = 0, tt.latency = 2}` attributes
- The `tt.dot` has `{loop.stage = 2, loop.cluster = 0}` (last stage)
- The scatter `tt.store` is outside the `scf.for`, no stage attribute

**After `TritonGPUPipeline`**:
- The `scf.for` is transformed with prologue/epilogue
- `tt.load` → `ttg.async_copy_global_to_local` + `ttg.local_load`
- `ttng.async_commit_group` and `ttng.async_wait {num = 2}` inserted
- The scatter `tt.store` is **unchanged** and appears after the loop

## Hypotheses

- With a Hopper GPU (SM90), the loads might become TMA operations (`tt.DescriptorLoadOp`) with `mbarrier` synchronization instead of `cp.async` + `ttng.async_wait`. The overall structure (scatter store unchanged) would be the same.

- If the scatter were inside the k-loop (e.g., scatter partial accumulation), the pipeliner might give it a stage attribute — but the semantics of partial scatter would be incorrect unless the receiver uses atomic operations.

## Open Questions

- Does the pipeliner interact differently with Hopper's `ttng.WarpGroupDotOp` in ways that affect the scatter analysis?
- Would `ttg.BarrierOp` placement (to model the cross-GPU synchronization) prevent the pipeliner from running at all on the outer loop?

## Next Actions

- Confirm the before/after IR diff by reading `diff_ir.txt`
- Check if any existing `TMAStoresPipeline.cpp` logic could be repurposed for async scatter
