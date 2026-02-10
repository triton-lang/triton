# Membar Analysis: False Positive Barriers for Async Shared Memory Writes

## Problem

The membar analysis (`lib/Analysis/Membar.cpp`) inserts unnecessary `ttg.barrier local` ops between async DMA copies (e.g., `amdg.async_tdm_copy_global_to_local`) and subsequent `ttg.local_load` ops within the same loop iteration.

### Concrete example

In a warp-pipelined GEMM kernel with triple-buffered shared memory, the loop body has two stages:

```
scf.for ... {
    scf.execute_region {                       // stage0
        async_tdm_copy → a_buffer[prod%3]      // async Write to shared memory
        async_tdm_copy → b_buffer[prod%3]      // async Write to shared memory
        local_load     ← a_buffer[cons%3]      // Read from shared memory
        local_load     ← b_buffer[cons%3]      // Read from shared memory
    }

    amdg.async_tdm_wait {num = 2}             // wait for DMA completion
    // MemWaitOpTrait auto-inserts ttg.barrier local here

    scf.execute_region {                       // stage1
        tt.dot ...                             // WMMA compute
    }
}
```

The `MemWaitOpTrait` on `async_tdm_wait` already causes the membar analysis to auto-insert a `ttg.barrier local` after the wait, which provides cross-iteration synchronization. Yet the analysis **also** inserts a spurious `ttg.barrier local` between the `async_tdm_copy` ops and the `local_load` ops inside stage0. This barrier is unnecessary because:

1. The async DMA write has **not completed yet** — the data written by `async_tdm_copy` is not visible until after `async_tdm_wait` + barrier. There is no RAW (Read After Write) hazard.
2. The producer and consumer access **different slots** of the triple buffer (`prod%3 != cons%3`). They never alias.

### Root cause in the analysis

The `update()` function in `Membar.cpp` processes memory effects at lines 292-311:

```cpp
if (isa<MemoryEffects::Write>(effectInstance.getEffect()))
    curBlockInfo.syncWriteSlices[slice].insert(op);
```

The `async_tdm_copy`'s `MemWrite<SharedMemory>` is recorded as a synchronous write. When the subsequent `local_load` is processed, `isIntersected()` finds the write and the read on the same allocation with unknown subslice offsets (because `memdesc_index` uses dynamic indices), and conservatively inserts a barrier.

Two independent analysis limitations contribute:

1. **No async write model**: All writes are treated as immediately visible. The analysis has no concept of deferred/async writes.
2. **No dynamic offset tracking**: `AllocationSlice` only tracks static offsets from `MemDescSubsliceOp`. `MemDescIndexOp` with dynamic indices produces unknown offsets, causing conservative intersection.

---

## Proposed Solutions

### Near-term fix: Backend filter using `MemAsyncWriteOpTrait`

**Concept**: A `ttg.barrier local` emits `s_barrier`, which synchronizes all threads in the workgroup and fences shared memory. However, it does not synchronize the DMA engine — async DMA writes are in-flight until an explicit wait (`async_tdm_wait`) completes them. Therefore, hazard pairs involving async writes should be filtered out of the membar analysis.

**Implementation**:

1. Define `MemAsyncWriteOpTrait` in `include/triton/Dialect/TritonGPU/IR/Traits.h` and register it in `TritonGPUAttrBase.td`.

2. Add the trait to all async-to-shared-memory ops (see [Affected Ops](#affected-ops)).

3. In the AMD backend filter (`MembarUtility.cpp`), use the trait to suppress false hazard pairs:

```cpp
bool isAsyncWrite(Operation *op) {
  return op->hasTrait<OpTrait::MemAsyncWriteOpTrait>();
}

bool filterAsyncWriteDependencies(Operation *op1, Operation *op2) {
  bool op1Async = isAsyncWrite(op1);
  bool op2Async = isAsyncWrite(op2);
  if (op1Async && op2Async)
    return true;   // WAW between two DMA ops — barrier can't help
  if (!op1Async && !op2Async)
    return false;   // neither async — don't filter
  // One async, one not — filter if the non-async side is a LocalLoad
  // synced via AsyncWait (the wait already provides ordering).
  return isLocalLoadSyncedViaWait(op1) || isLocalLoadSyncedViaWait(op2);
}
```

The core `Membar.cpp` is unchanged. The filter is passed into `isIntersected()` via the existing `MembarFilterFn` mechanism, so it only applies to backends that register it.

**Hazard analysis**:

| Pair | Hazard | Filtered? | Correct? |
|------|--------|-----------|----------|
| async write vs async write | WAW | Yes | Barrier can't order two DMA ops |
| async write vs local_load (synced via wait) | RAW | Yes | Wait already ensures visibility |
| local_load (synced via wait) vs async write | WAR | Yes | Wait + auto-barrier already fences the read |
| async write vs non-synced op | any | No | Conservative — correct |

**Cross-iteration correctness**:

The `MemWaitOpTrait` auto-barrier after `async_tdm_wait` syncs all state, so the back-edge carries empty `blockInfo` to the next iteration. If the auto-barrier were absent, reads would accumulate in `blockInfo.syncReadSlices`, flow via the back-edge, and `isIntersected` would correctly detect the WAR — inserting a barrier exactly where needed.

### Long-term: Proper async dependency tracking in BlockInfo

**Concept**: Extend `BlockInfo` with a third container for in-flight async writes, and model write visibility transitions explicitly.

```
BlockInfo:
    syncWriteSlices   — committed writes (trigger RAW and WAW with subsequent ops)
    syncReadSlices    — reads (trigger WAR with subsequent writes)
    asyncWriteSlices  — in-flight writes (trigger WAR and WAW as current op only)
```

**Rules**:

| Current op      | Prior state                    | Hazard | Barrier? |
|-----------------|--------------------------------|--------|----------|
| Read            | `syncWriteSlices` (committed)  | RAW    | Yes      |
| Read            | `asyncWriteSlices` (in-flight) | —      | No       |
| Write (sync)    | `syncReadSlices`               | WAR    | Yes      |
| Write (sync)    | `syncWriteSlices` (committed)  | WAW    | Yes      |
| Write (sync)    | `asyncWriteSlices` (in-flight) | —      | No (barrier can't help) |
| Write (async)   | `syncReadSlices`               | WAR    | Yes      |
| Write (async)   | `syncWriteSlices` (committed)  | WAW    | Yes      |
| Write (async)   | `asyncWriteSlices` (in-flight) | —      | No (DMA ordering) |

**Transitions**:
- **Async copy** → adds entry to `asyncWriteSlices`
- **Wait op** → promotes entries from `asyncWriteSlices` to `syncWriteSlices` (writes are now architecturally visible)
- **Barrier** → clears all three containers

For count-based waits (`async_tdm_wait {num = N}`), promotion rule: all entries except the N most recent are promoted. This requires `asyncWriteSlices` to be ordered.

**Advantages over the near-term fix**:
- Correct by construction; async writes live in their own container rather than being filtered after the fact
- Eliminates the `MemWaitOpTrait` auto-barrier heuristic; the wait promotes async writes and normal RAW/WAR/WAW logic decides if a barrier is needed
- Composes with multiple async streams and partial waits
- Backend-agnostic: works for AMD TDM, NVIDIA TMA, `ttg.async_copy_global_to_local`, future async ops without per-backend filters

---

## Relationship to Dynamic Offset Tracking

The false positive has two independent causes. The async write fix and better offset tracking are **orthogonal** improvements:

| Improvement              | What it solves                                          | Limitation                                         |
|--------------------------|---------------------------------------------------------|----------------------------------------------------|
| **Async write model**    | Eliminates false RAW for all async-to-shared-memory ops | Does not help synchronous writes to different slots |
| **Dynamic offset tracking** | Eliminates false conflicts for provably disjoint accesses | Cannot prove `prod%3 != cons%3` without symbolic reasoning about loop invariants |

For **this specific kernel**, the async fix is sufficient and more precise — it captures exactly why the barrier is unnecessary (the write hasn't happened yet) without requiring the compiler to reason about dynamic index relationships.

Better offset tracking would independently help a **different class** of false positives: synchronous `local_store` / `local_load` pairs accessing different slots of the same buffer. Today `AllocationSlice` handles static offsets from `MemDescSubsliceOp` but not dynamic indices from `MemDescIndexOp`. Extending it to handle constant-index `MemDescIndexOp` would be a small improvement, but the triple-buffer pattern uses dynamic `remsi` indices that require symbolic analysis beyond what `AllocationSlice::intersects` can practically do.

---

## Affected Ops

Ops carrying `MemAsyncWriteOpTrait`:

| Op | Dialect |
|----|---------|
| `AsyncCopyGlobalToLocalOp` | TritonGPU |
| `BufferLoadToLocalOp` | TritonAMDGPU |
| `AsyncTDMCopyGlobalToLocalOp` | TritonAMDGPU |
| `AsyncTDMCopyLocalToGlobalOp` | TritonAMDGPU |
| `AsyncTDMGatherOp` | TritonAMDGPU |

Candidates not yet carrying the trait:

| Op | Dialect |
|----|---------|
| `AsyncTMACopyGlobalToLocalOp` | TritonNvidiaGPU |
| `AsyncTMAGatherOp` | TritonNvidiaGPU |
