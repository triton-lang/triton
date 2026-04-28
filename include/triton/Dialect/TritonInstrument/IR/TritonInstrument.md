# Triton Instrument Dialect and Concurrency Sanitizer (ConSan)

## Overview

ConSan instruments Triton IR with runtime checks for illegal concurrent access
to shared memory and tensor memory. The pass tracks a per-buffer frontier of
visible reads and writes, models mbarrier synchronization, and models
commit-count synchronization for asynchronous operations such as `cp.async`,
WGMMA, TMA store, and AMD TDM copies.

The pass is target-hook based. The module target selects the hook
implementation:

- `cuda:*` uses the NVIDIA hooks.
- `hip:*` uses the AMD hooks.

ConSan currently supports one public entry point in the module. It uses
BufferRegion analysis to collect shared-memory buffers, tensor-memory buffers,
and barrier allocations, then creates auxiliary state in distributed tensors and
shared-cluster global scratch memory. Most state has a leading CTA dimension so
cluster and multicast effects can be modeled explicitly.

## Visibility Model

ConSan models races by tracking visibility of completed accesses, not by
building a general happens-before graph. For each tracked buffer, ConSan records
which logical threads can see the latest write, and which prior reads are
visible to each logical thread. A memory operation is legal only if the issuing
logical thread can see the completed accesses that would conflict with it.

Synchronization operations update visibility rather than creating vector-clock
style ordering metadata. Barrier arrives and commits publish the accesses that
are visible to the issuing logical thread. Barrier waits, commit-count waits,
and target-specific synchronization then transfer that published visibility to
the waiting logical thread and its peer thread classes. This keeps the runtime
state close to the property ConSan needs to check: whether a given access has
already been made visible as finished to the thread that is about to access the
same buffer.

## Thread Model

ConSan uses logical thread ids rather than hardware lane ids:

- Base threads: 16 logical warp-specialization slots. The default region is
  thread 0; warp-specialize partition regions use `partition_index + 1`.
- TMA peer threads: 16 additional slots at offset 16.
- Tensor Core peer threads: 16 additional slots at offset 32.
- CLC peer threads: 16 additional slots at offset 48.
- Total logical slots in use: 64. Visibility masks are 64 bits.

For a base thread, `getThreadPeersMask` returns the base thread plus its TMA,
Tensor Core, and CLC peers. For a TMA, Tensor Core, or CLC thread, it returns
only that helper thread. Commit-count tracking uses only the 16 base-thread
columns, so helper threads are folded back with `thread % 16` where commit
counters are involved.

At a `ttg.warp_specialize`, the pass copies the default thread's read and write
visibility to the destination partition peer masks so partition-local execution
starts with the visibility frontier that existed before specialization.

## Runtime State

ConSan keeps enough runtime state to answer two questions at each instrumented
operation: which logical threads can see the latest writes, and which prior
reads must be visible before a write is legal. The state is tracked separately
for shared memory and tensor memory when those memory types are present.

At a high level, the pass maintains:

- Buffer and barrier descriptors discovered by BufferRegion analysis.
- Read and write visibility frontiers for each tracked buffer.
- Barrier lifecycle, waiting, and barrier-to-buffer tracking state.
- Outstanding commit counters for commit-count-ordered asynchronous flows.
- Optional alias metadata when tracked buffer regions overlap.
- A shared-cluster lock that serializes instrumentation updates.

Most runtime state is CTA-indexed so cluster, multicast, and cross-CTA effects
can be represented directly. Scratch state is zero-initialized once before the
instrumented body runs, and the initialization is followed by a CTA or cluster
barrier before any instrumented operation can use it.

The exact auxiliary data layout is intentionally documented next to the
implementation in `AuxDataMap` in
`include/triton/Dialect/TritonInstrument/IR/Utility.h`.

## Memory Legality

For a read effect, ConSan checks that there is no outstanding write to the
selected buffer, or that the reading thread can see the latest write. For a
write effect, ConSan checks both write visibility and read visibility: the
writing thread must see the latest write frontier and all prior reads for the
selected buffer.

The runtime checks account for aliasing and CTA recipients. A check against one
buffer is expanded through the alias metadata when BufferRegion analysis found
overlapping tracked regions, and multi-CTA operations only inspect the CTA rows
that the operation can affect.

After a barrier-tracked read, ConSan records that the current peer thread mask
can see that read. After a barrier-tracked write, ConSan records the current
peer thread mask as the new write frontier for the buffer and clears prior read
state that the write supersedes.

All normal instrumentation emitted around one IR operation is wrapped in the
ConSan lock. Barrier waits are split into a locked pre-wait section and a locked
post-wait section.

## CTA Recipients and Multicast

Most multi-CTA instrumentation computes a CTA recipient bitset. That bitset is
converted to a mask over the CTA dimension so only relevant CTA rows are checked
or updated.

The target hooks compute recipients from the operation:

- Non-multicast operations usually target the current CTA.
- Multicast TMA loads update all result-recipient CTAs, while barrier arrivals
  route to the leader barrier CTA.
- NVIDIA two-CTA Tensor Core operations are predicated to the issuing CTA pair
  leader.
- TMA load effects that write one set of CTAs and signal a different leader
  barrier remember the effect-recipient CTA rows so a later wait can transfer
  write visibility to both the waiting CTA and the written CTA rows.
- CLC try-cancel effects use all CTA rows for the barrier and memory effect,
  with diagonal recipient handling for the written result rows.

## Barrier Synchronization

ConSan separates barrier tracking from visibility transfer.

For frontier-tracked barriers, an arrive or commit snapshots the current
thread's visible writes and reads into the barrier's tracking state. A later
wait transfers that tracked visibility to the waiting thread's peer class.

Some effects use precise write tracking instead of frontier tracking. For
example, NVIDIA TMA and CLC operations can attach only the buffers written by
that operation to a barrier, together with the CTA rows reached by the memory
effect.

On a barrier wait, ConSan:

1. Acquires the ConSan lock.
2. Verifies the barrier is initialized.
3. Sets the current base thread's waiting flag and phase.
4. [Deadlock] Checks whether all active base threads are waiting on matching barrier
   phases.
5. Releases the lock and lets the real wait execute.
6. Re-acquires the lock after the wait.
7. Transfers tracked write and read visibility from the barrier to the current
   thread's peer mask for shared memory and tensor memory.
8. Clears the current base thread's waiting bits.

Write transfers also consult the recorded effect-recipient CTA rows, which lets
TMA-style and CLC cross-CTA writes become visible in the CTA rows reached by the
memory effect. Read transfers update the current CTA row.

## Barrier Lifecycle and Deadlock Checks

The barrier state table models initialized, invalidated, phase, arrival-count,
and tx-count behavior:

ConSan asserts that barriers are initialized before use and not reinitialized
without invalidation. Arrivals are checked for count underflow and tx-count
range violations before the shadow barrier state is updated. When both the
current arrival count and tx-count reach zero, the shadow state flips phase and
reloads the current count from the initial count. Invalidation clears the
barrier lifecycle, waiting, and read/write tracking state.

Deadlock detection records the phase each base thread is waiting on. The check
aligns those stored phases with each barrier's current phase, filters to active
base threads, and asserts if every active thread is waiting on a matching phase.

## Commit-Count Synchronization

Commit-count synchronization is used for operations whose completion is ordered
by outstanding commit groups rather than by a barrier. It is only modeled for
shared-memory buffers.

Conceptually, an asynchronous access moves from staged, to committed but still
outstanding, to cleared by a wait. Waiting clears accesses older than the
pending-count threshold and transfers the corresponding read and/or write
visibility to the waiting thread's peer mask.

Before shared-memory reads and writes, ConSan checks target-defined outstanding
commit kinds. The check inspects all relevant CTA/buffer rows, expands aliases
when necessary, and can exclude the caller's own base-thread column for ordered
commit kinds. That exclusion is used by targets whose operations complete in
issue order within one ConSan logical partition, avoiding false positives for
same-partition ordering while still checking cross-partition races.

## Target Coverage

The common hook implementation covers these TritonGPU operations:

- `ttg.async_copy_global_to_local`: shared-memory write tracked with
  `AsyncCp` commit counts.
- `ttg.async_commit_group`: commits staged `AsyncCp` accesses.
- `ttg.async_wait`: clears `AsyncCp` entries beyond the pending-count threshold
  and transfers write visibility.
- `ttg.local_load`: barrier-tracked shared-memory read.
- `ttg.local_store`: barrier-tracked shared-memory write.
- `ttg.local_alloc` with a source: barrier-tracked shared-memory write.

NVIDIA hooks additionally cover:

- `ttng.init_barrier`, `ttng.wait_barrier`, and `ttng.inval_barrier` lifecycle
  and wait instrumentation.
- `ttng.barrier_expect`, including tx-count accounting and the non-leader CTA
  arrive path for multicast barriers.
- `ttng.arrive_barrier`.
- TMA loads as barrier-tracked writes with tx-count decrement and precise
  effect-write tracking.
- TMA stores as `TmaStore` commit-count reads, with `ttng.tma_store_wait`
  transferring read visibility.
- TMEM load, store, alloc-with-source, and copy operations.
- TCGen5 MMA, scaled MMA, commit, and TMEM copy operations as Tensor Core peer
  thread effects.
- CLC try-cancel as a CLC peer-thread write with EffectWrites barrier tracking,
  and CLC load-result as a barrier-tracked read.
- Async WGMMA operands in shared memory as `Wgmma` commit-count reads, with
  `ttng.warp_group_dot_wait` transferring read visibility.

AMD hooks additionally cover:

- AMD barrier init and wait instrumentation.
- AMD explicit barrier arrives, with arrive count scaled by warps and threads
  per warp.
- Async TDM global-to-local and local-to-global copies. With a barrier, these
  are modeled through barrier arrivals; without a barrier, they use `TmaStore`
  commit counts and implicit commits.
- AMD async wait variants for `AsyncCp`, and TDM wait variants for `TmaStore`
  commit counts.
- Ordered TDM commit kinds, using the self-column exclusion described above.

## Implementation Notes

- ConSan models one logical thread per warp-specialization partition, not every
  hardware lane. Target hooks compensate for known ordered same-partition
  commit flows where possible.
- Global-memory race checking is handled by the separate global sanitizer, not
  by ConSan.
- The pass expects exactly one public entry point in the module.
