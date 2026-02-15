# Triton Instrument Dialect and Concurrency Sanitizer (ConSan)

### Overview

ConSan instruments Triton IR to detect illegal concurrent accesses to shared and Tensor Core memory under warp specialization. It tracks per-buffer visibility of reads and writes across threads, models barrier-based synchronization, and models commit-count–based synchronization (cp.async, wgmma).

Auxiliary state is kept in distributed tensors and global scratch memory, with types created on-demand per warp-specialization partition.

### Thread model

- Base threads: 16 warp-specialization (WS) threads (allowing for up to 16 partitions).
- Peer classes: +16 Tensor Core (TC) threads and +16 TMA threads to model lack of ordering with base threads.
- Total logical threads: 48. Bitmasks are sized to the next power of two: 64.

Indexing uses a logical thread id in [0, 48), with column vectors sized to 64 for layout convenience.

## Auxiliary data structures

All types are generated on-demand (per partition) based on:

- B: number of tracked buffers (power-of-two padded)
- K: number of mbarriers (power-of-two padded)
- T_bits: 64 (bitmask width)
- T_commits: 16 (base threads; commit counters do not apply to TC/TMA helpers)

“tensor” means a distributed Triton tensor; “scratch” means a pointer into global scratch memory. Shapes below are logical; actual encodings are partition-local blocked layouts.

- buffers (tensor, <B x i64>): Base pointers of all (sub)buffers per memory space
- barriers (tensor, <K x i64>): Pointers of all mbarriers
- writeVisibility (scratch, <B x i64>): Per-buffer bitmask. Bit i set ⇒ thread i can see latest completed write to that buffer
- readVisibility (scratch, <B x 64 x i64>): Per-buffer, per-thread lanes. Each lane stores a 64-bit mask of other threads whose reads are visible to that lane’s thread
- writeTracking (scratch, <B x K x i8>): Map buffers → barriers tracking writes (boolean stored in i8)
- readTracking (scratch, <B x K x i64>): Map buffers → barriers tracking reads (bitmask of threads)
- barrierStates (scratch, <K x i32>): Packed barrier metadata. Bit 0 stores the current phase, bits [1..8] the initial arrival count, bits [9..16] the current arrival count. The verifier checks underflow before updating, and flips the phase when the current count reaches zero.
- waiting (scratch, <K x i32>): Per-barrier bitfield describing waiting threads. Each base thread gets two bits: bit (2 * thread + 0) is the waiting flag, bit (2 * thread + 1) stores the phase the thread is waiting on.
- outstandingCommits (scratch, <B x 16 x i8>): Per-buffer, per-base-thread commit counters for cp.async and wgmma

## Visibility and legality rules

- Reads are legal iff the reading thread sees the most recent write to the buffer (writeVisibility). There can be only one write in-flight.
- Writes are legal iff the writing thread sees both all prior writes and all reads completed for that buffer.

ConSan enforces these via two checks emitted before memory ops:

- experimental_verify_write_visibility: “no one else is writing, or I can see the write”
- experimental_verify_read_visibility: “my read-visibility lane is a superset of the OR of all lanes”

## Barrier-based synchronization

ConSan separates “tracking” from “visibility transfer”:

- At memory ops that are tracked by a barrier (loads/stores, some TMEM ops):
  - experimental_set_read_visibility / experimental_set_write_visibility updates the appropriate visibility table for the current thread and buffer.
  - experimental_track_visible_reads / experimental_track_visible_writes snapshots current per-buffer visibility into readTracking/writeTracking for the given barrier.
- At arrive/commit sites (e.g., tc commit, arrive on mbarrier): ConSan emits the track ops for both reads and writes.
- At waits: experimental_transfer_visible_reads / experimental_transfer_visible_writes propagates tracked visibility from the barrier back into the waiting thread’s visibility, and this transfer is repeated to peer threads (base, TMA, TC) to keep the three classes consistent.

### Barrier phase/count tracking

- experimental_init_barrier_state(barrier, count, barrierStates) initializes the per-barrier state with phase = 0 and both initial/current arrival counts = `count`.
- experimental_verify_barrier_arrive(barrier, count, barrierStates) checks that subtracting `count` from the current arrival count would not underflow. The codegen emits an assert if it would.
- experimental_update_barrier_state(barrier, count, barrierStates) applies the arrive: subtracts `count`, flips the phase when the count reaches zero, and reloads the current count from the initial count.

### Deadlock detection

ConSan records which phase each thread is waiting on:

- experimental_set_waiting(barrier, baseThread, phase, barriers, waiting) sets the waiting flag for `baseThread` and stores the requested `phase`. The flag/phase bits share the waiting bitfield (two bits per base thread).
- experimental_check_all_active_waiting(activeMask, barriers, waiting, barrierStates) filters waiting threads to those whose stored phase matches the current barrier phase. If all active threads are waiting on matching phases, it raises a deadlock assert.
- experimental_clear_waiting(barrier, baseThread, barriers, waiting) clears the waiting bits for `baseThread`. Each wait clears its own state after the wait completes.

## Commit-count–based synchronization

Some hardware ops synchronize via “number of outstanding commits” rather than mbarriers.

- Stage: experimental_stage_access_for_commit marks the current thread’s buffer lane with -1 (staged) in outstandingCommits[B x 16].
- Commit: experimental_commit_accesses turns -1 into 1 and increments positive entries for the committing thread column.
- Wait (cp.async): experimental_clear_outstanding_commits_set_write(thread, commits, writeVisibility, N) clears entries with count > N for the current thread, and sets the writeVisibility bit for rows where any thread’s entry was cleared.
- Wait (wgmma): experimental_clear_outstanding_commits_set_read(thread, commits, readVisibility, N) clears entries with count > N for the current thread, and sets the readVisibility bit for rows where any thread’s entry was cleared.

Legality checks for commit-count flows:

- For writes to shared memory affected by cp.async: experimental_check_outstanding_commits(buffer, commits, "async_copy_global_to_shared") asserts the row for the buffer is all zeros (no pending writes), across all base-thread columns.
- For reads of wgmma operands in shared memory: experimental_check_outstanding_commits(buffer, commits, "warpgroup_mma operand read") asserts the row is all zeros (no pending reads).

Note: The check op has no “thread” operand; it inspects the whole row for the buffer.
