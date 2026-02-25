# **GSan \- Concurrency Sanitizer for Global Memory**

# **1\. Introduction**

This proposal outlines the design and implementation of **GSan**, a triton/gluon sanitizer for detecting data races in global memory, including remote accesses within a single node via NVLink. GSan will utilize shadow memory techniques and instrument existing memory operations to work near-seamlessly with existing kernels.

## **1.1. Goals**

* Detect unsynchronized read-after-write, write-after-write and write-after-read violations
* Support peer-to-peer accesses within a single node NVLink domain
* Cover all communication patterns needed by existing code
* Support all TritonGPU memory access operators including: ld/st, atom, cp.async, and cp.async.bulk (aka TMA) operations.

## **1.2. Non-goals**

* Proving correctness
  * This is very difficult as we have hundreds of independent threads, and tracking the full state of each memory cell for all threads is impractical. Thus, we will need to rely on stochastic detection.
* Support inline assembly comms
  * We aim to instrument the TritonGPUIR directly, and parsing inline assembly is out of scope.
  * This means we will need to ensure there are sufficient primitives within the language and library to support comms.
* Detect races with non-triton/gluon code.
  * We assume that all relevant code will be visible to the instrumentation.
  * This means we miss races outwith instrumented code, and may detect a false-positive if uninstrumented synchronization primitives are used.
* Detect races between the threads within the same CTA, e.g. no thread barrier between reading and writing to the same memory location.
  * This may be possible as an extension, but generally we will assume that the IR executes as a single synchronous program.

# 2\. Proposed Algorithm

## 2.1 Vector clocks

The proposed algorithm is based on the concept of **vector clocks**, which is a method to verify “happens-before” relationships in distributed systems.

Each thread maintains a vector clock (`VC[N]`) of time **epochs** for each thread in the system. `VC[tid]` represents the thread’s current epoch, and `VC[i]` for `i != tid` represents the most recent epoch of thread `i` which has been acquired by the current thread.

When two threads act on the same memory location in a way that might race, we can compare each thread's vector clocks to check if the two events are strictly ordered. If `VC(A)[i] <= VC(B)[i]` for all `i`, then `A` strictly "happens-before" `B`, and so `A` does not race with `B`.

Note that “happens-before” is a term of art which implies not simply chronological ordering, but full visibility of the operations within the memory system. A writer can complete the write before a reader, but if the reader doesn’t explicitly synchronize with the writer through a (potentially transitive) acquire-release pattern, then we say the write did not happen-before the read.

## 2.2 Race Detection with Vector Clocks

Using vector clocks, you can construct a fully general race-detection algorithm as follows:

For every memory location we record the vector clock of the last write into a shadow memory region. Any subsequent reads/writes can compare their local vector clock to the writer’s. If the write strictly happens-before the new access, then there is no race. We can also use this clock to update the reader’s vector clock when the reader forms a release-acquire pattern with the writer. This models acquiring both: the writes from that writer (in `VC[tid]`), but also any transitively acquired writes from other threads.

Similarly, for each location we can record the elementwise maximum of the vector clocks of all readers.  This elementwise maximum is the smallest clock value where all readers individually happened-before it. On write, you can compare with the elementwise-max read clock to ensure all previous reads happened-before the write.

Unfortunately for us, this fully general algorithm is not practical because we will be tracking accesses for each CTA separately, and so each vector clock will have hundreds of elements. That would require us to use hundreds of times the original memory usage as overhead to store the clock data. So, instead we will discuss some modifications to reduce memory overhead.

## 2.3 Scalar write clocks

When you compare two writes by a single thread, you can find a strict ordering just by looking at that thread’s `VC[tid]`, since this gets increments for each ordered write. So, if we didn’t care about acquiring transitive dependencies, then we could just store the writing thread’s `VC[tid]` for each memory location. This is convenient for us, since most reads and writes in GPU programming are usually weakly ordered, and so cannot form acquire-release patterns.

Therefore, for non-atomic writes we will only store the local thread’s epoch along with the thread id. However, for atomic writes we will still need to store a full vector clock in order to allow matching atomic-acquires to update their vector clock appropriately.

As an implementation detail, I’m thinking that each thread can have a circular buffer of say 1024 vector clocks. On a GB200 system with 4 GPUs and 152 SMs, then this uses 722 MiB of extra memory which is perfectly reasonable compared to the significant overhead of other shadow memory metadata. We could also reduce this further if we limit the number of SMs that run in parallel, since the overhead scales with O(n^2) in the number of threads.

Also note that a circular buffer does run the risk of wrapping-around, which may lead to issues if it wraps around while a write is still yet to be read. We can, however, detect this edge case by storing the index into the circular buffer without-modulo, and tracking how many times the buffer has wrapped around. This at least means we can raise an error instead of reporting incorrect false-negatives.

## 2.4 Stochastic read clocks

Similar to how write ordering can be checked by only comparing `VC[tid]`, a single reader thread can also be compared by only its local epoch value. However, we have the added complication that there may be many readers between writes, any of which might race with the writer even if the most recent chronological reader did not. This means in general we cannot simply store a scalar.

Instead, we can store a random sampling of `VC[tid]` from readers. We have a fixed buffer of `num_slots` (`VC[tid],tid)` pairs, and on each read the reader will replace 1 of the slots, with a probability proportional to `num_slots / num_readers`. This results in all past readers being equally likely to be checked, no matter how long ago they read in chronological order.

This is the first change that introduces potential false-negatives, as now readers can race but if they are not included in the stochastic clock value then it isn’t reported. However, this is mitigated somewhat by the fact that in practice every CTA is running the same program, so if one reader isn’t synchronized then it’s likely that many aren’t synchronized properly either. This gives us many attempts to randomly find the failing cases.

## 2.5 Asynchronous memory accesses (cp.async/cp.async.bulk.tensor)

So far we’ve assumed that memory accesses happen at a fixed point in time for a given thread, however in the case of cp.async operations there is a window of time which may overlap with other memory interactions in an out of order way.

We can model this by considering two different vector clock values: the value upon issuing the async access (`VC_access`), and the value upon completion (`VC_completion`).  When comparing the existing shadow memory’s vector clock value, we compare it to `VC_access`. This ensures that those operations happened-before the earliest possible time the async access might happen. Then, in the shadow memory we actually store `VC_completion[tid]` as any future reads/writes must happen strictly after the completion of the async op.

In order to do this, we will need to instrument the async wait/mbarrier wait op that forms the actual completion, and do the shadow memory update then. For this we will need to record, for every asynchronous memory access: the associated `VC_access`, as well as metadata that allows us to map the async wait/mbarrier wait ops back to the operation.

## 2.6 Limitations

False Negatives:

* Since we only track a random sampling of the true read vector clock, this means if there are many readers to a single memory location then we may miss races with reads that happened in the past.

False Positives:

* If the thread-local vector clock buffer wraps around before all references to the clock value are done with, then we will have to report a potential race even if none occurred.

These limitations can be mitigated by increasing the number of read samples, or increasing the vector clock buffer size. However, both of these come with increases in the respective memory overhead.

## 3\. Shadow Memory Details

So far we’ve assumed that we can quickly and easily map between a given pointer and its corresponding shadow memory. To do this, we can take advantage of the CUDA driver API which has a number of primitives to control virtual memory addressing. `cuMemAddressReserve` allows us to reserve a large chunk of virtual address space, and `cuMemMap` allows us to map newly allocated memory pages anywhere we want within the reserved address space.

This gives us the tools we need to  implement a custom allocator that maps all tensor allocations into the higher address range with `cuMemMap`; and also creates a corresponding shadow allocation that gets mapped into the lower address range.

During kernel execution, we can tell if memory is managed by GSan by checking if it fits in the bounds of the reserved address space.

```
use_gsan = vmem_base_ptr <= ptr && ptr < vmem_base_ptr + VMEM_SIZE
```

Since we control the exact mapping within the address space, we can also place the shadow memory in a way that makes translation straight-forward. For example, if each 4-byte word has a corresponding shadow memory entry, then we can find the shadow memory by doing:

```c
// Mask out the high bit which indicates the real memory region
ALLOC_MASK = (VMEM_SIZE - 1) >> 1
byte_offset = (ptr - vmem_base_ptr) & ALLOC_MASK
word_offset = byte_offset / 4
shadow_ptr = vmem_base_ptr + word_offset * SHADOW_SIZE_BYTES
```

We also have the option to encode more information into the pointer addresses, since we have a full 64-bit address space to play with e.g. we might have one memory region for “bulk” data with course-grained shadow memory, and one for fine-grained word-level tracking.

# 4\. Triton API Changes

Currently, we have atomic read-modify-write operations which have memory semantic and scope qualifiers in the frontend. Strictly speaking this is all we need as you can simply add 0 to load, or use `atomic_xchg` to store. However, these have overly strong synchronizing effects which may hurt performance of distributed comms.

We may want to consider adding more memory primitives:

1. Scoped memory fences e.g. `tl.fence(scope=”sys”)`.
2. Atomic loads and stores e.g. `tl.atomic_load(ptr, sem=”acquire”, scope=”sys”)`
3. Atomic reduce operations e.g. `tl.reduce_add(ptr, 1, sem=”release”, scope=”sys”)`

These would allow more efficient communication patterns by minimizing the synchronization effects of operations. For example, fence \+ relaxed atomic store allows more memory ops to be placed in-between without causing over-synchronization.
