# Agenda:
1. Improving ILP (Instruction Level Parallelism) with Warp Specialization
2. Triton-shared (Progress and updates)
3. Question about generic tensor descriptors

# Meeting notes:

## Improving ILP (Instruction Level Parallelism) with Warp Specialization
Speakers: Hongtao Yu (Meta), Yuanwei (Kevin) Fang (Meta), Manman Ren (Meta)

Notes:
* Pytorch 2.6 with Triton release branch 3.2
* Targeting: Nvidia Hopper arch, Blackwell coming soon.
* Performance
  * Meta’s FP8Rowwise GEMM (3-5% improvement, 1D persistent loop)
  * FlashAttention (10-15% improvement, could be faster with pipelining and pingpong scheduling).
* What is warp specialization?
  * Improves hardware instruction scheduling. GPUs don’t have good dynamic instruction scheduling.
  * Use multi-way warp scheduler. Allows warps on a single core targeting different function units (e.g. memory, ALU, tensor core, etc.)  All run in parallel.
* Comparison using GEMM * *
  * Uniform warps: 8 warps, each loading/processing 1/8th of data.  Divided into two groups, each doing ½ the data. Good for GEMM but not for more complicated kernels.
  * Warp specialized: 12 warps, 4 warps for producing data-only do load, 8 for wgmma-only do wmma.  Frees up more capacity for more complex kernels like flash attention.
* Compiler implementation
  * How to enable warp specialization
    * Automaticlly enabled by adding two switches to autotune config.
      * Num_consumer_groups - non-load warp groups
      * Num_buffer_warp_spec - # of buffers between producer and consumer
  * Concept
    * Async tasks run in parallel with other async tasks.
    * Tasks should use different memory and GPU resources.
    * Coordination through shared memory and barriers for synchronization.
  * Compiler Implementation
    * Automatic task partitioning.
    * Dataflow Multi-buffering
  * Task partitioning
    * Automatic task partitioning identifies tasks like loads, alu ops, stores, etc.
    * Identifies dependency chains. Links producers to consumers.
    * Continue partitioning and inserting synchronization primitives in both producer and consumer warps.
  * Multi-buffering
    * Producer continues to load/populate buffers in round-robin while consumers processes individual buffer.
    * Producer blocks when no free buffers available.
  * In the future
    * Multi-buffering multi-dimensional loops
    * Buffer reuse in over multiple regions in a single group
    * Complex control flows, partition schemes (ping-pong, support for Blackwell)
* Case Study: Flash Attention - Kevin and Manman
  * Without WS
    * Compute Througput: 45%
    * Memory Throughput: 35%
    * SM Busy: 46%
    * No interleaving: CUDA core idle when tensor cores running
  * With WS
    * Compute Throughput: 69%
    * Memory Throughput: 35%
    * SM Busy: 71%
    * Interleaving (speed up due to):
      * Overlapping TMA with CUDA core op
      * Overlapping cuda core and tensor core
      * Overlapping tensor core and instruction issuing.
    * Data partitioning
    * Communication pipelining and ping-pong scheduling
    * Ping-pong is named barrier pair. Only one consumer can be in region.

## Questions
* Q> Is there an equivalent warp group for AMD? Does this apply to AMD GPUs?
* A> Meta is doing this for AMD. No named barrier in AMD. Simulating this using shared-memory atomics on AMD to get the same effect.

* Q> Would it make sense to promote these to a higher level inside Triton for complex cases where it would be difficult for the compiler to detect?
* A> Yes. We allow users to annotate programs with their partitions in [facebookexperimental/triton](https://github.com/facebookexperimental/triton).  We want to see if more automation is possible.

* Q> What should we target first? Warp specialization or software pipelining as an initial optimization? From your experience, which lowering is preferred?  Are you going to bring it to main?
* A> Not mutually exclusive.  You need to figure out what makes sense for yourself.  WS benefit: outerloop support for pipelining. WS benefit: overlapping of cuda core and tensor core.

* Q> What improvements are you seeing?
* A> Flash attention: 20%  + computational pipelining and ping-pong scheduling approaches flash attention v3 performance.

## Triton-shared (Progress and updates)
Presenter: Nhat Nguyen (Microsoft), Haishan Zhu (Meta)

Notes:

### Goal:
* Lower Triton IR to mlir core dialects (linalg, memref, …)  Easier path to running on CPUs.
* Focus on supporting strided memory access for accelerators
* Open-sourced at https://github.com/microsoft/triton-shared
  * Trying to keep it in sync with OSS triton (albeit a little delayed)

### Progress
* Modularizing compiler passes. Decoupled data extraction from lowering. Allowed for customized lowering flows. Predictable behavior for analysis failures.
  * Triton-to-structured
  * triton-arith-to-linalg
  * Structured-to-memref
* Improvements to pointer analysis
  * Supports nested loops
  * Non-contiguous memory access.
* Support for lowering unstructured access with single base pointer
* Support lowering triton ops to linalg/mlir (split, join, cat, etc.)

### Roadmap
* Complete support for non-contiguous pointers
* Detect other memory access patterns (e.g. row-gather/scatter pointer sequences)
* Extend to control flow ops

### Thanks!
Meta, Qualcomm and community

### Questions
* Q> Future plans, what are the higher priority items you want to work on?
* A> Many Triton kernel have memory access patterns  that can’t be detected. We don’t have fall back solutions (e.g. gather-scatter support). Need to wait for the mlir pointer dialect to land so we can use it.  MxN loads pointer analysis fails if loads are contiguous. But rows may be contiguous so we can split analysis into multiple chunks (row scatter, row gather).
* A> In places where pointer analysis can’t extract information, we leave the IR intact so existing passes that can deal with them. We can handle loop iteration over tensors of pointers (common patterns). More complicated operations like if/else look like low hanging fruit.

## Questions about Generic Tensor Descriptor
* Q> What is the progress on generic tensor descriptor programming?  Not Nvidia specific. (from last month).
* A> TMA accelerator will probably become more general across GPUs.
* A> TMA (tensor descriptors) support should be landing over next few weeks.  Will add compatibility mode for GPUs without TMA (but will probably be slower).  And will be adding block pointer support.  We will deprecate host side tensor descriptors (only provided minor performance benefit for persistent kernels).  Allow user to autotune.

## Minutes:
Recording link [here](https://www.youtube.com/watch?v=cIW6ZL_LmGc)
