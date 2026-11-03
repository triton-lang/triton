# Agenda:
* Triton to TileIR - Feiwen Zhu, Nvidia Shanghai (mzhu@nvidia.com)
* [Optimal Software Pipelining and Warp Specialization for Tensor Core GPUs](https://arxiv.org/pdf/2512.18134)  - Rupanshu Soi, Rohan Yadav, Fredrik Kjolstad, Alex Aiken, Mayam Mehri Dehnavi, Michael Garland and Michael Bauer,  Nvidia and Stanford

# Minutes:
* Triton to TileIR \- Feiwen Zhu, Nvidia Shanghai ([mzhu@nvidia.com](mailto:mzhu@nvidia.com))
  * CUDA Tile \- enables first class tile programming in CUDA.
  * CUDA Tile IR \- intermediate representation (semantics, ops & types for tiles)
  * Idea: add CUDA Tile IR as backend to Triton in addition to PTX
  * Why?
    * Nvidia SIMT model moving to tile-based abstractions
    * Nvidia profiler, debuggers, etc will support these abstractions
    * Extend Triton’s Python syntax to gain native access to  tensor cores
  * Biggest change: program CUDA using block/tiles not individual threads\!
  * Direct translation from Triton tiles to Tile IR (no lowering to threads). Programming done at the block/tile level.
  * Don’t need to rewrite Triton programs to use CUDA Tile IR backend. (target backend selectable using env vars.)  He uses virtual environments to toggle.
  * Can be done on a kernel by kernel basis.
  * Q\> Per kernel? You can annotate kernel to say which backend you want?
  * A\> Yes. But we’re currently just using venv to switch. We can add an API to make switch.
  * Q\> Could it be used by autotune to see which backend is faster?
  * A\> Yes. But we haven’t implemented that yet.
  * Q\> CuTILE distinguishes between kernel and device functions. Triton doesn’t.
  * A\> Limitation of CuTILE not CUDA Tile IR.
  * Tuning space is very different. New knobs
    * Occupancy \- **very critical**  register and shared mem usage. Prevents overagressive optimization.
    * Num\_warps \- (not needed for compute bound kernels), doesn’t affect performance.
    * Q\> If I want to tune generic kernel using CUDA Tile IR and change num\_warps what happens?
    * A\> Won’t change the performance, its ignored by the backend..
    * Q\> Won’t that confuse the programmer (I set it but its ignored, should the compiler enforce it?)
    * A\> We set it to 2 or 4 and let the CUDA Tile IR compiler take care of it for compute bound kernels. We may do something with it later.  For layered norm and some other kernels it may be important.
    * Num\_stages \- only used as a cost signal
    * Auto Warp Specialization (WS) is done differently for CUDA Tile IR backend.
    * Q\> How is occupancy exposed to developers?
    * A\> User doesn’t need to modify kernel, just add it to the config.
    * Q\> Will it use the current Auto WS implementation in Triton?
    * A\> No. CUDA Tile IR has lots of proprietary info that aren’t viewable by public to help with WS.
    * Q\> Latency optimization hint in Tile IR. Whats it used for?
    * A\> Used for scheduling if latency of ops like loads are known.  Let tuner or user tune this. For most cases, use the default, only used in corner cases where compiler can’t decide.
    * Num\_ctas \- **very critical**  enables  Blackwell 2CTA throughput speedup for MMA.
  * Limitations
    * Lots of Triton Ops not implemented yet (but planned support)
    * Used unordered memory model \- global memory accesses aren’t ordered (default)
    * Memory token semantics \- (needs Triton API extension)
    * Incorrect results if kernel
      * Uses memory aliases between different global memory accesses
      * Data transactions across tile blocks (splitK/streamK)
      * Deterministic reduction need global mem locking logic.
  * Performance issues
    * Small GEMMs slow (working on it)
    * Tensor-of-pointer load/store slow (working on it)
    * Num\_warps \- needed for XXNorm kernels, (working on it) Root cause: register spilling.
    * Q\> Do you need tensor descriptor API to get good perf now?
    * A\> Yes. descriptor gives you perf. But will be fixed in 13.2 so not necessary in future.
  * Supported arch roadmap
    * 13.1 \- Blackwell
    * 13.2 \- Blackwell and Ampere/Ada
    * 13.3 \- Blackwell Ampere/Ada and Hopper
    * Future: SIMT interop, native communication, profiler/debugger support.
  * Questions
    * Q\> What criteria used to select between PTX and CUDA Tile IR backend?
    * A\> Eventually we should have same perf on both.
    * Q\> How about CuTILE vs Triton-\>CUDA Tile IR comparison (is one better than the other)
    * A\> Difficult to answer. Perf should be the same if Nvidia invests for heavily in CuTILE.
    * Q\> How about translating Tile IR to Triton IR to PTX?
    * A\> No. Probably not.
    * Q\> Tile IR uses views and triton uses pointers, what are the implications (translating one to the other?)
    * A\> Follow up offline.
    * Q\> CuTILE frontend stability? More APIs coming in future?
    * A\> CUDA Tile IR will use TMA load by default but falls back to global load.

* Optimal Software Pipelining and Warp Specialization for Tensor Core GPUs ([paper](https://arxiv.org/pdf/2512.18134)) \- Rupanshu Soi, Rohan Yadav, Fredrik Kjolstad, Alex Aiken, Mayam Mehri Dehnavi, Michael Garland and Michael Bauer,  Nvidia and Stanford
  * New GPUs \-\> rewrite everything. E.g. Flash attention, FA2, FA3 and FA4
  * Why couldn’t this be done automatically?
  * Relies on software pipelining (SWP) and Warp Specialization (WS)
  * SWP (software pipelining) \- well-known, optimal algorithms known since 80s.
  * WS (warp specialization) \- only heuristics
  * Combo of SWP \+ WS not well understood.
  * Meta’s blog post: Warp Specialization in Triton: Design and Roadmap
  * Idea: treat it as a resource constraint scheduling problem.
  * Must solve *both* problems concurrently.
  * Twill
    * Input: program
    * 1\. (maximization problem) Generate optimal SWP
    * 2\. (satisfaction problem) Optimal SWP+WS.
    * Output: SWP program with WS annotations (for feeding into a compiler).
    * Q\> Is this specialization at the warp or warp group level? Warp group level can have different register allocation.
    * A\> Could be at either. Add a constraint. Will explain later.  You can tell the system that certain operations require warp groups.
  * Phase I. Generating Optimal SWP
    * Instruction level parallelism
    * Use modulo scheduling (standard way, we generate optimal for specific program and GPU) see (“An Integer Linear Programming Model of Software Pipelining for the MIPS R8000 Processor”)
  * Phase II. Optimal SWP \+ WS
    * When is WS helpful?
    * More warps means access to more registers (single warp has limit on number of registers it can access.)
    * Individual warps can process different instruction streams.
    * Move blocking instructions on a warp to another warp.
    * Variable latency instructions like TMA load (depends on cache hit/miss).
    * Variable latency instructions go to different warp, scheduled before main pipe.
    * Q\> Lots of loads \-\> saturate cache \-\> eviction? Will data be available when calc takes off?
    * A\> No guarantee. Might get lots of misses in L2 cache. Most of the time should be ok.
    * Basically doing static scheduling with instructions with known latencies.
    * Q\> Does this depend on data size?
    * A\> Cost model factors in data size (clock cycle latency). Based on throughput numbers for GPU.
    * All tiles have fixed tile sizes at compile time (we have cycle counts at compile time)
    * Implementation:
      * Start with optimal SWP program from Phase I.
      * Convert to SMT problem (use SMT solver) Fields:
        * Instruction
        * Instance
        * Issue cycle
        * Warp
      * Example constraints:
        * EXP() requires 80 cycles to execute.
        * Size of live data on warp doesn’t exceed register budget.
    * Results
      * Twill discovered FlashAttention3 impl from Triton DSL FlashAttention impl by just using SMT constraints!
      * Passing a flag changing constraints to Blackwell, came up with FlashAttention4 impl\!
    * Implementation
      * Triton compiler passes
      * TTGIR with warp annotations
      * Triton compiler generated poor performing code SO... hand translated the TTGIR to CUDA C++.
      * Considered gluon but it was too early, wanted the control we got with C++.
      * H100 \- on par with FA3 *better than Triton*
      * Blackwell \- on par with cuDNN/FA4 (faster than Triton)
      * Collected on Dec 2025.
    * Time to find solution (mostly in SMT solver)
      * 36s for H100
      * 22s for B200
    * How to use twill
      * Use optimal schedules to generate fast heuristics.
      * Optimal schedules can guide humans to write fast kernels.
    * Questions
      * Q\> Opensourced?
      * A\> No, not yet
      * Q\> Does it fit into the polyhedral methods/analysis?
      * A\> Haven’t thought about it. Pretty different. Not doing fusion, just scheduling.
      * Q\> Maybe use this as an automatic tool like helion, run auto tuner once to produce annotations and add to kernels? How to add this schedule to a kernel so it could be used anytime.
      * A\> Twill generate Triton IR. Rest of Triton compiler can’t generate performant code. Could be consumed by lower level tools. TLX can group instructions into tasks and warps run different tasks. Most triton compilers have problem accepting software pipelining IR.
      * Q\> What was the problem with TTGIR lowering?
      * A\> Compiler couldn’t generate performant code. Couldn’t allocate dmem correctly. Perf issues. Serialized all WGMMAs. Nvidia backend heuristic added weights after WGMMA (but wasn’t able to make it do it after a month.)
      * Q\> You were using the lower part of the WS pass in Triton. It wasn’t designed to take any TTGIR and lower correctly (expected it be generated by the existing WS Triton compiler code.) Would be interesting to see if we could make the compiler to take TTGIR and not fail in this pass (when TTGIR is generated).  Should have very few choices for SWP so it should just work. WS isn’t as easy to decompose.

# Recording
* Recording link [here](https://youtu.be/kuBGHVP4M3w)
