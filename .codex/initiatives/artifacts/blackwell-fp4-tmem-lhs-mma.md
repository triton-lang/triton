---
owner: jeffniu22@gmail.com
created: 2026-05-07T07:15:19Z
updated: 2026-05-08T19:40:59Z
---

# Blackwell FP4 Padded Weight Packing

## Rationale

`05-moe-bmm1-fused-gather.py` already had an optimized 2-CTA FP8 x MXFP4 ->
FP8 path whose main limiter was MXFP4 weight traffic. Earlier wins came from
swizzling MX scales for TMA coalescing, rearranging densely packed FP4 values in
HBM for TMA coalescing, and feeding weights as the LHS of `tcgen05_mma` so the
dense dimension sits on TMEM rows. That last choice is required when
`BLOCK_M < 64`; for larger `BLOCK_M`, it trades against the transposed
accumulator and can block removal of the epilogue layout conversion into
SwiGLU.

This branch tests a narrower hypothesis: mixed-precision FP4 x FP8 requires the
FP4 operand in `fp4_padded` format, where each 16-byte shared-memory segment
uses only the lower 8 bytes. Instead of leaving half of each weight buffer idle,
pack adjacent K tiles densely in HBM and fetch both with one larger TMA load.
The even tile is already MMA-compatible; the odd tile is register-unpacked with
shift + `prmt`, then stored into TMEM, because TMEM stores from registers are
far cheaper than writing the repacked tile back through shared memory. If
`local_load + unpack + tmem_store` fits under the first MMA / next weight-load
window, the extra tile should be nearly free while effective weight buffering
doubles.

The intended end state is evidence, not just implementation: prove or disprove
that reclaiming the unused half of `fp4_padded` shared memory improves memory
throughput and beats both the prior optimized kernel and the current reference
across the important batch regime.

## Invariants

- `fp4Padded` on `#ttng.tensor_memory_encoding` means padded MMAv5 E2M1 LHS
  byte storage, not a generic FP4 layout flag.
- `fp4Padded` TMEM requires `colStride = 1`.
- Non-scaled `ttng.tc_gen5_mma` must reject `fp4Padded` TMEM LHS.
- Scaled MMA may use TMEM LHS for FP8 formats and for E2M1 FP4; E2M1 storage
  is `i8`.
- Mixed E2M1 LHS with FP8 RHS requires padded FP4 TMEM. Dense FP4 TMEM remains
  the packed FP4 x FP4 path and its K stepping differs from padded FP4.
- `PromoteLHSToTMem` should only produce dense TMEM LHS and must not silently
  reinterpret padded FP4 shared layouts as promotable.
- Slicing, warp specialization, Gluon IR round-tripping, and layout hashing
  must preserve `fp4Padded`.
- TMA `cta_group::2` emission should follow the actual cross-CTA mbarrier
  layout, not a stale module-level two-CTA assumption.
- The experiment only succeeds if the odd-tile unpack path is mostly hidden
  under the even-tile MMA / next TMA load window.
- The weight buffer should be releasable after the odd half has been copied into
  registers, so the next TMA load can overlap odd-tile unpack and MMA2.
- Two-CTA synchronization is part of the performance contract: correctness alone
  is insufficient if the protocol serializes away the reclaimed bandwidth.

## Current Understanding

### Branch Snapshot

- Branch: `codex/fp4`.
- Head inspected: `7a2f2a4893`.
- Merge base with local `origin/main`: `cd8e4acc26`.
- Diff inspected against `origin/main...HEAD`: 27 files, 1,532 insertions,
  359 deletions.
- Working tree was clean before this project document was created.
- `git config user.email` was unset in this checkout; the initiative owner was
  initialized from the HEAD author email.

### Systems Involved

- MLIR attribute surface: `TensorMemoryEncodingAttr` now has `fp4Padded` in
  `include/triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUAttrDefs.td`, and
  `lib/Dialect/TritonNvidiaGPU/IR/Dialect.cpp` verifies it only with
  `colStride = 1`.
- MLIR op verification: `lib/Dialect/TritonNvidiaGPU/IR/Ops.cpp` rejects
  padded TMEM LHS for non-scaled MMA, validates scaled MMA TMEM LHS dtype and
  FP4/FP8 combinations, and adjusts K dimension logic for dense versus padded
  FP4 TMEM.
- Compiler transforms and lowering:
  `lib/Dialect/TritonNvidiaGPU/Transforms/PromoteLHSToTMem.cpp` allows 8-bit
  dense LHS promotion but refuses padded FP4 promotion, and
  `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv5.cpp`
  computes TMEM offsets differently for dense FP4 versus padded FP4.
- Cross-CTA TMA lowering:
  `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/LoadStoreOpToLLVM.cpp`
  now emits `cta_group::2` when the barrier layout is cross-CTA.
- Gluon exposure: `python/src/gluon_ir.cc` and
  `python/triton/experimental/gluon/language/nvidia/blackwell/__init__.py`
  expose `TensorMemoryLayout(..., fp4_padded=True)` through IR construction,
  layout round-tripping, mangling, and hashing.
- Tensor-kernels layout:
  `python/triton_kernels/triton_kernels/tensor_details/layout_details/blackwell_value_packed_shuffled.py`
  adds `BlackwellMX4ValuePackedShuffledLayout`, which packs two FP4 K tiles per
  physical Blackwell TMA/shared-memory tile.
- MoE example:
  `python/examples/gluon/05-moe-bmm1-fused-gather.py` is being adapted to use
  the packed layout, separate weight and weight-scale pipelines, 2-CTA barriers,
  and a two-step first/second K tile MMA flow.
- Prior workload shape:
  the pre-experiment kernel already had the low-batch tuning ladder, scale
  swizzle, densely packed FP4 HBM rearrangement, and weight-as-LHS orientation
  that made 2-CTA fused gather competitive. Those are the baseline to beat, not
  optional cleanup.

### Validation Already Represented In The Branch

- MLIR verifier tests cover invalid padded colStride, dense FP4 TMEM LHS with
  FP8 RHS, wrong padded FP4 storage, and unsupported FP16 TMEM LHS in
  `test/TritonNvidiaGPU/invalid.mlir`.
- MLIR op round-trip tests cover FP8 TMEM LHS, dense FP4 TMEM LHS, and padded
  FP4 TMEM LHS in `test/TritonNvidiaGPU/ops.mlir`.
- LLVM conversion tests cover TMEM LHS lowering and K stepping for FP8 and
  padded FP4 in `test/Conversion/tritongpu_to_llvm_blackwell.mlir`.
- Gluon tests cover scaled MMA LHS-in-TMEM combinations and TMEM padded
  store/load roundtrip in `python/test/gluon/test_core.py`.
- Gluon frontend tests cover `TensorMemoryLayout(fp4_padded=True)` printing in
  `python/test/gluon/test_frontend.py`.
- Tensor-kernels tests cover packed-shuffled layout roundtrip, block shape, and
  column mapping in
  `python/triton_kernels/tests/test_tensor_details/test_layout_blackwell.py`.

### Measured Baseline

- The last pre-experiment optimized kernel is commit `9d9a2d7ac0`; the packed
  pair experiment starts at `2e5acfcf2a`.
- On the same-input CUDA-graph harness, the pre-experiment kernel beats the
  current packed path across the measured sweep:

```text
┌────────────┬──────────────────────┬──────────────────────┐
│ Batch size │ Prepacked baseline    │ Current packed path  │
├────────────┼──────────────────────┼──────────────────────┤
│        128 │ 29.78 TFLOPS / 5.50 TB/s │ 10.40 TFLOPS / 2.00 TB/s │
│      1,024 │ 133.92 TFLOPS / 5.20 TB/s │ 67.96 TFLOPS / 2.75 TB/s │
│      8,192 │ 821.33 TFLOPS / 4.61 TB/s │ 580.53 TFLOPS / 3.26 TB/s │
│     16,384 │ 1,379.49 TFLOPS / 4.13 TB/s │ 894.31 TFLOPS / 2.68 TB/s │
│     30,720 │ 1,942.78 TFLOPS / 3.32 TB/s │ 1,142.59 TFLOPS / 1.95 TB/s │
└────────────┴──────────────────────┴──────────────────────┘
```

- The project has now switched the active kernel default to a 1-CTA
  `BLOCK_N=256`, `4/4/8` pipeline with separate replay partitioning and a
  two-slot odd-LHS TMEM ring. That path passes exact reference comparisons at
  `bs=128`, `bs=1024`, and `bs=8192`, plus twenty repeated same-process
  `bs=1024` launches.
- Representative NCU at `bs=8192` shows the regression is local to the packed
  replay path, not an abstract lack of occupancy:

```text
┌────────────────────────────┬──────────────────────┬──────────────────────┐
│ Metric                     │ Current packed path  │ Prepacked baseline   │
├────────────────────────────┼──────────────────────┼──────────────────────┤
│ Duration                   │             81.98 us │             50.11 us │
│ DRAM throughput            │               25.64% │               40.37% │
│ Compute throughput         │               41.07% │               58.26% │
│ No eligible                │               68.25% │               65.53% │
│ Warp cycles / issued inst  │                15.73 │                11.59 │
│ Long scoreboard stall      │                 8.33 │                 4.29 │
│ Barrier stall              │                 3.24 │                 3.05 │
│ Excess shared wavefronts   │              608,256 │                    0 │
└────────────────────────────┴──────────────────────┴──────────────────────┘
```

- The current odd-tile path contains clustered shared loads before `prmt` and
  TMEM store, and those loads are the leading suspect for the excessive shared
  wavefronts and added scoreboard latency.
- Affine analysis of the replay address map explains that result. For the
  upper-half holes only, the 32-bit shared column is
  `c = 4 * j + 2 + s`, so the bank address keeps one bit fixed:
  `bank = 2 + s + 4 * (j0 xor n0) + 8 * (j1 xor n1) + 16 * (j2 xor n2)`.
  That leaves only 16 distinct banks across a warp, which makes the current
  2-way scalar-LDS conflict pattern optimal for any HBM reordering that still
  stores replay bytes only in the holes. A hole-only `LDS.64` path is worse:
  only eight aligned bank pairs remain, giving a 4-way lower bound.
- Two experiments bounded the escape hatches:
  encoding the replay half as `odd xor even` in HBM made both halves
  semantically necessary, forced ideal `LDS.128` whole-segment loads, and
  removed excessive shared wavefronts, but slowed `bs=1024` single-launch NCU
  duration from `79.392 us` to `83.136 us`; an opaque passthrough helper that
  merely kept the lower half syntactically live still collapsed back to
  `LDS.64`.
- An early scheduling experiment hoisted the unavoidable replay LDS and replay
  side `w_empty_bar` arrival before `mma1`, while keeping PRMT and TMEM store
  where they overlap `mma1`. At `bs=1024`, same-kernel NCU timing improved from
  `79.392 us` to `74.752 us` with the same `473,088` excessive shared
  wavefronts, so the short-window win was latency hiding rather than conflict
  removal. Later repeated-launch testing showed this ordering is not stable at
  larger batch, so it is now rejected rather than retained.
- Lowering `MMA_REGS` from 96 to 80 is the first retained tuning win for the
  packed path. After fixing the custom-config benchmark plumbing so the chosen
  `KernelConfig` is actually passed into `matmul()`, corrected same-input A/B
  reruns still show a clear gain:

```text
┌────────────┬──────────────────────┬──────────────────────┬──────────────┐
│ Batch size │ MMA_REGS=96          │ MMA_REGS=80          │ Bench window │
├────────────┼──────────────────────┼──────────────────────┼──────────────┤
│        128 │ 10.38 TFLOPS / 2.00 TB/s │ 10.85 TFLOPS / 2.09 TB/s │        30 ms │
│      1,024 │ 67.67 TFLOPS / 2.73 TB/s │ 70.78 TFLOPS / 2.86 TB/s │        30 ms │
│      8,192 │ 576.39 TFLOPS / 3.23 TB/s │ 598.82 TFLOPS / 3.36 TB/s │        10 ms │
└────────────┴──────────────────────┴──────────────────────┴──────────────┘
```

- The benchmark helper now accepts an optional `KernelConfig` end to end:
  `prepare_case(..., p=...)`, `run_kernel(..., p=...)`, and
  `benchmark_kernel(..., p=...)` agree on the config being measured. Any
  pre-fix custom-config result should be treated as superseded.
- The 1-CTA TMEM-copy path became a real win only after the replay schedule was
  fixed, not after changing the underlying data motion. The retained schedule
  starts `tcgen05.cp` immediately after `w_ready_bar`, waits for
  `dense_copy_done_bar`, releases the shared weight slot before TMEM
  load/unpack/store, and uses separate replay full/empty barrier rings so the
  two-slot odd-LHS TMEM buffer is actually double-buffered rather than
  serialized by one global completion barrier.
- Same-input 1-CTA A/B with `rep=300` now shows the promoted `4/2/4`
  TMEM-copy default beating the old `4/4/8` LDS path across the measured sweep:

```text
┌────────────┬──────────────────────┬────────────────────────────┐
│ Batch size │ 1CTA LDS 4/4/8       │ 1CTA TMEM-copy 4/2/4       │
├────────────┼──────────────────────┼────────────────────────────┤
│        128 │ 39.46 TFLOPS / 2.09 TB/s │ 44.36 TFLOPS / 2.35 TB/s │
│      1,024 │ 141.10 TFLOPS / 2.02 TB/s │ 160.53 TFLOPS / 2.30 TB/s │
│      8,192 │ 710.90 TFLOPS / 1.65 TB/s │ 842.21 TFLOPS / 1.96 TB/s │
│     16,384 │ 1,031.45 TFLOPS / 1.30 TB/s │ 1,237.55 TFLOPS / 1.56 TB/s │
│     30,720 │ 1,308.49 TFLOPS / 0.97 TB/s │ 1,610.81 TFLOPS / 1.19 TB/s │
└────────────┴──────────────────────┴────────────────────────────┘
```

- The first focused `bs=16,384` pass says the in-register unpack is not the
  dominant gap versus `reference_matmul`. On same-seed 16k runs, the TMEM-copy
  path is already about 20% faster than the LDS replay path, and NCU shows it
  removes the LDS path's `2,849,792` excessive shared wavefronts. The larger
  loss is structural at `BLOCK_M=128`: the custom kernel still keeps FP4 weights
  on MMA LHS, while the reference selector uses the non-swapped orientation for
  `block_m > 64`. That leaves the candidate with roughly 2x the dynamic
  instruction count, 2x the MMA instructions, and far more spill traffic:

```text
┌────────────────────────────┬──────────────────────┬──────────────────────┐
│ Metric                     │ 1CTA TMEM-copy       │ reference_matmul     │
├────────────────────────────┼──────────────────────┼──────────────────────┤
│ Duration                   │            175.78 us │            114.82 us │
│ Executed instructions      │         51,359,825   │         26,069,948   │
│ MMA instructions           │            186,208   │             93,104   │
│ Local spill requests       │          2,415,896   │            107,492   │
│ Long scoreboard stall      │               6.31   │               3.91   │
└────────────────────────────┴──────────────────────┴──────────────────────┘
```

- The next 16k pass showed that the dominant remaining regression was register
  pressure in `load_activations`, not replay unpack latency. With
  `REPLAY_K_SUBTILE_FACTOR=4`, the replay worker can stay at `24` registers.
  Raising `LOAD_ACTIVATION_REGS` from `32` to `72` removes the hot producer
  spill loop, while `80` removes the remaining static and dynamic spill traffic
  without changing the critical `BLOCK_M=128` geometry. The large-slice selector
  now applies `BAND_N=10`, two N-warps in the epilogue,
  `SWIGLU_SUBTILE_FACTOR=16`, `REPLAY_K_SUBTILE_FACTOR=4`,
  `LOAD_ACTIVATION_REGS=80`, `REPLAY_REGS=24`, and `MMA_REGS=24`:

```text
┌────────────────────────────┬──────────────────────┬──────────────────────┬──────────────────────┐
│ Same-seed bs=16,384        │ Prior large-slice    │ LOAD_ACT_REGS=72     │ LOAD_ACT_REGS=80     │
├────────────────────────────┼──────────────────────┼──────────────────────┼──────────────────────┤
│ TFLOPS                     │               906.65 │             1,005.70 │             1,013.49 │
│ Local spill requests       │              907,190 │                63,418 │                    0 │
│ Static LDL/STL ops         │                  110 │                    22 │                    0 │
└────────────────────────────┴──────────────────────┴──────────────────────┴──────────────────────┘
```

- `LOAD_ACTIVATION_REGS=76` is the literal zero-spill threshold, but it loses
  to `72` at `996.50` TFLOPS. `80` is the retained point because it also gives
  zero static `LDL`/`STL` instructions and zero NCU local spill requests while
  improving throughput. A `BLOCK_M=64` zero-spill diagnostic is still useful as
  a shape comparison, but it is slower because the smaller tile increases the
  number of MMA/TMEM operations:

```text
┌────────────────────────────┬──────────────────────┬──────────────────────┐
│ Same-seed bs=16,384        │ BLOCK_M=128 retained │ BLOCK_M=64 zero spill│
├────────────────────────────┼──────────────────────┼──────────────────────┤
│ TFLOPS                     │             1,013.49 │               852.18 │
│ NCU duration               │            90.464 us │           105.824 us │
│ Local spill requests       │                    0 │                    0 │
│ TMEM loads                 │               203,136 │              296,240 │
│ TMEM stores                │               187,424 │              284,576 │
│ MMA instructions           │                97,336 │              148,120 │
└────────────────────────────┴──────────────────────┴──────────────────────┘
```

- After retaining the large-slice tuning, the focused same-seed 16k comparison
  is:

```text
┌────────────────────────────┬──────────────────────┬──────────────────────┐
│ Same-seed bs=16,384        │ Candidate retained   │ reference_matmul     │
├────────────────────────────┼──────────────────────┼──────────────────────┤
│ TFLOPS                     │             1,013.49 │             1,077.22 │
│ TBPS                       │                 2.92 │                 3.10 │
└────────────────────────────┴──────────────────────┴──────────────────────┘
```

- Current 16k conclusion: replay K-subtiling plus producer-register retuning is
  a material win, and the retained `BLOCK_M=128` path now reaches literal zero
  spill traffic. Since the reference gap remains after removing spills, the next
  material experiment should be a large-M path that keeps the non-swapped
  accumulator orientation, which likely means a different way to consume the
  packed odd FP4 tile because today's TMEM replay machinery is specialized
  around FP4-as-LHS.
- A follow-up large-M feasibility pass closed off two tempting shortcuts:
  - `tcgen05_mma_scaled` documentation says RHS TMEM is accepted, but the
    current verifier/effects/lowering only make `A` TMEM-capable; `B` is still
    lowered through the shared-memory loader. A non-swapped path cannot replay
    the odd tile through RHS TMEM without a larger compiler/ISA change.
  - A `BLOCK_N=128` swapped experiment is now legal after making the replay
    register layout shape-aware, but it is slower at 16k because halving N tile
    width doubles the output tile count: `848.15` TFLOPS versus `975.59`
    TFLOPS for the same-seed `BLOCK_N=256` path and `1,102.94` TFLOPS for the
    reference in that run.
- The retained 16k selector has been tightened further while preserving literal
  zero spill: `X_NUM_BUFS=6` and `W_NUM_BUFS=3` improve the packed TMEM-copy
  path from about `996` TFLOPS to about `1,034` TFLOPS in same-seed runs.
  An early cross-process spot check made `MMA_REGS=40` look slightly faster, but
  a controlled same-process, same-input paired comparison supersedes that result:
  `MMA_REGS=24` is faster by about `0.58%`, and `40` reintroduces both static and
  dynamic spill traffic.

```text
┌────────────────────────────┬──────────────────────┬──────────────────────┐
│ Same-seed bs=16,384        │ MMA_REGS=24          │ MMA_REGS=40          │
├────────────────────────────┼──────────────────────┼──────────────────────┤
│ Paired TFLOPS mean         │             1,036.29 │             1,030.26 │
│ Paired TFLOPS median       │             1,036.31 │             1,030.25 │
│ NCU duration               │            90.560 us │            92.160 us │
│ Local spill requests       │                    0 │                8,938 │
│ Static LDL/STL ops         │                    0 │                    4 │
│ Warp latency / issued inst │                13.25 │                13.20 │
│ Long scoreboard stall      │                 5.56 │                 5.52 │
│ Barrier stall              │                 3.34 │                 3.35 │
└────────────────────────────┴──────────────────────┴──────────────────────┘
```

- The extra `16` registers only help the MMA partition marginally: NCU shows a
  tiny reduction in average warp latency and long-scoreboard stall, but the
  hot-loop instruction mix is otherwise unchanged while `40` adds `5,290`
  dynamic local loads and `3,648` dynamic local stores. That small scheduler
  improvement is not enough to pay for the spill path under paired measurement,
  so the zero-spill selector keeps `MMA_REGS=24`.

```text
┌────────────────────────────┬──────────────────────┬──────────────────────┐
│ Same-seed bs=16,384        │ Retained packed path │ reference_matmul     │
├────────────────────────────┼──────────────────────┼──────────────────────┤
│ TFLOPS                     │             1,034.01 │             1,060.89 │
│ NCU duration               │            90.208 us │            74.144 us │
│ Local spill requests       │                    0 │               62,573 │
│ MMA instructions           │               97,336 │               48,668 │
│ TMEM loads                 │              203,136 │                8,464 │
│ TMEM stores                │              187,424 │                3,648 │
│ Barrier stall              │                 3.35 │                 1.40 │
│ Long scoreboard stall      │                 5.56 │                 4.51 │
└────────────────────────────┴──────────────────────┴──────────────────────┘
```

- The current bottleneck to beating the reference is therefore no longer spill
  traffic or register-unpack latency. It is the structural cost of the swapped
  replay design at `BLOCK_M=128`: twice as many MMAs plus the extra TMEM copy /
  load / store pipeline needed to materialize the odd packed tile.
- The deeper 16k profile reinforces that diagnosis. The retained packed path is
  not saturating tensor execution; it spends more time with insufficiently
  eligible warps because replay dependencies and barriers keep the hot path
  waiting, despite issuing far more work than the reference:

```text
┌────────────────────────────┬──────────────────────┬──────────────────────┐
│ Fresh full-profile metric  │ MMA_REGS=24          │ reference_matmul     │
├────────────────────────────┼──────────────────────┼──────────────────────┤
│ Executed instructions      │           25,025,663 │           13,838,894 │
│ Issue active               │               45.25% │               31.23% │
│ Eligible warps / cycle     │                 0.77 │                 0.45 │
│ Tensor pipe active         │               13.85% │               15.72% │
│ Memory-tensor active       │               16.38% │               16.27% │
│ Long scoreboard stall      │                 5.56 │                 4.75 │
│ Barrier stall              │                 3.34 │                 1.42 │
└────────────────────────────┴──────────────────────┴──────────────────────┘
```

- Read together with the unchanged MMA / TMEM op counts, those counters point
  at a latency-and-dependency bottleneck in the replayed swapped design, not a
  register-count bottleneck inside the MMA partition. Raising `MMA_REGS` nudges
  scheduler quality, but the material gap is still the extra replay pipeline and
  doubled MMA/TMEM work versus the reference orientation.
- A source-sampled stall pass says the in-register unpack arithmetic is not what
  is starving the MMA partition today. The direct 1-CTA MMA wait on
  `replay_full_bar` at the handoff before MMA2 accounts for only `38`
  long-scoreboard samples. By contrast, the epilogue wait on `acc_ready_bar`
  contributes `277 + 621` long-scoreboard samples, the activation producer's
  ring wait contributes `278`, and the replay partition's own wait for the
  TMEM dense-copy completion contributes `77`.

```text
┌────────────────────────────┬──────────────────────┬──────────────────────┐
│ Source-sampled wait        │ Inferred role        │ Long-SB samples      │
├────────────────────────────┼──────────────────────┼──────────────────────┤
│ `[... + 0x32120]`          │ `acc_ready_bar`      │              277+621 │
│ `[... + 0x32000]`          │ activation ring wait │                  278 │
│ `[... + 0x32160]`          │ `dense_copy_done_bar`│                   77 │
│ `[... + 0x320f0]`          │ `replay_full_bar`    │                   38 │
└────────────────────────────┴──────────────────────┴──────────────────────┘
```

- That barrier-name mapping is inferred from the SASS sequence and the Python
  allocation order around `mma_partition()` / `replay_partition()`. It is
  consistent with the source structure: `replay_full_bar` is the MMA-side wait
  before the second tile, `dense_copy_done_bar` precedes TMEM load + unpack, and
  `acc_ready_bar` precedes the epilogue TMEM load.
- The unpack instructions themselves do not show a meaningful sampled stall
  signature: all `PRMT` rows together have `0` long-scoreboard and `0`
  short-scoreboard samples, and all `LOP3` rows together have only `1`
  long-scoreboard and `10` short-scoreboard samples. The current limiter is
  broader dependency / synchronization latency around the replayed design, not
  the bit-manipulation sequence by itself.
- A final low-risk 16k selector sweep found no further retained win:
  occupancy above `1` regressed sharply, repeated finalist runs kept
  `BAND_N=10` ahead of `8` and `20`, and the 1CTA `BLOCK_N=512` TMEM-copy shape
  is over the tensor-memory limit before it becomes a viable benchmark point.

- The promoted TMEM-copy default passes exact reference checks at `bs=128`,
  `bs=1024`, `bs=8192`, and focused 16k retune checks, twenty repeated
  same-process `bs=1024` launches, and a
  `TRITON_INSTRUMENTATION_MODE=consan` smoke at `bs=1024`. Its retained
  large-slice config is `BLOCK_N=256`, `NUM_CTAS=1`, `X/W/W_SCALE=4/2/4`,
  separate replay partitioning, `REPLAY_K_SUBTILE_FACTOR=4`,
  `LOAD_ACTIVATION_REGS=80`, `REPLAY_REGS=24`, and `MMA_REGS=24`.
- A half-converted replay-ring attempt deadlocked because per-slot empty
  barriers let the replay producer lap a single global `replay_done_bar`. The
  correct 1-CTA protocol needs both per-slot full and per-slot empty barriers;
  once both rings were present, the promoted path cleared the bounded launch and
  repeated-correctness gates.
- A follow-up attempt to copy only the odd replay words through TMEM by slicing
  the rank-5 shared view and reshaping it to a smaller 2D descriptor failed at
  compile time: `memdesc_reshape` hit the shared-linear-layout size invariant
  before codegen. That sparse-copy direction needs a real descriptor/layout
  representation, not a reshape trick.
- A same-seed historical 2-CTA benchmark is still not a trustworthy comparator:
  the old 2-CTA specialization produced bounded results at `bs=128` and
  `bs=1024`, then stopped making progress at the next size and was killed under
  the hang policy. The current 1-CTA direction is the only path considered
  complete in this slice.

## Assumptions and Risks

### Assumptions

- The main product goal is to prove whether denser HBM packing plus odd-tile
  TMEM replay can beat the earlier optimized 05 kernel by improving effective
  weight-load throughput.
- Compiler/Gluon support exists to make the workload experiment legal; it is
  necessary infrastructure, not the success metric by itself.
- The added `profile1.ncu-rep` through `profile4.ncu-rep` files are local
  profiling artifacts unless the user explicitly wants them versioned.
- The local `origin/main` may be older than the remote default branch because no
  fetch was performed during this read-only discovery.

### Risks

- `verifyAsyncTMALoadOp` has `verifyTMABarrierLayout` commented out in
  `lib/Dialect/TritonNvidiaGPU/IR/Ops.cpp`; the branch needs either a
  replacement verifier or a documented reason this check is obsolete.
- The packed-shuffled layout test expects a row-phase XOR column mapping, but
  `_tma_column_indices` currently appears not to include row phase. A local
  attempt to make code match the test broke live kernel correctness, so the
  test expectation and actual kernel contract need reconciliation before PR
  cleanup.
- There is no direct lit test yet proving the new TMA `cta_group::2` emission is
  driven by cross-CTA mbarrier layout.
- `pack_and_unpack()` in the MoE example looks like a useful smoke test for the
  two-K-tile layout, but it is currently a manual helper rather than a pytest.
- `RaggedTensorMetadata.block_sizes_log2()` now includes 256 for all backends,
  not only HIP; this may affect CUDA metadata scheduling beyond the MoE path.
- The branch adds four binary Nsight Compute reports totaling about 10 MB; these
  are likely inappropriate for a clean PR unless explicitly needed.
- The retained 1CTA config is now tuned enough to beat the retained LDS
  comparator, but PR-readiness claims still need CUDA-graph benchmark evidence
  with `rep >= 300`.
- The current path may pay synchronization overhead twice: once to make the
  multicast/cross-CTA weight load visible before MMA1, then again to manage the
  odd-tile TMEM lifetime before MMA2.
- The retained 1CTA TMEM-copy default now passes a `consan` smoke at
  `bs=1024`; older 2CTA experiments had reported scale-copy warnings, so keep
  treating sanitizer output as protocol-specific rather than assuming one path
  proves another.
- A naive removal of `load_sync_bar` improves raw throughput by about 5%, but it
  is not a valid candidate: `consan` reports an odd-half read-before-write and
  `test_op[128-c0]` produces large mismatches.
- A follower-only `load_sync_bar` signal using a predicated count-2 arrive hung
  on the first bounded launch, so it is not a drop-in replacement for the
  symmetric 2-CTA protocol.
- A simple double-buffered odd-tile TMEM attempt caused 2-CTA hangs, and merging
  the scale producer into `load_weights` regressed large-batch performance.
- With the corrected config runner, `W_NUM_BUFS=4` is not a usable candidate:
  `bs=1024` times out under bounded CUDA-graph benchmarking while the retained
  `W_NUM_BUFS=5` path completes. `W_NUM_BUFS=6` exceeds shared-memory capacity.
- `MMA_REGS=64` and `MMA_REGS=88` also timed out under bounded benchmarking;
  `72`, `96`, and `104` completed but lost to `80`.
- Long CUDA-graph windows at larger batch sizes can still time out even when a
  single launch and a short graph complete. Source inspection did not reveal a
  cumulative phase leak; treat this as an unresolved reliability/perf-debug item
  before making broad benchmark claims.
- Two odd-load layout experiments were rejected:
  a K-heavy load followed by register layout conversion regressed `bs=1024`
  from about `70.8` to `62.8` TFLOPS, and an alternate TMEM-store-legal layout
  failed bounded single-launch checks.
- A direct shifted `fp4_padded` shared-memory view into the upper half parsed,
  but bounded single-launch runs hung at both `bs=128` and `bs=1024`; treat it as
  hardware-illegal or at least unsafe until proven otherwise.
- A full-segment replay path is structurally possible only when the replay tile
  really depends on both halves. The `odd xor even` HBM encoding proved that by
  producing ideal `LDS.128` loads with zero excessive shared wavefronts, but it
  is currently slower than the scalar replay path because it doubles the local
  load footprint and adds decode work.
- An opaque lower-half passthrough inside inline asm did not defeat DCE: SASS
  still returned to hole-only `LDS.64`.
- Hoisting replay LDS even earlier, before `load_sync_bar`, remained correct in
  focused tests but measured slightly worse than the now-rejected post-sync
  hoist at `bs=1024` (`75.648 us` versus `74.752 us` in NCU).
- Moving second-tile scale-copy setup ahead of the odd-tile TMEM store regressed
  `bs=1024` from about `70.8` to `67.3` TFLOPS.
- The older conclusion that 1 CTA was only a fallback has been superseded by the
  split-replay fixes and the retained `4/2/4` TMEM-copy retune above.
- For `BLOCK_M >= 64`, keeping weights as LHS may preserve the memory-throughput
  advantage but lose epilogue layout-conversion savings. Orientation should be
  treated as a tunable regime choice, not dogma.
- Added `ttg.memdesc_to_i32` plus Gluon `shared_memory_descriptor.to_i32()` /
  `tensor_memory_descriptor.to_i32()` so inline PTX can consume raw shared/TMEM
  addresses without descriptor DCE. Shared descriptors now lower to their raw
  shared address rather than a 24-bit CTA-local offset; stripping those upper
  bits was the reason the early 2CTA raw-LDS path faulted only on the follower
  CTA. Buffer-region analysis now explicitly accepts this pure address
  extraction op, so ConSan can compile kernels that use it.
- The corrected 2CTA affine replay map is
  `32 * n_local + 4 * (seg ^ (n_local & 7)) + 2 + word`, with
  `n_local = n & 255`. With raw-address lowering fixed, inline raw LDS passes
  bounded single launches at `bs=128` and `bs=1024`, but repeated `bs=1024`
  launches still hang with one CTA at `load_sync_bar` and its peer at
  `mma_done_bar`. Local post-wait barriers, MMA multicast, and a pair of
  leader/follower handoff sketches did not make the 2CTA raw-inline path
  deterministic, so it is explicitly rejected rather than treated as a
  candidate.
- The replay-load hoist that had looked promising in a short profile is not
  stable at larger batch: `bs=1024` times out while the original post-MMA replay
  ordering completes. Keep the old ordering until the ownership protocol is
  reworked.
- The affine model also explains a practical codegen rule: putting the two
  hole-half words in register bases lets LLVM fuse them into `LDS.64`, which has
  the 4-way aligned-bank lower bound. Restoring the older lane/register basis
  keeps replay as scalar `LDS` (`ld.shared.b32` in PTX), matching the hole-only
  2-way lower bound. That improves the raw experiment, but it does not resolve
  the separate 2CTA completion-handshake bug above.
- The retained active 2CTA path has been restored to the local-load replay
  implementation. It now passes bounded `bs=128` and `bs=1024` launches, exact
  reference checks at both sizes, and twenty same-process `bs=1024` launches.
  Short CUDA-graph measurements on that retained path are `0.039983 ms` at
  `bs=128` and `0.056991 ms` at `bs=1024`.
- The separated replay partition was not fundamentally wrong in 1CTA; it was
  missing odd-tail ring advancement when `K_TILES=23`. After keeping the replay
  cursor aligned across the unpaired final tile, the 1CTA split path passes a
  bounded launch, exact `bs=128` reference comparison, five same-process
  correctness iterations, and a clean 1CTA `consan` run.
- The 2CTA split replay path is now correctness-complete as an experiment. The
  missing edge was a local replay-done handoff: each CTA's MMA worker signals
  its local replay worker after `load_sync_bar`; each replay worker stores to
  TMEM and signals a local `replay_done_bar`; then the MMA worker performs the
  existing cross-CTA `unpack_sync_bar`. It passes bounded `bs=128` and
  `bs=1024` launches, exact reference checks for both sizes, and repeated
  same-process launches, but it measured only `4.37 TFLOPS / 0.84 TB/s` at
  `bs=128` and `bs=1024` CUDA-graph benchmarking timed out, so it is not retained
  as the 2CTA fast path.
- The true two-slot completion-ring experiment is now bounded more precisely.
  The first attempts were invalid because they waited on phase `1` before either
  replay slot had ever been produced. After fixing initial-empty handling and
  adding a matching two-slot replay-done ring, the experiment still deadlocked:
  both replay workers waited on slot completion while both MMA workers waited
  for the next produced slot. Ordinary batched completion barriers deadlock;
  batched `two_ctas=True` completion barriers are rejected by the verifier
  (`completion barrier cga_layout must be [[1]], got [[0]]`). The current scalar
  completion barrier plus two replay TMEM slots remains the only working 2CTA
  protocol in user code today.
- Remaining measured dead ends this pass: double-buffered 2-CTA completion
  still hung, and odd-only sparse TMEM replay still lacks a legal descriptor
  representation. The dense-TMEM replay route itself is no longer a dead end in
  1 CTA; after schedule repair it is the retained fast path.

## Plan

### Phase 1: Reconstruct The Baseline

- [x] Create the project source of truth.
  - Artifact: `.codex/initiatives/artifacts/blackwell-fp4-tmem-lhs-mma.md`.
  - Validation: branch discovery from `git diff`, `rg`, `nl`, and subagent
    read-only analysis.
- [x] Identify the last pre-experiment optimized 05 kernel and preserve it as a
  benchmark baseline.
  - Artifact: commit reference plus same-input A/B harness.
- [ ] Decide whether to remove or explicitly justify `profile*.ncu-rep`.
  - Artifact: cleanup commit or handoff note.
- [x] Build a correctness gate that catches hangs, output drift, and repeated
  nondeterminism before any performance iteration.
  - Artifact: bounded single-call runner, repeated correctness run, and optional
    `TRITON_INSTRUMENTATION_MODE=consan` lane.

### Phase 2: Measure The Current Hypothesis

- [x] Build Triton before running tests.
  - Artifact: successful `make`.
- [x] Verify and, if needed, fix `BlackwellMX4ValuePackedShuffledLayout`
  column mapping.
  - Artifact: passing focused pytest for
    `python/triton_kernels/tests/test_tensor_details/test_layout_blackwell.py`.
- [x] Benchmark current candidate versus prior optimized 05 and reference on the
  same prepared inputs.
  - Artifact: broad batch sweep plus isolated high-rep A/B reruns.
- [x] Profile one representative case to answer where the current candidate is
  losing: TMA issue rate, barrier stalls, odd-tile unpack path, TMEM store path,
  or epilogue conversion.
  - Artifact: NCU report, TTGIR/PTX/SASS notes.

### Phase 3: Optimize The Pipeline

- [x] Retune kernel hyperparameters for the packed pipeline.
  `MMA_REGS=80` is retained after corrected A/B measurement; the active default
  is now 1 CTA with `BLOCK_N=256`, `4/2/4` buffers, and TMEM-copy replay.
  - Artifact: benchmark matrix over block sizes, buffer counts, warp splits,
    register caps, and 1-CTA versus 2-CTA variants.
- [x] Tune instruction scheduling and barrier ownership for the retained 1-CTA
  path. The winning schedule hoists `tcgen05.cp`, releases shared memory before
  unpack/store, and uses paired replay full/empty rings so the odd-LHS TMEM
  buffer is genuinely double-buffered. The 2CTA split-replay ownership model is
  still rejected on performance, and the true 2CTA completion-ring variant
  remains blocked by current completion-barrier semantics.
  - Artifact: measured experiments for early W release, alternate arrival
    signaling, cross-CTA sync placement, and odd-tile replay ordering.
- [x] Try alternate decomposition when the shared partition is the limiter.
  The 1CTA separated replay worker plus dense-TMEM replay is now the active
  retained path; the true 2CTA double-buffered completion-ring variant is
  blocked by current completion-barrier semantics rather than by an unfinished
  kernel protocol.
  - Artifact: measured experiments for separate unpack partition, dense-TMEM
    replay, double-buffered replay TMEM, and 1-CTA variants.
- [ ] Keep only changes that beat the prior optimized kernel broadly and survive
  repeated correctness runs.
  - Artifact: retained patch set plus dead-end log.

### Phase 4: PR Readiness

- [ ] Run lit tests for changed compiler paths from the build directory.
  - Artifact: command output for affected files under `test/TritonNvidiaGPU`,
    `test/TritonGPU`, and `test/Conversion`.
- [ ] Run focused pytests for Gluon and tensor-kernels changes with xdist.
  - Artifact: command output with pass/fail status.
- [ ] Run hygiene checks.
  - Artifact: `git diff --check` output and any relevant formatting/build
    checks.
- [ ] Prepare final branch cleanup.
  - Artifact: worktree with only intended source, test, and project artifacts.

## Execution Log

- `2026-05-07` Completed: created this project after branch discovery.
  - Artifact: `.codex/initiatives/artifacts/blackwell-fp4-tmem-lhs-mma.md`.
  - Validation: inspected branch status, commits, diff stats, changed files,
    compiler diffs, Python/Gluon diffs, tensor-kernels layout, tests, MoE
    example, and binary profile artifacts. No build or test was run because this
    task only created the project source of truth.
  - Learnings: the branch is coherent around FP4/FP8 scaled MMAv5 with TMEM LHS,
    but has PR-readiness risks around barrier verification, packed-layout column
    mapping, binary profiling artifacts, and automated MoE smoke coverage.
  - Plan updates: next work should start with correctness and cleanup before
    any performance tuning.
- `2026-05-07` Completed: reframed the initiative from generic TMEM support to
  the actual performance hypothesis supplied by the branch author.
  - Artifact: updated initiative rationale, invariants, and work plan.
  - Validation: author-provided design intent reconciled with current
    `05-moe-bmm1-fused-gather.py` implementation.
  - Learnings: success means reclaiming the idle half of `fp4_padded` shared
    memory without paying that gain back in unpack or synchronization overhead;
    the earlier optimized 05 kernel is the benchmark that matters.
  - Plan updates: baseline reconstruction and same-input measurement now precede
    compiler cleanup work.
- `2026-05-07` Completed: reconstructed the historical baseline and measured the
  current conjecture against it.
  - Artifacts: detached comparator worktree at commit `9d9a2d7ac0`, same-input
    CUDA-graph A/B harness, `/tmp/current_packed_bs8192.ncu-rep`, and
    `/tmp/prepacked_bs8192.ncu-rep`.
  - Validation: `make`; focused packed-layout pytest; bounded single-launch
    candidate run; repeated correctness at `bs=128`; same-input benchmark sweep;
    representative NCU/SASS read.
  - Learnings: the packed path is currently a large regression, with the odd-half
    replay path producing `608,256` excessive shared wavefronts and roughly
    doubling long-scoreboard pressure versus the prior optimized kernel.
  - Plan updates: generic retuning is now lower priority than reducing odd-half
    replay cost and preserving correctness while simplifying readiness sync.
- `2026-05-07` Completed: evaluated several first-pass experiments.
  - Artifacts: retained 1-CTA layout generalization in
    `python/examples/gluon/05-moe-bmm1-fused-gather.py`; reverted local
    experiments for odd-TMEM double buffering, merged scale production, and
    unsafe W-readiness sync removal.
  - Validation: candidate launches, focused correctness, same-input benchmark
    reruns, and `TRITON_INSTRUMENTATION_MODE=consan` on the unsafe sync variant.
  - Learnings: 1-CTA is a viable tuning axis but not a win yet; the simple
    double-buffer approach deadlocked 2-CTA; removing `load_sync_bar` helps speed
    but violates visibility ordering.
  - Plan updates: next work should target the shared-load access pattern for the
    odd tile, then revisit W-readiness with a correctness-preserving protocol.
- `2026-05-07` Completed: fixed the custom-config benchmark plumbing and reran
  the first tuning sweep against the config actually requested.
  - Artifacts: optional `p` propagation through `prepare_case`, `run_kernel`,
    and `benchmark_kernel`; corrected local config runner scripts under `/tmp`.
  - Validation: focused `test_op[128-c0]`; corrected `MMA_REGS` A/B at
    `bs=128`, `bs=1024`, and short-window `bs=8192`; corrected W-buffer and
    1-CTA checks.
  - Learnings: `MMA_REGS=80` remains a real win; `W_NUM_BUFS=4` is not stable;
    prior custom-config numbers that did not pass `p` into `matmul()` were not
    measuring the intended kernel.
  - Plan updates: keep `80`, discard `4` W buffers, and treat subsequent config
    work as invalid unless the config reaches both data preparation and launch.
- `2026-05-07` Completed: tested the first odd-load and scheduling alternatives.
  - Artifacts: reverted local experiments for a load-friendly register layout,
    an alternate TMEM-store-legal register layout, and earlier second-tile scale
    setup.
  - Validation: bounded single-launch checks, focused correctness, and
    corrected `bs=1024` CUDA-graph measurements.
  - Learnings: register conversion costs dominate the first load-friendly
    layout; TMEM-store legality alone does not make a layout operationally safe;
    moving scale-copy work earlier does not hide useful latency in the current
    schedule.
  - Plan updates: the next credible frontier is a wider/vectorized odd-half
    local-load design or another way to eliminate the replay path itself, not
    more small reorderings around the current scalar-LDS sequence.
- `2026-05-07` Completed: modeled the replay LDS access affinely and tested the
  remaining layout escape hatches.
  - Artifacts: `/tmp/scalar_baseline_bs1024.ncu-rep`,
    `/tmp/hoist_replay_bs1024.ncu-rep`,
    `/tmp/xor_fullsegment_bs1024.ncu-rep`, and source/SASS inspection.
  - Validation: `make`; focused packed-layout pytest; focused
    `test_op[128-c0]`; three repeated direct correctness runs; bounded
    single-launch checks; NCU source counters and SASS inspection.
  - Learnings: hole-only replay loads cannot beat the existing 2-way scalar
    conflict pattern; whole-segment `LDS.128` requires true semantic use of both
    halves; removing the LDS from the exposed post-issue path is currently more
    profitable than widening it.
  - Plan updates: retire hole-only HBM permutations as a bank-conflict strategy;
    next evaluate a separate replay/unpack partition or deeper overlap without
    relying on the unstable hoist.
- `2026-05-07` Completed: added raw memdesc address support and reran the
  LDS-focused experiment matrix.
  - Artifacts: `ttg.memdesc_to_i32` compiler/Gluon slice; retained scalar replay
    layout; reverted local candidates for replay hoist, split replay partition,
    dense-TMEM replay, double-buffered replay TMEM, and forced raw-LDS address
    probe.
  - Validation: `make`; focused frontend pytest for `memdesc_to_i32`; direct
    conversion check; bounded single launches; five repeated `bs=1024`
    launches; CUDA-graph A/B screens; PTX inspection confirming scalar
    `ld.shared.b32` replay loads; `consan` rerun documenting the pre-existing
    scale-copy warning.
  - Learnings: the current HBM packing hypothesis still has line of sight, but
    the hole-only LDS escape hatch is scalarization, not 64-bit vectorization.
    Dense-TMEM replay avoids LDS but loses too much to extra copy/load work;
    1-CTA removes CGA sync but remains well behind the 2-CTA scalar path.
  - Plan updates: keep the new compiler hook, keep the scalar replay layout, and
    focus future work on a correct cross-CTA ownership protocol or a raw
    `LDS.128` path with a verified affine address formula rather than more
    unsound schedule hoists.
- `2026-05-07` Completed: repaired the 1CTA separated replay worker instead of
  discarding it after its earlier correctness failure.
  - Artifact: retained 1CTA-only `replay_partition` path in
    `python/examples/gluon/05-moe-bmm1-fused-gather.py` with odd-tail ring
    alignment.
  - Validation: `make`; bounded default 2CTA one-shot launches at `bs=128` and
    `bs=1024`; bounded 1CTA one-shot launch plus exact `bs=128` reference
    comparison; five same-process 1CTA correctness iterations; five same-process
    default 2CTA `bs=128` correctness iterations; five default 2CTA `bs=1024`
    kernel-only iterations; 1CTA `consan`.
  - Learnings: with `K_TILES=23`, a replay-only partition must still advance its
    packed-weight ring cursor over the final unpaired tile. The existing tail MMA
    already contributes both weight-buffer releases via its completion barrier
    plus `tcgen05_commit`; adding another manual arrive over-releases the slot.
  - Plan updates: treat the 1CTA split experiment as correctness-complete but
    not yet performance-winning; keep the 2CTA split experiment open because its
    cross-CTA ownership protocol is still invalid.
- `2026-05-07` Completed: revisited the experiments that had previously failed
  by hang or correctness instead of treating them as evidence.
  - Artifacts: raw shared-address lowering for `ttg.memdesc_to_i32`; corrected
    2CTA split replay protocol with local `replay_done_bar`; cuda-gdb captures
    for the raw-LDS and completion-ring deadlocks.
  - Validation: full `make`; focused frontend pytest for `memdesc_to_i32`;
    bounded retained-path launches at `bs=128` and `bs=1024`; exact reference
    checks at both sizes; repeated same-process launches for the retained path;
    bounded and repeated correctness runs for the corrected 2CTA split replay
    experiment.
  - Learnings: early raw-LDS faults came from stripping raw shared-pointer bits,
    not from the affine map; 2CTA split replay needed a local handoff before the
    pair-level barrier; true per-slot completion reuse is not currently
    representable as a working 2CTA user-space protocol with the available
    completion-barrier semantics.
  - Plan updates: keep the stable scalar-completion inline path as the retained
    baseline; treat raw inline replay and batched completion-ring work as future
    synchronization/compiler tasks, not as valid performance candidates.
- `2026-05-07` Completed: drove the remaining failed experiments to a correct
  boundary instead of promoting paths that only passed one-shot launches.
  - Artifacts: post-wait convergence barriers on the retained MMA path;
    ConSan-compatible buffer-region handling for `ttg.memdesc_to_i32`;
    restored local-load 2CTA replay as the active path after rejecting the
    unresolved raw-inline handoff variants.
  - Validation: `make`; frontend pytest for `memdesc_to_i32`; lit coverage for
    `Analysis/test-buffer-region.mlir` and `Conversion/tritongpu_to_llvm.mlir`;
    bounded `bs=128` and `bs=1024` launches; exact reference checks at both
    sizes; twenty same-process `bs=1024` launches; short CUDA-graph timing at
    both sizes; ConSan compile/runtime smoke showing the pre-existing scale-copy
    warning at line 502 rather than a compiler failure.
  - Learnings: the raw-LDS address map is fixed, but the 2CTA inline completion
    protocol is still not correct. One-shot correctness is insufficient here;
    the active path only moved back to "retained" after surviving the repeated
    launch gate.
  - Plan updates: keep the local-load 2CTA path as the active baseline, keep the
    corrected split-replay experiment as the only correctness-complete 2CTA
    alternative, and treat raw-inline replay as rejected until its completion
    handoff is understood.
- `2026-05-08` Completed: turned the 1CTA dense-TMEM replay experiment into the
  retained fast path.
  - Artifacts: promoted `4/2/4` TMEM-copy default in
    `python/examples/gluon/05-moe-bmm1-fused-gather.py`; paired replay
    full/empty barrier rings for the two-slot odd-LHS TMEM buffer.
  - Validation: `make`; bounded single-call checks; exact reference comparisons
    at `bs=128`, `bs=1024`, and `bs=8192`; twenty repeated same-process
    `bs=1024` launches; `TRITON_INSTRUMENTATION_MODE=consan` smoke at
    `bs=1024`; paired `rep=300` LDS-versus-TMEM A/B sweep through
    `bs=30,720`.
  - Learnings: `tcgen05.cp` itself was not the losing idea. The first correct
    version serialized replay behind an unnecessary readiness gate, held the
    shared slot too long, and used one completion barrier despite two TMEM LHS
    slots. Hoisting the copy, releasing shared memory immediately after the copy
    completes, and making the LHS protocol a real two-slot full/empty ring
    converts the no-LDS route into a broad performance win.
  - Plan updates: the retained 1CTA path is now TMEM-copy replay, not LDS replay.
    Remaining work should focus on PR-readiness cleanup and any future 2CTA
    protocol/compiler support, not on re-proving the 1CTA hypothesis.
- `2026-05-08` Completed: drove the 16k spill investigation to literal zero
  local traffic without changing the required `BLOCK_M=128` geometry.
  - Artifacts: retained 16k large-slice selector update in
    `python/examples/gluon/05-moe-bmm1-fused-gather.py`;
    `/tmp/bs16k_f4r24_profile.ncu-rep`,
    `/tmp/bs16k_load72_profile.ncu-rep`, and
    `/tmp/bs16k_la80_profile.ncu-rep`,
    `/tmp/bs16k_bm64_profile.ncu-rep`.
  - Validation: same-input 16k sweeps over replay K-subtiling, register
    budgets, and `BLOCK_M`; exact 16k reference comparisons for the retained
    `BLOCK_M=128` path and the zero-spill `BLOCK_M=64` path; ten repeated exact
    launches for each; NCU source/raw inspection; SASS inspection.
  - Learnings: the large spill regression was in `load_activations`, not the
    odd-tile register unpack. `LOAD_ACTIVATION_REGS=72` plus replay K-subtiling
    cuts dynamic local traffic from `907,190` to `63,418` requests; raising the
    activation partition to `80` removes the remaining spill traffic entirely
    and raises 16k throughput to `1,013.49` TFLOPS. `76` is the zero-spill
    threshold but slower than `72`; `80` is the retained point. A literal
    zero-spill `BLOCK_M=64` shape is correct, but slower at `852.18` TFLOPS
    because it raises MMA/TMEM work materially.
  - Plan updates: the retained `BLOCK_M=128` path is now zero-spill. The next
    high-value experiment remains the large-M non-swapped orientation.
- `2026-05-08` Completed: narrowed the post-spill 16k gap and exhausted the
  remaining low-risk 1CTA selector knobs.
  - Artifacts: retained large-slice `X_NUM_BUFS=6` / `W_NUM_BUFS=3` selector in
    `python/examples/gluon/05-moe-bmm1-fused-gather.py`;
    `/tmp/bs16k_w3x6m24_profile.ncu-rep`;
    `/tmp/bs16k_ref_fresh_profile.ncu-rep`.
  - Validation: bounded one-shot `BLOCK_N=128` experiment after making the
    replay load layout shape-aware; exact 16k reference comparisons; ten
    repeated exact launches for the retained selector; zero static `LDL`/`STL`
    SASS inspection; same-seed selector sweeps; filtered NCU candidate/reference
    profiles.
  - Learnings: extra buffering recovers roughly `38` TFLOPS while preserving
    literal zero spill, but the reference gap remains because the retained path
    still pays 2x MMA issue count and large odd-tile TMEM replay traffic.
    `BLOCK_N=128` is correct but slower, `MMA_REGS=40` is slightly faster but
    violates the zero-spill constraint, and `BLOCK_N=512` is not viable for the
    1CTA TMEM-copy shape because its tensor-memory footprint exceeds hardware
    capacity.
  - Plan updates: stop spending time on launch-shape tuning for the current
    swapped replay design. The next material experiment should be a large-M
    non-swapped dataflow; because RHS TMEM is not implemented today, that likely
    means replaying the odd packed tile through shared memory instead.
- `2026-05-08` Completed: checked the 1CTA MMA-loop barriers in lowered IR.
  - Artifacts: `/tmp/mma24_dump/7ZPHMPMTAR4JDCNMH5TM5AELXQLJJS34SNHF54TV23SZ7ES62NTA/ws_matmul_kernel.llir`;
    `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv5.cpp`.
  - Validation: mapped the explicit source barriers at lines `528`, `611`, and
    `657` to LLIR debug locations `!185`, `!218`, and `!249`, then matched the
    adjacent LLIR barriers at `!186`, `!219`, and `!250` to the following
    `tcgen05_mma_scaled` calls.
  - Learnings: when `tcgen05_mma_scaled` has completion barriers, MMAv5 lowering
    emits its own partition-local CTA barrier before branching into the
    single-issuer MMA block. In the current 1CTA path that produces two adjacent
    `barrier.cta.sync.aligned.count 6, 128` instructions before each MMA: the
    explicit `gl.barrier()` and the lowering-generated MMA barrier. The explicit
    MMA-loop barriers are therefore a real redundancy candidate, not a required
    property inferred only from source-level reasoning.
  - Plan updates: test deleting the explicit pre-MMA `gl.barrier()` calls with
    repeated correctness, ConSan, and same-input 16k timing before deciding
    whether to retain them.
- `2026-05-08` Completed: validated which MMA-loop barriers are actually
  removable and rejected the 1-warp MMA partition experiment.
  - Artifacts: retained source barriers in
    `python/examples/gluon/05-moe-bmm1-fused-gather.py`; 2CTA MMA-partition
    invariant asserted in the same file.
  - Validation: same-input repeated `bs=1024` exact checks; fresh-input repeated
    checks to rule out mutable harness state; `TRITON_INSTRUMENTATION_MODE=consan`
    smoke; exact `bs=16,384` checks; paired same-input `16k` timing.
  - Learnings: the lowering-generated MMA barrier is not a drop-in replacement
    for every source barrier. Removing all three pre-MMA source barriers fails
    repeated same-input correctness. Removing only the second-tile barrier or
    the odd-tail barrier also fails. The first-tile barrier survives the repeated
    gate, but direct A/B timing shows no measurable upside, so the synchronized
    source form stays intact. `MMA_WARPS=1` looked attractive and measured about
    `0.59%` faster at `16k`, but repeated same-input validation was flaky once it
    became the default; the partition still issues `tcgen05.cp ... warpx4`, and
    the 2CTA branch has distributed replay work, so the four-warp MMA partition
    remains the retained configuration.
  - Plan updates: do not revisit the pre-MMA barriers or one-warp MMA partition
    without a new synchronization mechanism or hardware-model evidence that
    explains the repeated-run failures.
- `2026-05-08` Completed: resolved why the second-tile / odd-tail MMA barriers
  are required and why ConSan did not report the failure.
  - Artifacts:
    `/tmp/mma_barrier_baseline_dump/LUYWHSWYSDTU5CR6AQE2QF2WUVHZJPMPMZB3LZGRBO5H245KCMRA/ws_matmul_kernel.{ttgir,llir,sass}`;
    `/tmp/mma_barrier_missing_mma2_dump/MFI3XZF22OQACHPFE64ISI6X7TE6LOQG6FVVK4SXEKZYPTU4J5OA/ws_matmul_kernel.{ttgir,llir,sass}`;
    `/tmp/{base,miss}_{pre,post}_membar.ttgir`.
  - Validation: compared source TTGIR, post-`test-print-membar` TTGIR, LLIR,
    and SASS for the retained kernel versus the single-second-barrier removal
    candidate; cross-checked ConSan’s thread and visibility model against
    `include/triton/Dialect/TritonInstrument/IR/TritonInstrument.md` and
    `lib/Dialect/TritonInstrument/Transforms/ConcurrencySanitizer.cpp`.
  - Learnings: this is a post-wait ownership problem, not a missing data-race
    fence that current membar analysis can infer. In the failing second-tile
    path, `test-print-membar` leaves
    `ttng.wait_barrier %x_ready_bar -> ttng.tc_gen5_mma_scaled` adjacent when the
    source `gl.barrier()` is removed; `ttng.wait_barrier` is not a
    `MemWaitOpTrait`, so the generic post-wait insertion rule in
    `Membar.cpp` does not apply, and the pass only inserts barriers before waits
    for ordinary shared-memory hazards here. LLIR still contains MMAv5 lowering’s one
    partition-local barrier before the single-issuer block, but SASS shows that
    as one `BAR.SYNC.DEFER_BLOCKING 0x6, 0x80`; the retained source form yields
    two back-to-back deferred barriers before the issuer path and is the form
    that survives repeated-run correctness. The required invariant is that the
    empty ring slot must not be released until every MMA warp carrying ring state
    has observed the ready epoch. ConSan misses this because it models one
    logical thread per warp-specialization partition, not the four individual
    MMA warps; its visibility frontier can therefore declare the partition
    synchronized even though one hardware warp can still be late in the wait
    loop while the single issuer advances the ring.
  - Plan updates: treat these source barriers as protocol barriers, not as
    redundant fence cleanup. A future real fix would need either a compiler
    primitive that expresses post-wait partition convergence for single-issuer
    tensor-core ops or finer-grained sanitizer modeling of intra-partition warp
    progress.
- `2026-05-08` In progress: corrected the MMA-barrier diagnosis and narrowed the
  remaining corruption to the 1CTA TMEM-copy replay route.
  - Artifacts: repeated same-input launch sweeps on the original `bs=1024`
    shape; source inspection of `replay_partition()` and the existing
    `dense_replay_tmem` / `replay_tmem` handoff;
    `/tmp/replay_sync_trace_dump/4BIVUIDNA2MJRKKATUCSIPZ23DI3CLAZ4V57VZZVCYY6NFQAAHDQ/ws_matmul_kernel.{ttgir,llir,sass}`.
  - Validation: extending the earlier x20 smoke exposed rare tile-local drift in
    the retained kernel itself. The current replay-side barrier set can pass one
    500-launch same-input sweep, but another identical sweep still failed at
    launch 386 with a tile-local mismatch; the LDS replay route
    (`REPLAY_VIA_TMEM_COPY=False`) remained exact for 300 repeated launches.
  - Learnings: the earlier “one MMA barrier is insufficient, two are required”
    conclusion was a masking symptom, not the root cause. In lowering,
    `ttng.wait_barrier` becomes a per-thread spin loop, while
    `ttng.arrive_barrier` already injects a local barrier before the elected
    arriver and TMEM stores already inject `tcgen05.wait::store` plus a local
    barrier. The extra MMA-side barrier therefore changes timing around an
    existing race; it does not explain the required protocol by itself. The
    remaining confirmed failure class is specific to the TMEM-copy replay route,
    where the single-buffered `dense_replay_tmem` handoff and distributed replay
    work still have at least one missing intra-partition ordering edge. ConSan
    misses this class because it models one logical thread per warp-specialized
    partition rather than the individual replay warps.
  - Plan updates: keep investigating replay-side handoff and reuse ordering.
    Treat the replay publication barriers as partial mitigations, not a proven
    final fix. Do not use the second-tile MMA source barrier as evidence of an
    MMAv5 deferred-barrier requirement.
- `2026-05-08` Completed: isolated the actual replay-side correctness edge and
  disproved the earlier "tcgen fences alone fix it" theory.
  - Artifacts:
    `/tmp/replay_sync_trace_dump/4BIVUIDNA2MJRKKATUCSIPZ23DI3CLAZ4V57VZZVCYY6NFQAAHDQ/ws_matmul_kernel.{ttgir,llir,ptx,sass}`;
    `/root/.triton/cache/N4N2HSZ54M2N6FK2VPBIVTZAUSG6YHWCJU4DB5JUDMZF3IO3YFZA/ws_matmul_kernel.{ttgir,llir,ptx}`;
    PTX ISA 9.2 `bar{.cta}` synchronization semantics.
  - Exact compiler gap: Triton can express the replay handoff's ordinary
    `ttng.arrive_barrier` / `ttng.wait_barrier` pair, but it has no source-level
    primitive or lowering path for the **cross-partition replay+MMA rendezvous**
    that this schedule needs before MMA2 consumes the replay TMEM tile. In the
    emitted PTX, replay publication is only
    `tcgen05.wait::st; bar.sync 5, 128; mbarrier.arrive`, and the MMA side is
    only `mbarrier.try_wait...` followed later by its own partition-local
    `bar.sync 6, 128`. No instruction synchronizes the 128 replay threads with
    the 128 MMA threads as one 256-thread producer/consumer group. The existing
    `gl.barrier()` surface also lowers partition-locally, so it cannot express
    this edge from Gluon today.
  - The earlier specialized-fence hypothesis was too broad. Replacing the
    replay-full path with `mbarrier.*.relaxed.cluster` plus
    `tcgen05.fence::{before,after}_thread_sync` made failures rarer but did not
    make the kernel deterministic, so those instructions are not sufficient for
    this bug.
  - Actual no-delay proof:

```text
┌──────────────────────────────────────────────┬──────────────────────────────┐
│ Replay-full handoff variant                  │ Same-input bs=1,024 stress   │
├──────────────────────────────────────────────┼──────────────────────────────┤
│ Baseline handoff, no extra sync              │ failed at launch 62          │
│ Inline `bar.sync 14, 256` on both sides      │ passed 10,000 launches        │
│ Same inline sync restored after A/B removal  │ passed 5,000 launches         │
└──────────────────────────────────────────────┴──────────────────────────────┘
```

    This is a real synchronization experiment, not a delay surrogate: the only
    retained code change is the same named CTA barrier reached after replay
    publication and after the MMA partition observes `replay_full_bar`.
  - Secondary compiler finding from a scratch TMEM handoff reproducer:
    repeated `ttng.tmem_load` operations can be CSE'd across intervening barrier
    waits when the only producer lives in another warp-specialized partition.
    That is separate from the retained kernel fix, but it confirms that
    `ttng.wait_barrier` is not currently a strong enough dependency carrier for
    all TMEM cross-partition protocols.
  - Plan updates: treat the concrete replay bug as a missing cross-partition
    synchronization primitive/lowering path, not as proof that all TMEM handoffs
    need tcgen fences. The compiler follow-up is to add a first-class way to
    express this rendezvous and a regression that checks the required shared
    barrier appears in emitted PTX.
- `2026-05-08` Completed: corrected the replay-root-cause diagnosis with a
  spec-backed no-delay fix and separated two independent edges.
  - Artifacts:
    `/tmp/triton_dense_generic_exact_cache/*/ws_matmul_kernel.ptx`;
    `/tmp/triton_dense_generic_no_before_cache/*/ws_matmul_kernel.ptx`;
    `/tmp/triton_dense_generic_no_after_cache/*/ws_matmul_kernel.ptx`;
    `/tmp/triton_dense_generic_no_replay_sync_cache/*/ws_matmul_kernel.ptx`;
    `/tmp/triton_final_densewait_plus_replaysync_cache/*/ws_matmul_kernel.ptx`;
    PTX ISA 9.2 sections `9.7.16.6.4.2`, `9.7.16.6.4.4`,
    `9.7.16.6.4.5`, `9.7.16.12.1`, and `9.7.13.14`.
  - Exact compiler bug: Triton's generic `ttng.wait_barrier` lowering is wrong
    for the `dense_copy_done_bar` used after `tcgen05.cp`. The producer is a
    `tcgen05.commit...mbarrier::arrive::one.shared::cluster`, but
    `WaitBarrierOpConversion` lowers the consumer to
    `mbarrier.try_wait.parity.shared::cta`. NVIDIA's tcgen memory-model examples
    use a cluster-scoped generic-address wait
    (`mbarrier.try_wait...relaxed.cluster [mbar]`), and the ordinary acquire wait
    is not a substitute for async tcgen synchronization. Replacing only this one
    dense-copy wait with a generic-address `relaxed.cluster` wait removed the
    corruption for 10,000 same-input launches with no delay.
  - Discriminating proof:

```text
┌────────────────────────────────────────────────────────────┬──────────────────────────────┐
│ Variant                                                    │ Same-input bs=1,024 stress   │
├────────────────────────────────────────────────────────────┼──────────────────────────────┤
│ LDS replay path                                            │ passed 10,000 launches       │
│ TMEM-copy path, Triton default dense-copy wait             │ failed at launch 69          │
│ Delay after dense-copy wait                                │ passed 10,000 launches       │
│ Generic `relaxed.cluster` dense-copy wait only             │ passed 10,000 launches       │
│ Generic wait + old replay-full `bar.sync 14,256` retained  │ passed 10,000 launches       │
└────────────────────────────────────────────────────────────┴──────────────────────────────┘
```

  - The replay-full path is a separate protocol edge, not the same compiler bug.
    Removing the retained 256-thread replay+MMA rendezvous still failed at launch
    167 even after the dense-copy wait was fixed. A follow-up discriminator ruled
    out the looser "MMA only needs local convergence" explanation: replacing the
    cross-partition rendezvous with an MMA-only `bar.sync 14,128` still failed at
    launch 70.
  - The direct overwrite story is wrong here. Source-level ownership is already
    explicit: replay waits `replay_empty_bar` before touching `replay_tmem[idx]`
    again, and the second MMA passes that same barrier as its completion signal.
    The "late MMA warp loses the old `replay_full_phase`" explanation is also
    ruled out by the lowered program: in the no-replay-sync artifact, Triton
    emits a 128-thread MMA-local barrier immediately after the
    `replay_full_bar` wait because the following `ttng.arrive_barrier` lowering
    inserts a local barrier before the elected arriver. Replay publication is
    likewise locally converged before `mbarrier.arrive(replay_full_bar)`.
  - What remains proven is narrower: the emitted no-sync program has only
    replay-local and MMA-local barriers; it has no 256-thread replay+MMA
    rendezvous. Adding exactly that cross-partition `bar.sync 14,256` fixes the
    stress test.
  - Follow-up correction: do not treat the successful diagnostic
    `replay_seen_bar` experiment as proof that replay semantically needs a
    consumer acknowledgment before advancing its cursor. LLIR shows that after
    `mbarrier.arrive(replay_full_bar)`, replay only advances local indices; on
    the next loop iteration it first waits on the next `replay_empty_bar` before
    touching the next `replay_tmem` slot. The abstract `replay_full` /
    `replay_empty` ring is therefore sufficient for slot ownership.
  - What the diagnostics actually show is only that extra ordering/timing around
    the handoff suppresses the corruption:
      - `tcgen05.fence::{before,after}_thread_sync` around the existing
        replay-full edge still failed at launch 522.
      - A temporary `replay_seen_bar` wait passed 10,000 launches, but that wait
        also delays replay before its next iteration and does not prove a needed
        protocol edge.
    The retained `bar.sync 14,256` is still an effective workaround, but the
    precise reason remains open. Current evidence is more consistent with a
    missing lower-level ordering requirement or compiler/hardware issue in the
    TMEM handoff than with an abstract ring-buffer protocol bug.
  - Plan updates: keep the local inline generic wait as the application-level
    workaround, but treat the compiler follow-up as a targeted lowering bug:
    introduce a first-class tcgen completion wait that lowers to the generic
    cluster-scoped form and add a regression around emitted PTX. Do not remove
    the replay-full rendezvous until the schedule is redesigned around a real
    cross-partition protocol rather than one-way mbarrier notification alone.
- `2026-05-08` Completed: reduced a separate TMEM handoff compiler bug to four
  loop iterations and ruled it out as the current replay-full explanation.
  - Minimal scratch repro:
    `/tmp/tmem_handoff_cse_repro.py`.
    It has one producer partition, one consumer partition, a two-slot TMEM ring,
    explicit `wait_barrier(..., deps=(slot,))` on both sides, and an inline
    `bar.sync 14, 256` after each handoff. The expected lane-0 sequence is
    `[1, 2, 3, 4]`; the compiled kernel returns `[1, 2, 1, 2]`.
  - Generated evidence:
    `/tmp/tmem_handoff_cse_repro_cache/FGVMDL7BPZK6DDJ25IBQGRQLMFUBVSQV3PP4KQ62ON7FNMBCLYLQ/kernel.{ttgir,llir}`.
    TTGIR contains four `ttng.tmem_store` operations but only two
    `ttng.tmem_load` operations; later stores reuse the first two loaded SSA
    values. LLIR likewise contains eight `tcgen05.st` instructions but only two
    `tcgen05.ld` instructions.
  - Exact compiler issue for this repro: after static unrolling, the consumer
    partition has repeated reads of the same two TMEM slices and no visible
    TensorMemory write in its own region. `ttng.tmem_load` has a real
    `MemRead<TensorMemory>` effect, but `ttng.wait_barrier` only declares
    shared-memory effects; its `deps` operands are not represented as
    TensorMemory effects for optimization. CSE therefore reuses the first load
    from each slot across later barrier generations even though the producer
    partition has overwritten the slots.
  - Why this is separate from the live replay bug:
    the failing MoE kernel's replay and MMA loops remain `scf.for` loops in
    TTGIR, and MMA consumes `replay_tmem` through `ttng.tc_gen5_mma_scaled`
    rather than `ttng.tmem_load`. Adding
    `deps=(p.replay_tmem.index(replay_idx),)` to the live
    `replay_full_bar` wait still failed at launch 310, so this optimization hole
    does not explain the retained `bar.sync 14,256` workaround.
  - Plan updates: keep the four-iteration handoff as the standalone compiler
    reducer for TMEM-load CSE across warp-specialized partitions, and continue
    reducing the live replay failure from the full three-party weight-ring
    protocol instead of using the CSE repro as a proxy for it.
- `2026-05-08` Completed: narrowed the remaining live replay failure to a
  timing-sensitive consume-after-copy edge and disproved the current
  replay-rendezvous explanation.
  - Re-ran the live 1CTA kernel with the retained generic-address
    `dense_copy_done_bar` wait but without the replay-full
    `bar.sync 14,256`. Both-sided `nanosleep.u32 10000`, replay-side-only
    sleep after publication, and MMA-side-only sleep after the full-barrier
    wait each passed 10,000 same-input launches. That means the retained
    cross-partition barrier is not yet proven necessary as synchronization; at
    minimum it also supplies enough delay to hide the bug.
  - Moving the same sleep earlier localized the sensitive window more tightly:
    a single sleep immediately after
    `replay_mma_wait_relaxed_cluster(p.dense_copy_done_bar, ...)` and before
    the first `dense_replay_tmem.load()` also passed 10,000 launches with the
    replay-full barrier removed.
  - Actual synchronization substitutes at that localized point were not enough:

```text
┌────────────────────────────────────────────────────────────┬──────────────────────────────┐
│ Variant                                                    │ Same-input bs=1,024 stress   │
├────────────────────────────────────────────────────────────┼──────────────────────────────┤
│ Sleep after dense-copy wait                                │ passed 10,000 launches       │
│ `tcgen05.fence::after_thread_sync` only                    │ failed at launch 193         │
│ Replay-local `bar.sync 15,128` only                        │ failed at launch 376         │
│ Replay-local barrier + `after_thread_sync`                 │ failed at launch 147         │
└────────────────────────────────────────────────────────────┴──────────────────────────────┘
```

  - Earlier full-path discriminators point the same way: all-producer
    release/acquire publication on `replay_full_bar` failed at launch 471, and
    an all-producer generic `relaxed.cluster` publication plus
    `tcgen05.fence::{before,after}_thread_sync` failed at launch 52. The live
    issue is therefore not explained by the first obvious PTX-level replacement
    for `bar.sync 14,256`.
  - Current strongest statement: immediate consumption of
    `dense_replay_tmem` after the async dense copy completes is unstable in the
    live three-party protocol, but the exact missing ordering edge is still not
    identified. The evidence no longer supports claiming that replay must
    rendezvous with MMA merely to advance local ring state.
  - Current reducer status:
    `/tmp/tmem_replay_chain_repro.py` preserves the dense-copy/unpack path but
    passes without delay because it uses static dense inputs. A first
    three-party reducer,
    `/tmp/three_party_weight_ring_repro.py`, keeps producer/refill, first MMA,
    replay copy/unpack, and second MMA structure, but all modes fail almost
    immediately because the generic shared-store producer is not faithful to
    the full kernel's TMA refill semantics. The next reduction step is a tiny
    TMA-backed producer or another equally faithful way to preserve the real
    weight-buffer handoff.
  - Plan updates: keep the live issue open, pursue a faithful smaller repro from
    the full three-party protocol, and do not promote delay-based explanations
    to root cause without a PTX-spec-backed synchronization proof.
- `2026-05-08` Completed: found a substantially smaller real-kernel reproducer
  and identified odd-tail shapes as a required ingredient for the remaining live
  failure.
  - Reduced live repro:
    `/tmp/replay_sync_tail_repro.py`.
    It uses the production kernel with
    `MLPConfig(hidden_size=17 * 128, intermediate_size=5760)`,
    `batch_size=512`, `seed=0`, and the retained generic
    `dense_copy_done_bar` wait. With `replay_mma_bar_sync()` temporarily reduced
    to a no-op, three fresh trials failed at launches `5`, `4`, and `1`.
    Restoring the retained `bar.sync 14,256` made the same reduced workload pass
    1,000 launches.
  - The full-kernel shrink exposed two useful requirements that the earlier
    scratch reducers missed:

```text
┌────────────────────────────────────────────┬──────────────────────────────┐
│ No-replay-sync live workload               │ Same-input stress result     │
├────────────────────────────────────────────┼──────────────────────────────┤
│ GPT shape, bs=128                          │ passed 1,000 launches        │
│ GPT shape, bs=256                          │ passed 1,000 launches        │
│ GPT shape, bs=512                          │ failed at launch 112         │
│ GPT shape, bs=1,024                        │ failed at launch 6           │
│ 14 / 16 / 22 / 24 K tiles, bs=512         │ passed 1,000 launches        │
│ 15 K tiles, bs=512                         │ failed at launch 453         │
│ 17 K tiles, bs=512                         │ failed at launch 7           │
│ 19 K tiles, bs=512                         │ failed at launch 209         │
│ 23 K tiles, bs=512                         │ failed at launch 93          │
└────────────────────────────────────────────┴──────────────────────────────┘
```

    The failure is not a single-tile steady-state bug: it needs enough outer
    block progression, and odd-tail shapes are a strong trigger. That explains
    why the earlier one-block replay-chain reducers were too small.
  - Also corrected a misleading synthetic lead. In
    `/tmp/tmem_replay_chain_repro.py`, the smallest old failing feature pair
    (`w_pair_first_mma` passing vs. `w_pair_first_mma_use_acc` failing) stopped
    failing once the reducer used the same generic-address
    `relaxed.cluster` dense-copy wait as the live kernel. With that wait fixed,
    both features passed 1,000 launches in `none`, `delay`, and `copy_delay`
    modes. That split was only the already-known dense-copy wait bug, not the
    remaining live issue.
  - Plan updates: future synthetic reducers must preserve the odd-tail
    block-boundary protocol in addition to the generic dense-copy wait. The next
    question is whether the live corruption comes from tail-only cursor
    advancement, tail MMA completion, or the phase relationship created when the
    next block begins after an odd tail.
  - Standalone handoff:
    `python/examples/gluon/replay_sync_tail_race_repro.py` is now the smallest
    faithful one-file repro kept in the repo. It is a reduced copy of the live
    1CTA kernel with the `17 * 128` / `bs=512` workload hardcoded in
    `run_repro()`. After removing unrelated benchmark/test/demo code and all
    reachable 2CTA branches, it still failed at launch `16` with
    `maxdiff=0.40625`.
  - Standalone 2x2 variant check:

```text
┌────────────────────────────────────────────┬──────────────────────────────┐
│ Standalone variant                         │ Same-input stress result     │
├────────────────────────────────────────────┼──────────────────────────────┤
│ Generic dense-copy wait + no-op sync calls │ failed at launch 1           │
│ Generic dense-copy wait + sync calls gone  │ failed at launch 2           │
│ Standard `mbarrier.wait` + sync calls      │ failed at launch 12          │
│ Standard `mbarrier.wait` + calls gone      │ failed at launch 1           │
└────────────────────────────────────────────┴──────────────────────────────┘
```

    Removing the no-op `replay_mma_bar_sync()` calls does not remove the race.
    Replacing the generic dense-copy wait with ordinary `mbarrier.wait` also
    repros, but that reintroduces the already-proven separate dense-copy wait
    lowering bug, so the kept standalone repro continues to use the generic wait
    to isolate the remaining issue.
  - Latest standalone reduction:
    `python/examples/gluon/replay_sync_tail_race_repro.py` is now specialized
    to the fixed 1CTA path only. The known-false gather-reuse, direct-LDS replay,
    selector, non-packed-weight, and large-slice branches are removed; the
    known-true odd-tail, one-fragment replay, TMEM-copy, and `warps_n=1` paths
    are literal code again.
  - Per request, the standalone file now uses direct
    `mbarrier.wait(p.dense_copy_done_bar, dense_copy_phase)` instead of the
    generic inline wait. With that exact source, the focused command
    `PYTHONPATH=python:python/triton_kernels timeout 120s python3 python/examples/gluon/replay_sync_tail_race_repro.py`
    timed out with exit code `124` before printing either `PASS` or `FAIL`.
    This keeps the direct-wait behavior visible, but it no longer isolates the
    later mismatch race because it falls back into the separate dense-copy wait
    lowering problem first.
  - Tree-pruned replay repro:
    after restoring the fast-failing generic-wait baseline from
    `07077510bf`, the file was reduced one node at a time with a stress run
    after every edit and a commit after every retained step. The current head
    keeps the mismatch while removing the fixed epilogue warp branch, gather
    reuse branch, packed-weight config toggle, large-slice selector, dead
    inline-release plumbing, the non-TMEM replay path and its LDS scaffolding,
    the replay subtile-factor loop and plumbing, both odd-tail guards, and the
    dead `mma_done_bar` field.
  - Current retained checkpoint:
    `957bdadfd8` still fails with the focused command
    `PYTHONPATH=python:python/triton_kernels timeout 120s python3 python/examples/gluon/replay_sync_tail_race_repro.py`
    and currently reports `FAIL launch=13 maxdiff=0.4609375`. The exact launch
    count moves as codegen changes, so the validation gate for pruning is
    “still mismatches quickly”, not a specific failing iteration.
  - Follow-on outer-helper pruning:
    later commits kept reducing the same file without touching the core replay
    protocol. The retained cuts removed the dead FP4-init fallback, the
    `prepare_case` reference/config branches, the wrapper helpers around
    `matmul`, fixed output dtype/shape fields in `PreparedCase`, the unused
    `MLPConfig.name` field, and the dead float32 descriptor arm.
  - Latest saved repro head:
    `a824c5eaaa` still fails with the same focused command and most recently
    reported `FAIL launch=25 maxdiff=0.21484375`.
  - Larger blast-radius follow-up:
    a later host-side collapse kept the mismatch without needing to binary
    search any of it back out. Commit `0751e0f528` deletes the standalone
    `MLPConfig` / `PreparedCase` / `prepare_case` / `make_precision_config` /
    `init_routing_data` scaffolding and inlines the fixed repro setup directly
    into `run_repro()`. That head still failed with the focused command and
    most recently reported `FAIL launch=12 maxdiff=0.2734375`.
  - Routing-freeze reduction:
    the current reduction path no longer needs live MoE routing construction to
    trigger the race. The repro now freezes the original failing local routing
    payload as literals:
    `FROZEN_SLICE_SIZES = [2, 11, 1, 2, 1, 0, 3, 7, 5, 13, 1, 0, 1, 0, 3, 4]`
    and the full `2048`-entry gather vector from the same seed. The semantic
    routing path (`make_expt_dict_uniform`, `topk`, local-expert histogram
    construction) is gone from `run_repro()`.
  - RNG-stream dependency:
    freezing routing alone made the repro pass. The missing ingredient was not
    the routing algorithm itself, but the RNG stream it consumed before
    generating `x`, `w`, and `bias`. Restoring only a small RNG burn
    (`torch.randint(0, 8, ...)` plus `make_prod_like_logits(...)`, whose output
    is discarded) brought the mismatch back while keeping top-k and expert
    mapping out of the active repro path.
  - Reduced trigger shape:
    the frozen local schedule sums to `54` rows, and `gather_len == 54`
    repeatedly passed, while `gather_len > 54` could fail. That makes the
    current best hypothesis sharper: the race needs the mismatch between the
    logical gather/output length and the locally scheduled row count, not the
    full live MoE routing machinery.
  - Truncation boundary:
    with the frozen-routing file and the preserved RNG burn, `gather_len=64`
    was still sufficient to fail (`FAIL launch=54 maxdiff=0.41015625`) while
    `gather_len=54` passed for `200` launches. The checked-in repro keeps the
    full `2048`-entry gather payload because it fails sooner (`FAIL launch=20`
    in the same harness), but the now-known smaller failing payload is only ten
    rows beyond the locally scheduled work.
  - Device-path reduction checkpoint:
    two whole device-side generality paths are now proven unnecessary and have
    been removed from the standalone repro source. First, the failure survives
    plain row-major `(pid_m, pid_n)` scheduling, so the banded-`N` scheduler
    path is gone and `apply_block_schedule()` now uses only
    `schedule_pid_m = block_id // GRID_N` and `pid_n = block_id % GRID_N`.
    Second, the failure survives `SWIGLU_SUBTILE_FACTOR=1`, so the epilogue
    subtiling helpers and loop are gone; the repro now emits one direct
    epilogue store per accumulator tile.
  - Current reduced validation:
    after the routing freeze plus RNG burn, the focused command
    `PYTHONPATH=python:python/triton_kernels timeout 120s python3 python/examples/gluon/replay_sync_tail_race_repro.py`
    still fails, and after removing banded scheduling plus epilogue subtiling
    it most recently reported `FAIL launch=1 maxdiff=0.25`.
  - Deeper device-path reduction:
    the current repro no longer carries the frozen gather payload or the
    original matmul wrapper contracts. The activation path now uses contiguous
    `slice_offset + offs_m` rows instead of `gather_indx`, the host side keeps
    only `FROZEN_GATHER_LEN = 2048`, and the kernel interface no longer takes a
    gather pointer. The active epilogue also no longer threads bias, flex
    scales, SwiGLU parameters, or an output descriptor; it directly loads the
    accumulator tile, packs to FP8, and stores with a fixed contiguous packed
    output stride. The host-side `PrecisionConfig` / `FusedActivation` wrappers
    have been collapsed to direct `w_scale` plus fixed `reduction_n = 2`.
  - Schedule metadata reduction:
    the device scheduler no longer unpacks `RaggedTensorMetadata` on the fly.
    Host code now trims the schedule to `grid_m`, precomputes per-block
    `pid_m`, `slice_idx`, `slice_offset`, and `shape_m`, and passes those
    arrays directly into the kernel. The helper now just loads those four
    values plus `pid_n`, and the reduced repro still fails immediately
    (`FAIL launch=1 maxdiff=2.625` after the contiguous-store cut).
  - Fully flattened block schedule:
    the latest shrink removes the remaining `block_id // GRID_N` and
    `block_id % GRID_N` arithmetic from the device. Host code now expands the
    row schedule across all `pid_n` columns into per-`block_id` arrays
    (`block_pid_m`, `block_pid_n`, `block_slice_idx`, `block_slice_offs`,
    `block_shape_m`), so every partition just loads block metadata directly by
    `block_id`. The repro still fails on the first relaunch
    (`FAIL launch=1 maxdiff=2.5`).
  - Direct block metadata:
    the next shrink deletes the remaining schedule helper entirely. The active
    kernel no longer reconstructs per-block row/output metadata from
    `(pid_m, pid_n, slice_idx, slice_offset, shape_m)`. Host code now
    precomputes direct per-block arrays for `block_scale_idx`,
    `block_row_base`, `block_rows`, and `block_out_off_n_packed`, and the
    activation, scale-load, and epilogue partitions load those values directly.
    This version still fails immediately (`FAIL launch=1 maxdiff=2.4375`).
  - Kernel-prelude and helper cleanup:
    the repro no longer derives `k_tiles`, output packed width, or schedule
    geometry inside the kernel entry; those values are now computed on the host
    and passed directly. The activation issue helper and the single-use
    epilogue helpers were also inlined, leaving the file with fewer JIT
    functions and a shallower device call graph. The mismatch still reproduces
    on the first relaunch (`FAIL launch=1 maxdiff=2.625`).
  - Launch-path cleanup:
    the standalone repro no longer carries the `KernelConfig` object. The file
    now uses fixed module-level literals for the block geometry, buffering, and
    warp/register partitioning, and `alloc_randn_fp4()` no longer takes a
    config argument. After fixing the host-side `MXFP_BLOCK_SIZE` integer vs
    constexpr type mixup, the reduced file still repros with
    `FAIL launch=1 maxdiff=2.625`.
  - Single-row / single-slice reduction:
    the active repro no longer needs any ragged schedule state at all. Host
    code now runs a single-row `x` tile against a single logical weight slice,
    the block metadata arrays are gone, and each partition derives its work
    directly from `program_id(0)` plus fixed literals. The outer
    `for block_id in range(...)` loops are gone as well because the launch is
    now the literal two-block grid that exercises the race. This much smaller
    source still fails with the focused command, most recently as
    `FAIL launch=6 maxdiff=0.625`.
  - Fixed-constexpr reduction:
    the next shrink proves the kernel does not need runtime constexpr plumbing
    to reproduce. Buffer counts, tile sizes, replay geometry, and warp/register
    partitioning now live as module-level `gl.constexpr` constants instead of
    flowing through the kernel signature and `PartitionArgs`. The launch site
    now passes only the three descriptors plus `out_ptr`, and the specialized
    source still fails (`FAIL launch=22 maxdiff=0.625`).
  - Current lower bound:
    the standalone repro is now down to a single-row, single-slice,
    single-buffer setup for `x`, `w`, replay, and accumulator state. The MMA
    side no longer consumes dynamic weight scales at all; `w_scale_tmem` is
    filled once with a constant, while a separate one-warp scale partition
    still issues the original TMA traffic into an unused shared buffer. A plain
    dummy `uint8` tensor is enough for that scale descriptor, so the failing
    case no longer depends on real MX scale values or the unswizzle helper.
    The current source is `575` lines and still fails quickly, most recently as
    `FAIL launch=4 maxdiff=1.3125`.
  - Failed shrink boundaries:
    several larger cuts no longer reproduce and therefore mark the current
    boundary. Reducing `K_TILES` from `3` to `2` stopped producing a clean fail
    signal within the 20-second guard. Deleting the scale-side TMA partition
    entirely, or replacing it with only a dummy pacing partition and barrier
    arrivals, made the repro pass for `1000` launches. Replacing the packed-FP8
    epilogue with a direct raw-accumulator row store ran into Gluon indexing
    limits, so the packed store remains part of the smallest working source.
  - Warp-specialization reduction checkpoint:
    the current standalone repro no longer uses the scale copy or epilogue as
    warp-specialized partitions. The one-off scale TMA issue is now inlined in
    the kernel prelude, the FP8 epilogue runs after `gl.warp_specialize()`
    returns, and the specialized tree itself is down to four functions:
    default `load_activations`, plus `load_weights`, `replay_partition`, and
    `mma_partition`. Several pieces of state that used to flow through
    `PartitionArgs` are now partition-local instead: the scale-copy shared
    buffer and ready barrier, the dummy MMA scale barrier, the dense replay
    TMEM scratch, the dense-copy completion barrier, and both constant scale
    TMEM descriptors. This smaller form still fails quickly
    (`FAIL launch=1 maxdiff=1.3125`).
  - New reduction boundaries:
    some additional folds now have precise outcomes. A one-copy scale prelude
    is sufficient as long as it exists before `warp_specialize()`. Merging that
    copy into the weight-loader worker made the repro pass. Folding
    `replay_partition` into the 1-warp weight-loader worker failed at compile
    time because the TMEM load path requires at least four warps. Folding
    replay into the default activation path triggered the 20-second hang guard.
  - Load-partition merge:
    the activation and weight load paths can share the same default partition
    as long as their issue order still preserves overlap. The current lower
    bound issues the first `W` TMA copy before the `X` gather loop, then issues
    the second `W` copy only after waiting on `w_empty_bar` once replay has
    released the buffer. That removes the dedicated 1-warp `load_weights`
    worker from `gl.warp_specialize()`, leaving only the default merged load
    partition plus `replay_partition` and `mma_partition` as active workers.
    This smaller tree still fails immediately (`FAIL launch=1 maxdiff=1.3125`).
  - Failed merged-load shape:
    a simpler merged form that loaded all `X` tiles first and only then started
    issuing `W` traffic did not hold; it tripped the 20-second hang guard.
  - Odd-tail MMA removal:
    the final dense-`W` MMA in `mma_partition` is not required for the race.
    Removing the last `w_ready_bar` wait, the last dense `tcgen05.mma`, and the
    trailing `tcgen05_commit(w_empty_bar)` still leaves a fast repro
    (`FAIL launch=2 maxdiff=0.890625`). The current trigger only needs the
    first dense MMA plus the replay MMA.
  - Epilogue removal attempt:
    replacing the FP8 pack/store epilogue with a raw `float32` accumulator dump
    does not preserve the repro once the output mapping is correct. A first
    cut appeared to fail, but that version incorrectly aliased both `pid_n`
    programs onto the same output region. After restoring the missing `pid_n`
    offset, the raw dump path passed for 1,000 launches. Conclusion: the
    current repro still needs the original packed epilogue; removing it hides
    the race instead of isolating it.
  - Latest accepted shrink batch:
    the repro still fails after removing the third activation tile, the second
    dense `W` TMA copy, the `acc_empty` ring, and the now-unused cross-partition
    `w_empty` protocol. Host tensors can shrink to two K tiles, and the prelude
    scale-side traffic can collapse to a minimal valid rank-2 TMA transfer over
    a `1x16` dummy tensor. Current gate: `FAIL launch=1 maxdiff=0.890625`.
  - Rejected shrink branches:
    reducing the launch grid to one N program passed for 1,000 launches, so the
    second program is still part of the trigger. Likewise, storing only the
    visibly corrupt second program while leaving both programs alive passed for
    1,000 launches; the first program's packed epilogue work is part of the
    timing envelope even though the observed bad values land in the second
    output half.
  - Latest lower-bound collapse:
    the repro no longer needs warp specialization, separate load/replay workers,
    the replay empty/full ring, or the dummy prelude TMA. Replay can run in the
    same function as the loader, then call MMA sequentially, and the failure
    still fires. Geometry also shrank to `BLOCK_M=16`, `BLOCK_N=128`,
    `BLOCK_K=128`; the activation side needs only one masked gather, and the
    replay path can dense-copy into `replay_tmem` then unpack/write back in
    place. Current gate on this reduced witness:
    `FAIL launch=1 maxdiff=25411`.
  - Host-side collapse:
    no MXFP wrapper or layout conversion is required for the witness anymore.
    Raw random `uint8` weights with a rank-2 TMA descriptor are sufficient, and
    the packed epilogue can store `int16` lanes directly instead of re-packing
    into `int32`.
  - Correction after further minimization:
    the post-`8c2b57b6de` shrink path was not a valid witness. `9db7959302`
    replaced the real scale pipeline with a dummy scale TMA that has no
    `mbarrier.expect`, is never waited, and whose barrier storage is later
    reused for an MMA completion barrier. `b983fe67a5` then removed the bounded
    epilogue while still launching two N programs: with `BLOCK_N=128`, each CTA
    writes 64 packed `int16` lanes, `pid_n=1` starts at lane 32, and the row
    stride is also 64. That creates overlapping stores and a tail overrun before
    any later shrinking. The final `f59d0981f6` version compounded that by
    shrinking the host output to `16x32`, so the second launch overwrote the
    adjacent `expected = out.clone()` allocation and manufactured the observed
    `2x32` mismatch. The repro source has been rewound to `8c2b57b6de`, the last
    state before the malformed dummy TMA was introduced and while the original
    masked epilogue was still present.
  - Current minimal core:
    the repro now fails with one CTA, one `tcgen05.mma`, one masked `X` gather,
    one direct nonzero `replay_tmem.store(...)`, and the packed FP8 epilogue.
    Removed since the previous checkpoint: all warp specialization, the replay
    worker, replay barriers, dummy prelude TMA, weight descriptor/setup, dense
    `tcgen05.cp`, replay TMEM load/store roundtrip, register unpack, first dense
    MMA, and the second N program. A zero replay TMEM fill passes, but the
    literal value `1` still fails on the second launch.
  - Fresh lower-bound checks:
    direct register initialization of `x_buf` passes, so the masked TMA gather
    remains live. Raw accumulator stores still pass, so the packed FP8 epilogue
    remains live. Geometry floors are `BLOCK_M=16`, `BLOCK_N=128`,
    `BLOCK_K=128`; `num_warps=4` is the legal floor for the TMEM register layout.
  - Final minimized repro state:
    the standalone file is now 158 lines. The failing kernel contains only:
    one masked `tma.async_gather` into `x_buf`, one nonzero `replay_tmem.store`,
    one scaled `tcgen05.mma`, one accumulator completion wait, and the full
    packed-FP8 epilogue. The harness is one CTA, four warps, two launches, and
    a self-compare. Current gate:
    `PYTHONPATH=python:python/triton_kernels timeout 20s python3 python/examples/gluon/replay_sync_tail_race_repro.py`
    -> `FAIL`.
  - Final rejected reductions:
    all of the following stop reproducing or become illegal: zeroing
    `replay_tmem`; replacing the masked TMA gather with direct SMEM stores;
    replacing the packed FP8 epilogue with raw FP32 stores or direct FP8 stores;
    storing only the first output row or half of the packed columns; shrinking
    below `BLOCK_M=16`, `BLOCK_N=128`, `BLOCK_K=128`, or `num_warps=4`.

## Next Up

- [ ] Decide whether the older 2CTA/local-load path should remain in the branch
  as a comparator or be trimmed before PR cleanup.
- [ ] Prototype the large-M non-swapped accumulator orientation now that the
  16k spill hypothesis has been bounded by the zero-spill experiment.
- [ ] Verify the raw `LDS.128` address formula against the compiler-emitted
  affine map before attempting another inline-PTX replay load.
- [ ] Resolve the remaining PR-readiness questions around verifier coverage,
  packed-layout expectations, and binary profile artifacts.
- [ ] Decide whether future 2CTA work needs compiler support for a real
  completion-ring abstraction instead of more user-space protocol sketches.
- [ ] Add a compiler-level tcgen completion-wait primitive or lowering path that
  emits the generic cluster-scoped wait form for `tcgen05.commit` barriers.

## Open Questions

- Is `BlackwellMX4ValuePackedShuffledLayout` intended as a stable public
  tensor-kernels layout or a temporary workload-specific layout?
  - Owner: `jeffniu22@gmail.com`.
  - Resolution path: validate the MoE path and decide before PR cleanup.
- For the regimes where `BLOCK_M >= 64`, when does weight-as-LHS still beat the
  epilogue-layout advantage of the non-transposed accumulator?
  - Owner: `jeffniu22@gmail.com`.
  - Resolution path: compare both orientations in the same-input sweep after
    the packed path is stable.
- Should `RaggedTensorMetadata.block_sizes_log2()` expose 256 globally on CUDA?
  - Owner: `jeffniu22@gmail.com`.
  - Resolution path: run focused metadata tests and inspect any downstream
    scheduling assumptions.

## Deferred / Out Of Scope

- General RHS-in-TMEM support.
- Non-Blackwell implementations of the packed MXFP4 layout.
- Performance claims based on host-overhead benchmarks.
