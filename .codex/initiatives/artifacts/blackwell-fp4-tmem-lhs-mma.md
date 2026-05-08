---
owner: jeffniu22@gmail.com
created: 2026-05-07T07:15:19Z
updated: 2026-05-08T03:03:55Z
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
