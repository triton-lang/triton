# Native packed tcgen05 K=96

## NVFP4 follow-up: 7.935 sustained PFLOPS; 8 remains unmet

The retained native Gluon kernel reaches **7.934985 PFLOPS** at
**M=N=16384, K=16128**, versus **7.652739** for the contemporaneously replayed,
pre-optimization example-07 binary: **+3.69% throughput / −3.56% latency**.
This is the original NVFP4 target case. It did **not** sustain 8 PFLOPS.
A separately disclosed M=N=16384, K=19200, traversal-width16 run reached
7.986153 PFLOPS over seven 500-ms samples; its short 8.019368 screen did not
hold. A subsequent sustained replay of that nearby case measured 7.968620.
Neither a different K nor rounding is used to claim the target.

### Final frozen-binary gate

| Format | M=N | K | Frozen example 07 PFLOPS | New example 07 PFLOPS | Throughput uplift | New/frozen latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NVFP4 | 8192 | 7680 | 7.032437 | 7.269058 | +3.36% | 0.967448 |
| NVFP4 | 16384 | 16128 | 7.652739 | 7.934985 | +3.69% | 0.964430 |
| NVFP4 | 32768 | 32256 | 7.183970 | 7.454270 | +3.76% | 0.963739 |
| MXFP4 | 8192 | 7680 | 7.375524 | 7.386394 | +0.15% | 0.998528 |
| MXFP4 | 16384 | 16128 | 8.104257 | 8.104300 | +0.00% | 0.999995 |
| MXFP4 | 32768 | 32256 | 7.998173 | 8.000901 | +0.03% | 0.999659 |

**All six cases pass the individual ≤1.02 latency gate.** Controls are the
previous example-07 binaries, not the slower original experimental native
kernel. They are hash-verified `CompiledKernel` launches, never recompiled.
Seven alternating samples use identical seed123 inputs on GB300 GPU0
`GPU-4ed02509-778a-be63-cad2-f338cc3fc883`, CUDA graphs, and no L2 flush.
The benchmark's `rep_ms` is 500 for 8K/16K and 750 for 32K. Every sample has
at least 720 device executions; maximum within-case sample coefficient of
variation is 0.147%. Clocks and power limits are unchanged and unlocked.
No candidate is instrumented during performance measurement.

The final source checkpoint is `f24a2f3cfa91fd08914db71962fe3860deea37eb`,
with a clean worktree and one unchanged compiler-library hash throughout the
sweep. The assembler is bundled ptxas-blackwell 13.0.88. The integrated kernel
also reproduces the exact frozen winning scratch implementation within 0.022%
latency (7.932027 versus 7.933748 PFLOPS in their paired recovery run).
Full samples, launch manifests, source/compiler/binary hashes, resources,
replay results, and supplementary experiments are in
[nvfp4-k96-measurements.json](nvfp4-k96-measurements.json). Earlier measurement
files remain unchanged; their results below describe historical phases.

### What changed

No new compiler or language change was needed. The existing typed K96 API
expresses the winner without inline MMA or raw shared addresses. Unsigned
packed FP4, block16/block32 scales, FP32 accumulation, FP16 output, useful
`2*M*N*K` FLOP accounting, and exact K256 TMA producer transfers are unchanged.
Three producer stages still feed eight K96 instructions per K768, without
padding or excess data/scale traffic. This gain combines producer, scheduling,
and epilogue changes; it is **not** an isolated K96-versus-K64 comparison.

NVFP4 now uses six data slots and five independently released scale slots,
48-register load/MMA workers, and a single-stage N16 epilogue. A/B data loads
are issued before waiting for a reusable scale slot, then B scales before A
scales. K128 scale copies are interleaved with native MMA issue; their total
remains 36 `tcgen05.cp` instructions per K768. Independent ring counters persist
across output tiles. The seven native operations are:

| Operation | Data window | Scale-factor start (block16) | K96 instructions | Data release |
| --- | --- | ---: | ---: | --- |
| 1 | slot0 `(0,96)` | 0 | 1 | none |
| 2 | slot0 `(96,192)` | 6 | 1 | none |
| 3 | slot0 → slot1 `(192,288)` | 12 | 1 | slot0 |
| 4 | slot1 `(32,128)` | 18 | 1 | none |
| 5 | slot1 `(128,224)` | 24 | 1 | none |
| 6 | slot1 → slot2 `(224,320)` | 30 | 1 | slot1 |
| 7 | slot2 `(64,256)` | 36 | 2 | slot2 |

Scale-slot completion commits follow the second K128 scale copy for each
slot, after MMA instructions 1, 4, and 6 respectively. Data completion remains
after instructions 3, 6, and 8. The epilogue drains two N128 accumulator halves
early into FP16 registers and releases the accumulator before output staging.
Ordinary tiles issue sixteen N16 output stores. The last SPS tile reuses a
contained view of retired input storage for one collective Gluon store:
**four hardware N64 TMA boxes per CTA**, not one hardware transaction. The
verified final SASS has an unpredicated store after a 128-thread barrier;
all four epilogue warps participate. CLC retains the narrow-store epilogue.

Defaults use SPS for both formats; CLC remains selectable. Traversal width is
16 for K≤16384 and 8 above, with `--tile-width` for explicit experiments. MXFP4
retains its six-slot coupled data/scale pipeline and N32 epilogue. FP32 output
retains the staging byte budget through coupled-ring defaults: six slots/N16
for MXFP4 and five slots/N32 for NVFP4. The measured FP16 NVFP4 binary uses
231,596 shared-memory bytes, 512 TMEM columns, 255 reported registers, and no
spills. No unsupported shared-memory limit override is used.

### Validation and rejected experiments

Passed: **28 functional cases, 18 ConSan pipeline cases, 20 FPSan window cases,
and 14 hardware device-memory checks**. These cover MX/NV scales, FP16/FP32
output, SPS/CLC, independent ring wraparound, M/N tails including the final
SPS reuse path, and CUDA-graph replay. The raw memory-check log separately
records handled `cuFuncSetAttribute` capability-probe errors in the unchanged
loader. The final device-access check uses `--report-api-errors no` and reports
zero device errors; this does not claim the raw loader probes succeeded.
The large CLC/ConSan variant required lengthy LLVM/ptxas compilation, but passed.

All twelve final candidate/control binaries and all 36 original historical
launches pass FP32-reference checks, two direct launches, and ten graph replays
with all Triton compilation entry points patched to fail. Each pair is
bit-identical. The legacy indexed-control CLI and the explicit traversal-width
CLI also pass correctness smoke tests; their short timings are not acceptance
evidence. Every final candidate has complete executable-section SASS coverage,
eight K96 instructions, correct scale selectors, four absolute-leading
continuation descriptors, the expected copy/commit ordering, and no stack or
local-memory spills. Only Python example, benchmark, and existing tests changed;
no native rebuild was necessary.

Rejected experiments remain in the task-owned archive, not dependencies:
N128 tiles/double accumulators (about 6 PFLOPS), extra/compact scale rings,
separate scale-copy warps, direct global stores, paired K512 TMA loads, wider
CTA clusters, XOR traversal, and polling/election/assembler-only PTX changes.
TMEM scale-placement screens appeared slightly faster but gained less than
0.1% in sustained testing and regressed the nearby K19200 case, so were not
retained. Short screens above 8 PFLOPS are retained as screening evidence only.

```bash
python python/examples/gluon/07-pure-k96-matmul.py \
  --format nvfp4 --size 16384 --k 16128 --modes native \
  --frozen-native /path/to/archived/example07/nvfp4-16384/native \
  --repeats 7 --rep-ms 500 --output /tmp/nvfp4-k96
```

## Historical dense example 07: the initial 9-PFLOPS attempt

`07-pure-k96-matmul.py` is a runnable native MXFP4/NVFP4 dense example. It uses only K96 MMAs, unsigned packed inputs, FP32 accumulation, and FP16 output. Three exact K256 producer stages feed each K768 macrotile, without operand padding or extra transfers. The existing native controls and historical measurements are unchanged.

**Observed peak medians: 8.154 PFLOPS MXFP4 and 7.669 PFLOPS NVFP4. Neither reached 9 PFLOPS.** These are end-to-end kernel timings, not issuer-only or compute-only measurements.

| Format | M=N | K | Original native PFLOPS | Example 07 PFLOPS | Throughput uplift |
| --- | ---: | ---: | ---: | ---: | ---: |
| mxfp4 | 8192 | 7680 | 7.391 | 7.448 | +0.77% |
| mxfp4 | 16384 | 16128 | 7.726 | 8.154 | +5.53% |
| mxfp4 | 32768 | 32256 | 7.928 | 8.012 | +1.07% |
| nvfp4 | 8192 | 7680 | 6.972 | 7.075 | +1.49% |
| nvfp4 | 16384 | 16128 | 7.249 | 7.669 | +5.79% |
| nvfp4 | 32768 | 32256 | 7.190 | 7.191 | +0.01% |

All six cases meet the per-case 2% preservation gate; worst latency ratio is 0.999906. Each has seven alternating samples and at least 510 device executions per variant per sample, identical seed123 inputs and the same GPU0 GB300 (GPU-4ed02509-778a-be63-cad2-f338cc3fc883). Timing uses CUDA graphs, unlocked clocks, and no L2 flush. The actual assembler was bundled ptxas-blackwell13.0.88. Source checkpoint: `a1a3186a41d107c19b487c9a852ff5259ea7410c`.

The retained change is tile traversal: width16 for K≤16384, width8 above that, with static persistence except for NVFP4 K>16384, which retains CLC. MXFP4 keeps six producer buffers/N32 epilogue; NVFP4 keeps five/N64. No compiler change was needed.

The exploration retained24 screening reports and70 candidate measurements. Larger issuer register budgets, early/whole-tile accumulator draining, a separate scale warp, independently released scale staging, exact K512 producers, N128 double-buffered accumulators, and four/eight-CTA multicast did not provide a broad improvement over the chosen two-CTA path. Larger cluster grids also exposed second-wave tails: the driver reported76/36/15 resident clusters for2/4/8 CTAs; correcting the grids did not make wider clusters win. Rejected code remains archived, not in the supported example.

A cycle-instrumented diagnostic attributed roughly56–58% of sampled issuer intervals to readiness waits and8–11% to accumulator reuse waits. These are perturbed issuer-local timings, not whole-GPU idle fractions or a promised speedup. The exact baseline NVFP4 profile showed69.5% tensor activity and16.1% HBM throughput.

Validation:20 FP32-reference/graph cases,16 ConSan pipeline cases, and10 hardware memcheck cases passed, with zero memory errors. Coverage includes MX/NV scales, SPS/CLC, odd/even producer rings, ring wraparound, repeated launches, and M500/N600 edge tiles. The expanded API’s earlier FPSan coverage is unchanged; this slice adds no compiler/API semantics. All12 final archived native/control binaries replay correctly without GPU recompilation and produce bit-identical outputs.

All six exact measured candidate binaries pass PTX/SASS checks: eight K96 instructions per K768, release commits after instructions3/6/8, the final accumulator commit after8, correct scale selectors and continuation descriptors, and no stack/local spills. Complete launch manifests, compiler metadata, source hashes, individual samples, and final checks are in `dense-k96-measurements.json` and the task-owned archive.

```bash
python python/examples/gluon/07-pure-k96-matmul.py --format nvfp4 \
  --size 16384 --k 16128 --modes native --compare-native \
  --repeats 7 --rep-ms 500 --output /tmp/dense-k96
```

## Native compiler migration: frozen controls

The immutable source checkpoint is `d93e07b76de305b651df579ce50b64ab7d237618`.
[Launch manifests and replay validation](k96-frozen-controls.json) preserve 16
unique control binaries spanning all 36 recorded launches. Every recorded shape
passes the FP32 reference and ten bit-identical CUDA-graph replays through the
normal `CompiledKernel` launcher, without recompiling the archived controls.
The benchmark accepts `--verify-frozen DIR` and `--frozen DIR`. The native replacement must take no more than 1.02 times the
paired control's median latency at each mixed/pure case. Original measurements
remain unchanged. Binary payloads currently reside in the task-owned
`native-k96/frozen` artifact directory; remote publication is not yet verified.

## Native programming model

The current kernel uses `tcgen05_mma_scaled` directly. The experimental
address-escape operation, inline MMA implementation, and tensor-map encoder
patch have been removed. Existing descriptor views retain their normal
ownership, bounds, layout, and lifetime rules; no new tensor type or general
non-power-of-two layout is introduced.

```python
tcgen05_mma_scaled(
    a0, b0, acc, sa, sb, "e2m1", "e2m1",
    k_range=(192, 288), instruction_k=96,
    a_next=a1, b_next=b1,
    scale_block_size=32,
    a_scale_offset=6, b_scale_offset=6,
    multicast=True, mbarriers=[empty0],
)
```

`k_range` is a half-open interval in logical elements, relative to the first
operand view. Each operand independently continues into its corresponding
`*_next` only when necessary. The backing descriptors remain explicit SSA
operands, including when their ring indices are dynamic. Range metadata is
compile-time. Omitted scale offsets are `k_range.start // scale_block_size`;
explicit offsets address independently staged scale streams. Partial operations
use logical, two-dimensional TMEM scale views. An explicit instruction width
must exactly divide the selected interval; K96 never pads or emits a remainder.

`index`, `slice`, `permute`, and valid `reinterpret` can supply the operand
views. A normalization pass reuses physical-region/provenance analysis to prove
16-byte alignment, K-major non-padded 128-byte swizzling, zero matrix base
offset, scale coverage, and legal physical crossings. A continuation boundary
must coincide with the physical 128-byte K boundary. For example, a tail view
beginning at logical K128 can continue after its remaining K128; a K128 view
beginning at K0 cannot splice another allocation at that same logical length.
LLVM lowering uses only the operation, normalized metadata, types, and converted
SSA operands—never producer traversal or loop-carried-value inspection.

### Automatic mixed selection

Full-operand FP4×FP4 operations automatically use `96+96+64` per K256 when
eligible: sm103, the supported 2CTA/M256 instruction, packed K-major SMEM
operands with 128-byte swizzling, and reduction extent divisible by 256. The
compiler must prove the physical view alignment. Ineligible operations retain
the existing selection. `enable_fp4_k96=False` disables automatic selection;
an explicit `instruction_k` takes precedence. Existing calls retain their
full-operand semantics.

With no completion barriers (`None` or `[]`), old calls remain synchronous.
The small `is_async=True` extension exposes the IR's existing asynchronous
issue mode without a completion barrier; the next completing MMA or explicit
`tcgen05_commit` completes those issues. Operations 1 and 3 below use that flag.
The barrier-count helper accepts continuation descriptors alongside the current
data/scale views, rather than assuming a four-descriptor maximum.

### Original coupled-ring pipeline: five operations per K768

| Operation | Current/continuation view | `k_range` | Scale start | Completion |
| --- | --- | --- | --- | --- |
| 1 | slot0 | `(0, 192)` | `0` | none |
| 2 | slot0 → slot1 | `(192, 288)` | `192 / vec` | release slot0 |
| 3 | slot1 | `(32, 224)` | `288 / vec` | none |
| 4 | slot1 → slot2 | `(224, 320)` | `480 / vec` | release slot1 |
| 5 | slot2 | `(64, 256)` | `576 / vec` | release slot2 |

These issue eight K96 instructions. Each producer still transfers exactly K256
of packed operands and their useful scales. There is no operand padding,
additional copy, excess TMA payload, or changed FLOP accounting. Readiness
waits, scale copies, accumulator handoff, SPS/CLC scheduling, and the TMA
epilogue remain intact. MXFP4 uses six buffers/N32; NVFP4 uses five/N64.
In this original coupled-ring/N64 pipeline, six NVFP4 producer slots require
233,472 bytes, above the 232,448-byte SMEM limit, so its odd/even sanitizer
stress tests use four/five NVFP4 slots and five/six MXFP4 slots. The later
independent-ring NVFP4 pipeline described above fits six data slots with five
scale slots and an N16 epilogue.

### Frozen-binary acceptance

The first native expressibility gate is preserved in commit `faddd74` and
[native measurements](native-k96-measurements.json). Every one of the 18
recorded mixed/pure cases passed: worst median latency ratio 1.00052347 versus
the allowed 1.02. Seven alternating samples used at least 510 device executions
per sample and identical inputs on the same GPU. Peak native MXFP4 pure
throughput was 7.9319 PFLOPS. Instrumented executions are correctness tests only.

The complete post-integration gate at `be23bec20762267d43e3d885e514cd67626628c1`
passes **all 18 cases**, separately: twelve mixed comparisons and six pure-K96
comparisons. The worst median latency ratio is **1.00019780**
(**+0.0198%**, versus the allowed +2%). Seven alternating samples
contain at least **460 device executions per variant per sample**.
All samples, launch/compiler metadata, source and binary hashes, code-generation
checks, and validation receipts are preserved in
[native measurements](native-k96-measurements.json). The initial native gate is
retained there separately; the interrupted seven-case run at `471bfb4` is
historical evidence, not final acceptance.

The final sweep uses the same prepared inputs and GPU
`GPU-4ed02509-778a-be63-cad2-f338cc3fc883` (GB300/sm103), CUDA/ptxas 13.0,
and the unchanged CUDA-graph timing method, without an L2 flush. Clocks were
not locked. The source commit and compiler-library hash are identical across
all twelve reports; the worktree was clean throughout. No candidate was
instrumented. Frozen controls are normal-launcher `CompiledKernel` replays,
never recompiled; every archived file is checked against its hash. Across all
36 historical control points, replayed median latencies differ from the original
records by -1.075% to +1.540%; the largest within-run sample coefficient of
variation is 0.297%. No candidate result is borderline.

Throughput is useful FP4 FLOP/s, in PFLOPS. "Same-producer gain" compares native
pure K96 with the replayed inline-mixed control using the same buffers and
N-size epilogue. Mixed uses the original five-buffer/N64 configuration.

| Format | M=N | K | Native mixed | Native pure | Same-producer gain | Pure/frozen latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MXFP4 | 8192 | 7680 | 7.119 | 7.386 | +3.75% | 0.997476 |
| MXFP4 | 16384 | 16128 | 7.345 | 7.717 | +2.88% | 0.996442 |
| MXFP4 | 32768 | 32256 | 7.393 | 7.923 | +2.13% | 0.993084 |
| NVFP4 | 8192 | 7680 | 6.948 | 6.989 | +0.93% | 0.995932 |
| NVFP4 | 16384 | 16128 | 7.172 | 7.251 | +1.44% | 0.993091 |
| NVFP4 | 32768 | 32256 | 7.183 | 7.186 | +0.43% | 0.995073 |

With the producer and epilogue matched, native pure K96 gains 2.13–3.75% for
MXFP4 and 0.43–1.44% for NVFP4 in this sweep. Peak native pure throughput is
7.923 PFLOPS. The NVFP4 gain is modest; these are measured kernel comparisons,
not an isolated estimate of instruction throughput.

PTX/SASS inspection of every measured native artifact confirms exact K96 widths
and scale selectors, four absolute-leading continuation descriptors in the pure
kernel, eight pure instructions, and slot releases after instructions 3, 6, and
8 (plus the final accumulator completion). Scale copies remain 18 for MXFP4 and
36 for NVFP4 per K768 macrotile. There are no local-memory spills or stack
allocations. Mixed loops contain `96+96+64` groups, including the compiler's
peeled first iteration where present. No adjacent-issue fusion was necessary.

### Compiler and sanitizer validation

- **295 compiler tests pass**, with two unsupported tests, including range
  verification, physical normalization, automatic eligibility/opt-out,
  pipelining, allocation, view/alias, barrier/fence, NVWS, and commit lowering.
- **127 native GPU tests pass**: 81 core correctness/codegen cases, 20 FPSan,
  24 ConSan, and two GSan cases. Coverage includes both scale formats, both
  crossings, independent continuations and scale offsets, scale subviews,
  reinterpretation, nonadjacent views, partial M/N tiles, predication,
  accumulator initialization, ring wraparound, and repeated launches.
- **378 existing targeted GPU regressions pass** (25 skipped), plus **15
  multicast sanitizer tests** (nine skipped).
- **80 NVIDIA memcheck cases pass with zero errors**, on the second isolated
  GB300. API-error reporting is disabled only for the driver's unsupported-SMEM
  capability probes; memory-error detection and a failing error exit code stay on.

FPSan reconstructs the selected reduction from K32 payload chunks and independent
scale streams. ConSan tracks selected K/scale-index regions and all continuation
reads, including replicated TMEM scale words. Positive tests include unused data,
scale, and continuation regions; negative tests cover missing continuation waits,
premature reuse, scale overwrite, and false completion. Persistent tests cover
multiple output tiles, odd/even rings, SPS/CLC, and repeated launches.
Synchronous `None`/empty-barrier compatibility is tested with the peer-CTA
visibility barrier required before cooperative shared-buffer reuse.

Unsanitized graph replay is covered. GSan is tested with repeated ordinary
launches; its runtime does not support graph capture. Instrumented timings are
not performance evidence. These results cover the native FP4 pipelines, not
arbitrary reduction tails, non-power-of-two tensor layouts, or quantized
replacements for gather/attention.

### Publication

Implementation and validation are complete locally. Brix Git uploads still time
out at `PrepareServer`; no GitHub branch update is claimed. The branch has no
upstream PR, and the current handoff helper requires a PR/create entry, so a
branch-only handoff cannot be generated without creating an unrequested PR.
The task-owned archive and local Git bundle preserve the exact implementation
and evidence; their publication status is reported separately.

```bash
python python/examples/gluon/bench-tcgen05-pure-k96.py \
  --format mxfp4 --size 32768 --k 32256 --modes mixed native \
  --buffers 6 --epilogue 32 --repeats 7 --rep-ms 500 \
  --frozen /path/to/native-k96/frozen --output /path/to/results
```

## Historical all-K96 prototype

The measurements and implementation discussion in this section describe the
immutable `d93e07b` prototype. Its source and rejected exact-TMA alternatives
remain archived evidence, not dependencies of the native implementation.

**Yes: a genuinely all-K96 stream improves MXFP4 beyond the mixed 96+96+64
kernel.** Across the final three-shape sweep, pure K96 adds **1.6–3.7%** with the
same producer/epilogue configuration. Combining it with six producer buffers
and a smaller epilogue tile gives **3.4–6.6%** over the original mixed kernel,
peaking at **7.878 PFLOPS**. NVFP4 is effectively neutral: **−0.5% to +0.3%**
versus its native mixed control. This is a measured prototype, not an exhaustive
optimum or a production compiler interface.

### Packed producer/consumer redesign

The archived experimental kernel consumes K768 with eight
K96 instructions fed by three full packed K256 TMA producer slots. It uses an
experimental SMEM/TMEM-address compiler hook and Gluon inline PTX, not CUDA C++.
The original Gluon TMA producer, warp specialization, TMA epilogue, and SPS/CLC
schedulers remain in use. There is no operand padding, shared-memory operand
hole, extra operand copy, or extra operand/scale transfer in this packed path.

| Logical K start | Source slot(s) | Byte offset in first slot | Lifetime action |
| ---: | --- | ---: | --- |
| 0, 96 | 0 | 0, 48 | Retain slot 0 |
| 192 | 0 → 1 | 96 | Release slot 0 after this MMA |
| 288, 384 | 1 | 16, 64 | Retain slot 1 |
| 480 | 1 → 2 | 112 | Release slot 1 after this MMA |
| 576, 672 | 2 | 32, 80 | Release slot 2 after the last MMA |

The two crossing instructions use the descriptor's absolute leading-dimension
address. Corresponding scales are copied into TMEM before either crossing;
scale-word addresses advance by four TMEM columns for A and eight for B. The
scale allocation has a power-of-two logical envelope, but only the useful scale
factors are transferred or consumed. Hardware TMEM allocation remains 512
columns for all final controls and candidates. All MMA instructions in `pure`,
`exact192`, and `exact384` are K96; there is no K64 remainder.

### Final same-input measurements

Hardware: GB300, GPU UUID `GPU-4ed02509-778a-be63-cad2-f338cc3fc883`, CUDA/ptxas
13.0. Clocks were **not locked**. Each point uses seven alternating samples;
the CUDA-graph helper replays each graph ten times. Graph construction budgets
are 500 ms for the first two sizes and 750 ms for the largest. Inputs and
nonuniform bounded scales are identical across variants, with no L2 flush.
K is deliberately divisible by 768: no padded tail is counted as useful work.
The first size uses SPS; the larger two use CLC.

MXFP4 throughput in PFLOPS. Native controls use five producer buffers/N64
epilogue; inline mixed and pure use six buffers/N32.

| M=N | K | Native K64 | Native mixed | Inline mixed control | Pure K96 | Pure/native mixed | Pure/inline mixed |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8192 | 7680 | 6.383 | 7.062 | 7.042 | 7.305 | +3.44% | +3.73% |
| 16384 | 16128 | 6.670 | 7.327 | 7.490 | 7.673 | +4.73% | +2.45% |
| 32768 | 32256 | 6.778 | 7.388 | 7.757 | 7.878 | +6.63% | +1.56% |

The combined gain over K64 is 14.4–16.2%. Do not attribute the whole improvement
to instruction width alone: the matched inline mixed control separates the
pure-stream gain from producer/epilogue tuning. The pure samples' coefficients
of variation are 0.135%, 0.104%, and 0.026%, respectively. Paired comparisons
are more useful than absolute throughput across independently collected runs.

NVFP4 throughput in PFLOPS. All variants use five buffers/N64; six buffers do
not fit with block-16 scales.

| M=N | K | Native K64 | Native mixed | Inline mixed control | Pure K96 | Pure/native mixed |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8192 | 7680 | 6.308 | 6.900 | 6.885 | 6.922 | +0.31% |
| 16384 | 16128 | 6.557 | 7.160 | 7.139 | 7.178 | +0.25% |
| 32768 | 32256 | 6.659 | 7.184 | 7.153 | 7.149 | −0.48% |

There is no broad NVFP4 win beyond the mixed kernel. A separate data/scale-ring
redesign could provide more operand lookahead without six complete scale stages,
but that is an untested next experiment, not an established improvement.

[Raw samples, binary identities, and replay validation](pure-k96-measurements.json)
preserve the complete final sweep and relevant development screens. All six
points use the same experimental source hash, committed in `e4b2d9e`; the first
point was collected before that commit, with its source snapshot preserved.

### Exact K192/K384 transfer alternatives

An alternative uses exact 96-byte operand TMA boxes (K192 packed FP4), with
scales transferred once per K384: twelve MXFP4 or twenty-four NVFP4 factors per
row. `exact384` releases paired K192 transfers together; `exact192` releases each
independently. Neither pads input values, transfers, or MMA arithmetic. Both
reserve unused shared-memory bytes: a 96-byte transfer still occupies a
128-byte-swizzled row. A separate TMA probe checked the actual byte mapping.
The experimental runtime descriptor override changes hardware box sizes and
barrier byte counts while keeping power-of-two compiler-visible allocations.

These variants lost in 7680-cubed MXFP4 screens: **6.48 PFLOPS** for independent
K192 stages and **6.25 PFLOPS** for paired K384 stages, versus **7.39 PFLOPS** for
the packed all-K96 stream. More operand TMA packets, swizzle holes, and less
lookahead are plausible contributors; these screens do not isolate one cause.
Six packed K256 buffers cover K1536, versus K1152 for six exact K192 buffers.
Thus matching every producer stage to an integer number of K96 operations is
not necessary for the best result: packed K768 consumption across producer
boundaries is faster here and still transfers only useful bytes.

The manual mixed path first matched native mixed with identical five-buffer
configurations (7.151 versus 7.157 PFLOPS at 7680 cubed). Moving the lead-CTA
branch outside inline assembly removed avoidable control overhead. A separate
coalesced direct-store epilogue reached only 3.94 PFLOPS and was rejected.
These development screens are not substituted for the frozen-source sweep.

### Validation and limits

**53 focused and existing tests pass**, including 24 pipeline cases spanning
MXFP4/NVFP4, all four experimental modes, partial output tiles, ring wraparound,
SPS/CLC, and repeated CUDA-graph replay. These tests compare FP32 outputs with
FP32 references and verify the K96 bit in every pure-path MMA descriptor.
For all six final performance points, the exact measured raw/pure cubins also
passed ten graph replays and strict FP32-reference checks; their FP16 storage
bits match standalone execution and both native K64/mixed outputs.

CUDA Compute Sanitizer reports zero memory errors for the measured MXFP4 packed
CLC binary and the exact-K192 TMA path. Complete SASS coverage was verified for the measured SPS binary,
including all eight MMA PCs. Inline-PTX memory operations are not fully modeled
by Triton's concurrency sanitizer; the previous native-path sanitizer result
must not be interpreted as coverage of this raw PTX path. The address hook does
not provide general non-power-of-two tensor semantics or a production
pointer-escape lifetime contract. This remains an opt-in research harness.

The packed path requires K divisible by 768; exact-transfer paths require K
divisible by 384. Arbitrary reduction tails and a K288-specific producer were
not implemented. All measured kernels use FP4×FP4, global M/N tiles 256, and
2CTAs on sm103. The other requested examples are assessed below without
silently changing their arithmetic precision.

[Reproduction harness](bench-tcgen05-pure-k96.py):

```bash
PYTHONPATH=python TRITON_BACKENDS_IN_TREE=1 python python/examples/gluon/bench-tcgen05-pure-k96.py \
  --size 8192 --k 7680 --buffers 6 --epilogue 32 --output /absolute/task-owned/path/pure-k96
```

Use `--size 16384 --k 16128` or `--size 32768 --k 32256 --rep-ms 750` for the
larger points. For NVFP4, add `--format nvfp4 --buffers 5 --epilogue 64`.
Exact K192 uses six buffers/N32 for MXFP4 or four/N64 for NVFP4; exact K384 uses
three/N32 or two/N64 respectively. Select them with `--modes k64 mixed exact192`
or `--modes k64 mixed exact384`. Keep all GPU runs on an otherwise idle device.

## Historical initial mixed-K96 experiment

## Result and scope

On a GB300, the opt-in compiler prototype improves the existing dense block-scaled
example by about **8–11%** in the initial interleaved sweep. It consumes the same
packed operands and scales, with the same TMA traffic, shared-memory allocation,
and TMEM allocation. It does not quantize the inputs or pad a K=96 operand to 128.

This is an experimental **96 + 96 + 64** decomposition of each existing K=256
producer tile, not an implementation that uses K=96 for every instruction.
The normal compiler behavior is unchanged unless `enable_fp4_k96=True` is passed.

| Format | M=N=K | Scheduler | K64 PFLOPS | K96 mix PFLOPS | Speedup |
| --- | ---: | --- | ---: | ---: | ---: |
| MXFP4 | 8192 | Static persistent | 6.523 | 7.216 | 10.62% |
| MXFP4 | 16384 | CLC | 6.742 | 7.413 | 9.94% |
| MXFP4 | 32768 | CLC | 6.872 | 7.436 | 8.20% |
| NVFP4 | 8192 | Static persistent | 6.412 | 7.026 | 9.58% |
| NVFP4 | 16384 | CLC | 6.644 | 7.233 | 8.88% |
| NVFP4 | 32768 | CLC | 6.709 | 7.237 | 7.86% |

These measurements use the example's fixed 2CTA configuration: M/N tiles 256,
K tile 256, five producer buffers, N=64 epilogue subtiles. They are not exhaustive
autotuning results. GPU clocks were not locked; runs use the device's normal
power-managed operating state. Absolute throughput is therefore environment-dependent.

A short producer-buffer screen did not improve the original five-buffer choice.
At 16384³ MXFP4, the K96 candidate reached 5.990 PFLOPS with three buffers,
7.024 with four, versus 7.413 with five. At 8192³, the corresponding values were
5.938, 6.934, and 7.216 PFLOPS. Reducing shared-memory use therefore did not
compensate for losing producer lookahead in these cases. Increasing the producer
K tile to 512 with two buffers also lost: 6.072 PFLOPS at 16384³. The original
K=256, five-buffer producer remains the best configuration tested here.

[Raw measurement samples and binary identities](k96-measurements.json) include
the main sweep and both producer-design screens.

## How the operands stay packed

A K=256 FP4 row occupies one 128-byte swizzle sector:

| Byte offset | Bytes consumed | Logical K | Instruction | MXFP4 scale indices |
| ---: | ---: | --- | --- | --- |
| 0 | 48 | 0–95 | K96 | 0, 1, 2 |
| 48 | 48 | 96–191 | K96 | 3, 4, 5 |
| 96 | 32 | 192–255 | K64 | 6, 7 |

No instruction crosses that sector's boundary, so the producer can retain its
128-byte-swizzled packed TMA layout. The second K96 instruction crosses a scale
word; the lowering computes scale addresses and byte IDs from the logical K
offset rather than assuming that an integral number of equal-sized MMAs fits
in a scale word. NVFP4 similarly consumes six block-16 scales per K96 operation.
No scale repacking or additional scale allocation is needed.

This removes 25% of MMA instructions, not 25% of total runtime. Loads, scale
copies, accumulator accesses, stores, scheduling, and synchronization remain.
The observed total gain should not be presented as a 33% or 50% end-to-end win.

## Representative profile

An Nsight Compute full profile of one 8192³ MXFP4 launch per variant used the
same cubins as the main sweep. K64/K96 shared memory was 212,092 bytes per CTA
and TMEM allocation was 512 columns. L2 throughput increased from 53.51% to
64.37% of reported peak, while DRAM throughput increased from 18.60% to 20.80%.
Achieved occupancy stayed around 12.5%; there was no occupancy gain. This is
consistent with the unchanged producer and epilogue becoming more significant
as MMA work gets shorter, but does not establish one exclusive bottleneck.
NCU replay timings are not used in the performance table; clocks were unlocked.

## Which requested examples can use it?

The [PTX matrix-shape specification](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-matrix-shape)
limits native K96 to sm103a, 2CTA, M=256 FP4×FP4 operations. The descriptor's
K96 bit does not add that shape to FP16, BF16, FP8, or mixed FP4×FP8 instructions.

- **Example 04, dense block-scaled matmul:** the FP4×FP4 modes are measured above.
  Its FP8 and mixed modes are not native-K96 candidates. Example 03 is an FP16
  matmul and is not the dense FP4 target.
- **Example 05, fused-gather MoE BMM1:** currently uses FP4 weights and FP8
  activations (`a_type="e2m1", b_type="e4m3"`). Native K96 would require a
  separate FP4-activation algorithm: gather/quantize each activation block,
  produce activation scales, retain packed rather than byte-padded FP4 weights,
  and use eligible 2CTA M=256 tiles. Its gathering, quantization cost, ragged
  expert utilization, and output error must all be measured. The dense gain
  cannot be transferred to it as a measured prediction.
- **Example 01, attention:** QK and PV currently use unscaled FP16/BF16/FP8 MMA.
  Native K96 would require quantized Q/K/V and a quantized, scaled probability
  operand for PV, plus eligible 2CTA tiles. QK's reduction is head dimension
  64 or 128, while PV reduces over sequence tiles. These need different tiling
  decisions; simply changing `BLOCK_N` to 96 does not produce a K96 QK instruction.
  Softmax, causal masking, rescaling, and attention accuracy remain part of the
  algorithm. No same-precision native K96 uplift is claimed for this example.

## Limits of the compiler prototype

The opt-in path supports packed shared-memory operands with 128-byte swizzling,
2CTA M=256 instructions, and power-of-two producer K tiles divisible by 256.
It preserves each operation's full logical reduction. Other operations retain
their existing lowering. It does not add arbitrary non-power-of-two tensor
layouts, cross-operation lowering, or TMEM-left-operand K96 support.

The initial mixed prototype did not implement a fully K96 stream. Such a stream consumes K=768 using eight instructions instead of the
12 baseline K64 instructions (the current mixed decomposition uses nine).
That requires coordinating data and scale lifetimes across producer tiles. The
[absolute-address SMEM descriptor mode](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-leading-dimension-absolute-address)
is relevant where a 48-byte read crosses a 128-byte swizzle boundary. The continuation above implements and benchmarks that producer/consumer redesign.

## Historical validation and reproduction

Commands here record the archived prototype; use the native command above for
current kernels and frozen-binary replay.

The starting revision is `f4fca2c0bc0e19930d2be76bb06ee7b232b60e86`, fetched from
`origin/main` on 2026-08-23. Hardware: GB300, GPU UUID
`GPU-4ed02509-778a-be63-cad2-f338cc3fc883`; CUDA/ptxas 13.0; PyTorch 2.13.0+cu130.

Validation: 29 focused and existing regression tests passed; the 12 new cases
also passed with concurrency sanitizer enabled. Python lint and syntax checks passed.

The focused tests cover MXFP4 and NVFP4, N=128/256, partial M/N output tiles,
K=256/1024, and producer K tiles 256/512. They compare against dequantized
FP32 matmul and verify the generated 4:3 MMA count reduction.

The timing harness prepares the same inputs once for both variants, validates
each with `atol=rtol=1e-3`, warms both, and alternates AB/BA order over seven
CUDA-graph measurements of 200 ms each. It does not flush L2 between launches.
It saves PTX, TTGIR, cubin SHA-256, resource usage, individual samples, and GPU
state. All six main sweep pairs have bit-identical FP16 outputs.

Timing inputs use nonuniform power-of-two scales in {1/8, 1/4, 1/2, 1}. The
original broader scale range is covered in smaller correctness tests. At
16384³, the original scale range produces a few near-zero mismatches against
the FP32 reference in **both** baseline and candidate (25 and 24 of 268,435,456
elements). Targeted FP64 dots show that both the kernel and FP32 reference have
rounding error of roughly 0.0015 there. That stress run was excluded from timing;
the reference tolerance was not relaxed to accept it.

From the repository root, using an isolated environment with CUDA PyTorch:

```bash
make
PYTHONPATH=python TRITON_BACKENDS_IN_TREE=1 python -m pytest -n 8 -s --tb=short \
  python/test/gluon/test_core.py::test_tcgen05_mma_scaled_k96_packed
PYTHONPATH=python TRITON_BACKENDS_IN_TREE=1 python python/examples/gluon/bench-tcgen05-k96.py \
  --sizes 8192 16384 32768 --formats mxfp4 nvfp4 \
  --output /absolute/task-owned/path/k96-results
```

The explicit `PYTHONPATH` prevents an older system Triton from shadowing an
editable checkout in a virtual environment that inherits system packages.
