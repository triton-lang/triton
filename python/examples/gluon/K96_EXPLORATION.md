# Packed tcgen05 K=96 experiment

## All-K96 Gluon continuation

**Yes: a genuinely all-K96 stream improves MXFP4 beyond the mixed 96+96+64
kernel.** Across the final three-shape sweep, pure K96 adds **1.6–3.7%** with the
same producer/epilogue configuration. Combining it with six producer buffers
and a smaller epilogue tile gives **3.4–6.6%** over the original mixed kernel,
peaking at **7.878 PFLOPS**. NVFP4 is effectively neutral: **−0.5% to +0.3%**
versus its native mixed control. This is a measured prototype, not an exhaustive
optimum or a production compiler interface.

### Packed producer/consumer redesign

The [experimental kernel](experimental-tcgen05-k96.py) consumes K768 with eight
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

## Initial mixed-K96 experiment

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

## Validation and reproduction

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
