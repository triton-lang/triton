# Packed tcgen05 K=96 experiment

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

A fully K96 stream would consume K=768 using eight instructions instead of the
12 baseline K64 instructions (the current mixed decomposition uses nine).
That requires coordinating data and scale lifetimes across producer tiles. The
[absolute-address SMEM descriptor mode](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-leading-dimension-absolute-address)
is relevant where a 48-byte read crosses a 128-byte swizzle boundary. Such a
producer/consumer redesign is not implemented or benchmarked here.

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
