# Mixed-precision reduction scheduling on Rubin

Measured on 2026-11-24 with the compiler at `e0f8c17084`, on a Rubin GPU
(compute capability 10.7), using CUDA 13.5 `ptxas`.

The benchmark is [07-reduction-schedules.py](07-reduction-schedules.py).
**K** is the number of elements reduced per output per thread. **O** is the
number of independent reductions/outputs per thread, so each thread owns K × O
narrow inputs. The reduction axis is entirely register-resident; there are no
inter-thread reduction shuffles or shared-memory reduction steps.

The measured workload repeats this recurrence for 128 iterations:

```text
narrow = cast_to_fp16_or_bf16(fp32(base) + carry)
carry  = (0.5 / K) * sum(fp32(narrow))
```

`base` remains constant and `carry` depends on the previous iteration. This
prevents loop-invariant reduction hoisting. Both modes use the same inputs,
layouts, launch configuration, and recurrence; only `in_thread_reduction`
changes. This synthetic workload includes conversion and elementwise work
around the reduction and is not an attention benchmark.

## Performance

Latencies are medians across alternating tree/sequential measurement pairs.
Each measurement averages repeated CUDA-graph replays with reused inputs.
Latency reduction is `(tree_latency - sequential_latency) / tree_latency × 100`.
Register counts are allocated registers per thread, reported by the CUDA driver.
“None” means no explicit `maxnreg` cap.

| Type | K / O | Register cap | Tree latency | Sequential latency | Latency reduction | Registers/thread, tree → sequential |
|---|---|---|---:|---:|---:|---|
| FP16 | 32 / 1 | None | 13.58 µs | 11.14 µs | **18.0%** | 59 → 44 |
| FP16 | 64 / 2 | None | 51.99 µs | 38.53 µs | **25.9%** | 168 → 146 |
| BF16 | 64 / 2 | None | 63.71 µs | 38.23 µs | **40.0%** | 166 → 146 |
| FP16 | 64 / 1 | 80 | 29.53 µs | 22.42 µs | **24.1%** | 80 → 74 |

For FP16 K64/O1 with the 80-register cap, reassembling the saved PTX with the
same flags reports 24 bytes each of spill stores and spill loads for tree, and
zero for sequential. Stack frames are respectively 24 and zero bytes.

For FP16 K32/O1, the loop body contains 8 mixed-precision packed adds plus 7
ordinary FP32 packed adds with tree, versus 15 mixed-precision packed adds with
sequential. The mixed instructions are `add.rn.f32x2.f16x2.f32x2` in PTX.
The BF16 variants emit `add.rn.f32x2.bf16x2.f32x2`.

The larger BF16 gain at K64/O2 comes from a slower tree baseline; sequential
latencies are similar for the two types. SASS loop-body instruction counts are
325 → 265 for FP16 and 357 → 267 for BF16. BF16 also benefits from better
FP32-to-BF16 conversion packing: 64 scalar plus 32 packed conversions with tree,
versus 4 scalar plus 62 packed conversions with sequential. Instruction counts
alone do not isolate each instruction sequence's contribution to latency.

## Measurement stability

CV is population standard deviation divided by mean latency, expressed as a
percentage. The range below is the latency reduction calculated separately for
each alternating pair, rather than a confidence interval.

| Configuration | Pairs | Tree CV | Sequential CV | Paired latency reduction |
|---|---:|---:|---:|---:|
| FP16 K32/O1, uncapped | 5 | 0.59% | 0.73% | 17.94–18.26% |
| FP16 K64/O2, uncapped | 5 | 0.003% | 0.004% | 25.883–25.890% |
| BF16 K64/O2, uncapped | 3 | 0.020% | 0.022% | 39.997–40.009% |
| FP16 K64/O1, cap 80 | 5 | 0.001% | 0.015% | 24.069–24.103% |

Sequential won every pair for these four configurations. FP16 K32/O1 showed a
small timing shift affecting both modes together. These are short-run
repeatability measurements, not a prolonged node stability study.

## Reproduction

Build this branch and run from the repository root with GPU access:

```bash
PYTHONPATH=./python python python/examples/gluon/07-reduction-schedules.py \
    --output-dir rubin-reduction-microbench-results \
    --pairs 3 --rep-ms 15 --warmup-seconds 5

PYTHONPATH=./python python python/examples/gluon/07-reduction-schedules.py \
    --output-dir rubin-reduction-microbench-regcaps \
    --dtype float16 --values-per-output 32 64 --outputs-per-thread 1 2 \
    --iterations 128 --maxnreg 0 64 80 96 \
    --pairs 5 --rep-ms 20 --warmup-seconds 3
```

Output directories must not already exist. The script saves raw measurements,
summary CSVs, PTX, cubins, and metadata including the script source and the
backend's assembler invocation code. The first, second, and fourth performance
rows above use the register-cap follow-up; the BF16 row uses the main sweep.

Both runs use four warps per CTA and a grid of twice the GPU's SM count.
FP fusion is disabled. This compiler baseline hardcodes
`--regAllocOptLevel=2`; the runs therefore do not measure `ptxas`'s unmodified
default register allocator. All correctness checks passed across 60 paired
configurations in the main sweep and 16 in the register-cap follow-up, against
a reference using FP64 summation and the same narrowing recurrence.

## Limits

Sequential is not universally faster. Uncapped FP16 K64/O1 takes
25.90 → 26.52 µs (about 2.4% higher latency) despite reducing allocated registers
from 92 to 72. At K64/O2 with a 64-register cap, sequential takes
135.61 → 210.87 µs (about 55.5% higher latency), even though its local-memory
frame is smaller. Register count and local-memory size alone do not predict
performance; dependency chains, conversion packing, and spill scheduling also
matter.

Single-iteration load/reduce/store measurements generally show smaller gains.
These CUDA-graph measurements reuse the same buffers and do not characterize
cold-memory bandwidth. Attention's live values, warp-specialization budgets,
and overlap with other work require separate measurements.
