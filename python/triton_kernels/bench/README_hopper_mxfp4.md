# Experimental Hopper MXFP4-to-FP8 matmul

This experiment compares FP8 and packed MXFP4 operands using Hopper FP8
WGMMA. It is a standalone benchmark, not integrated into the library's matmul
dispatch or tensor layout classes.

## Numerical contract

Inputs are E2M1 code matrices and one E8M0 scale byte per 32 elements along K.
The fast decoder accepts scale bytes **122 through 133** (exponents -5 through
6). Every decoded value in that range is exactly representable in E4M3 FP8,
including signed zero. Code-based packing helpers reject other scales; raw
repacking operations and runners assume already-validated inputs. There is no
fallback for the full MXFP4 exponent range.

The output is BF16. Each K=128 MMA tile starts with a zero accumulator, and
its result is added to an FP32 accumulator. Split-K uses an additional FP32
reduction. Comparisons against FP8 use the same accumulation decomposition;
comparisons against an FP32 reference report finite-precision error.

## Storage and execution

For a logical product `(M, K) @ (N, K).T`, the kernel computes transposed output
tiles internally. The logical RHS supplies the register operand; the logical
LHS is decoded and stored into the shared-memory operand.

- **RHS:** canonical packed rows, or a fixed 64-row by 128-K permutation that
  decodes directly into the register operand layout.
- **LHS:** canonical rows, blocked row-major tiles, or word-transposed tiles.
  A word-transposed tile stores `[K / 8, rows]` uint32 words. Each word holds
  eight consecutive FP4 codes along K; adjacent words hold different rows.
  Scales use `[K / 32, rows]` byte storage.
- **Word-transposed default:** 16 rows by 64 K. Tile dimensions are storage
  parameters. Lane/warp ownership, compute tile and split-K are separate
  runtime tuning choices; changing them does not require repacking.

The word-transposed path can store directly to the shared operand or explicitly
convert the distributed layout first. These encodings are experimental and are
not interchangeable with `HopperMXValueLayout`, even when tile dimensions match.

`hopper_mxfp4_gluon.py` contains packing helpers, `KernelConfig`, operand-format
enums and `make_runner`. The runner returns a reusable callable and output
buffer so allocation stays outside timing. Packing is likewise explicit.

## Validation and measurement

Run from the repository root with a working Hopper Triton installation:

```sh
# Check decoding, packing and matched-FP8 matmul before measuring.
CUDA_VISIBLE_DEVICES=0 python python/triton_kernels/bench/bench_hopper_mxfp4_fp8.py \
  --output /tmp/hopper-mxfp4-check --validate-only

# A small synthetic sweep.
CUDA_VISIBLE_DEVICES=0 python python/triton_kernels/bench/bench_hopper_mxfp4_fp8.py \
  --output /tmp/hopper-mxfp4-sweep \
  --shapes '1,1024,1024;32,512,512;256,2048,2048'

# Remeasure fixed configurations on a second seed and include LHS repacking.
CUDA_VISIBLE_DEVICES=0 python python/triton_kernels/bench/bench_hopper_mxfp4_fp8.py \
  --output /tmp/hopper-mxfp4-sweep \
  --confirm word_16x64 word_16x128 word_32x32
```

Omit `--shapes` for the 32-shape synthetic sweep. It compares 16 word-transposed
formats plus canonical and three blocked formats. Runtime tuning uses coordinate
search: select a compute configuration, test lane/warp mappings, retune compute,
then recheck mappings. It is not an exhaustive joint search.

Warm timing uses five CUDA-graph measurements. Cold timing uses 50 CUDA-event
samples with a 256 MiB cache flush outside each timed interval. Both include
loads, decoding, layout handling, shared stores, MMA and split-K reduction.
Cold measurements use **warm-selected configurations**, not independently
cold-tuned ones. Ranking minimizes worst cold-cache regret, then geometric-mean
cold slowdown. Regret is `latency / best measured latency at that shape - 1`.

The output directory contains all tuning trials, per-shape timing samples, separate FP8-LHS and
FP8-both baselines,
`regret_per_size.csv`, `ranking.json`, input hashes, runtime metadata and source
snapshots. Confirmation adds fresh-input measurements, generated PTX/TTGIR,
and canonical-to-word repack-plus-matmul timing. Its per-shape comparison set
contains the requested finalists, canonical rows and the original winners.
A source mismatch requires a new output directory rather than silently mixing
measurements from different implementations.

Initial quantization and allocations are excluded. Matmul-only results assume
already-packed inputs. Repacking on every invocation can erase a kernel gain,
especially for small matrices. These are synthetic kernel measurements, not
model-throughput claims or guarantees for unseen shapes.
