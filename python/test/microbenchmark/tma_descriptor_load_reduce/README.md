# Descriptor-load partial-reduction microbenchmark

This standalone microbenchmark measures a descriptor/TMA load immediately followed by a partial max reduction:

```text
[1, BLOCK_M, BLOCK_N] descriptor load
    -> convert to fp32
    -> tl.max(axis=1)
    -> BLOCK_N fp32 results
```

It compares the same kernel under the Triton 3.5.1 and 3.6.0 PyPI packages. The motivating cases use a small reduction axis, BF16 or FP16 inputs, and four warps. This is a performance reproducer and evidence-gathering tool, not a regression test with a timing threshold.

## Run

The runner reinstalls each requested Triton wheel in the selected Python environment. Use a disposable environment rather than a development environment whose Triton installation must be preserved.

Minimal affected case:

```bash
cd python/test/microbenchmark/tma_descriptor_load_reduce
GPU_LABEL=h100_sxm \
VERSIONS="3.5.1 3.6.0" \
SHAPES="32x128" \
DTYPES="bf16" \
WARPS="4" \
REPEAT=5 \
WARMUP=5 \
bash run_triton_versions.sh
```

Full comparison grid:

```bash
GPU_LABEL=h100_sxm \
VERSIONS="3.5.1 3.6.0" \
SHAPES="16x128,32x128,64x128,32x64,32x256" \
DTYPES="bf16,fp16,fp32" \
WARPS="4" \
M=8192 \
N=8192 \
REPEAT=5 \
WARMUP=5 \
bash run_triton_versions.sh
```

The script performs sampled exact correctness checks before timing and emits one JSON object per line under `results/`. A complete full-grid run contains an environment record followed by 15 records with `status="ok"` for each Triton version.

Tensor descriptors use TMA on supported NVIDIA GPUs. An A100 run is useful as a negative architecture control, but the descriptor kernel may be rejected or lower through a different path because Ampere does not have Hopper TMA.

## Reproducible environment

The measurements below used:

- PyTorch 2.8.0+cu128;
- matrix size `M=N=8192`;
- five shapes, three dtypes, and `num_warps=4`;
- five warmup launches;
- three timed samples per RTX 5090 case and five per H100/B300 case.

For B300, the public image `northcapitalca/triton-b300-cuda13:0.2` supplies CUDA/PTXAS 13.3 and sets `TRITON_PTXAS_PATH` so Triton 3.5.1 can assemble `sm_103a`. The benchmark is already available at `/workspace/triton_tma_reduction_bench` in that image.

## Initial results

Each cell is `3.5.1 median -> 3.6.0 median (3.6.0 / 3.5.1)`. All displayed cases passed `--check` and have `status="ok"`.

| Case | RTX 5090 | H100 SXM | B300 SXM6 AC |
|---|---:|---:|---:|
| 16x128 BF16 | 0.084439 -> 0.093082 ms (1.10x) | 0.050487 -> 0.167688 ms (3.32x) | 0.024827 -> 0.147604 ms (5.95x) |
| 32x128 BF16 | 0.083704 -> 0.084772 ms (1.01x) | 0.048684 -> 0.088257 ms (1.81x) | 0.022414 -> 0.077734 ms (3.47x) |
| 64x128 BF16 | 0.083125 -> 0.084059 ms (1.01x) | 0.048195 -> 0.052110 ms (1.08x) | 0.022045 -> 0.042958 ms (1.95x) |
| 32x128 FP32 | 0.164173 -> 0.164387 ms (1.00x) | 0.093879 -> 0.096339 ms (1.03x) | 0.041646 -> 0.051404 ms (1.23x) |

The regression is therefore not B300-only: it is large on H100 and B300 for short BF16/FP16 reductions, while this RTX 5090 grid is mostly flat. Increasing `BLOCK_M` reduces the slowdown, and FP32 is affected much less.

Timings alone do not prove a compiler root cause. In particular, the results do not yet establish that Triton 3.6.0 selected a layout that changed a thread-local reduction into a cross-thread or cross-warp reduction.

## Planned compiler analysis

For the affected `32x128 BF16` case and the `64x128 BF16` and `32x128 FP32` controls, the next step is to compare Triton 3.5.1 and 3.6.0 at each relevant stage:

- TTIR and TritonGPU IR before and after `OptimizeThreadLocality`;
- selected load and `SliceEncoding` layouts, including `sizePerThread`, `threadsPerWarp`, `warpsPerCTA`, order, and vector width;
- inserted `convert_layout` operations and whether the reduction is thread-, warp-, or CTA-local;
- LLVM IR, PTX, cubin/SASS, register count, and shared-memory allocation;
- shared-memory traffic, barriers, shuffles, barrier stalls, occupancy, and achieved bandwidth.

Additional H200, B200, and A100 measurements can localize the architecture/path boundary, but layout and lowering evidence is required before proposing a compiler or cost-model change.
