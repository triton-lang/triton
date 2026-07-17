# Fused Flash-Attention with native GQA/MQA (Triton)

A memory-efficient, IO-aware **exact** attention kernel (Flash-Attention v2
style) written in Triton, with **Grouped-Query / Multi-Query Attention handled
inside the kernel** — no `repeat_interleave`, no K/V broadcast in HBM. Forward
and backward are both fused, so it is a drop-in `torch.autograd.Function` for
training and inference.

```python
from flash_attention_gqa import flash_attention

# q: [Z, H, N, D]   k, v: [Z, H_KV, N, D]   (H % H_KV == 0)
out = flash_attention(q, k, v, causal=True)   # differentiable
```

## Why this is a non-trivial contribution

Modern LLMs (Llama-3, Mistral, Qwen, Gemma, DeepSeek) use **GQA**: far fewer
K/V heads than Q heads. The naive way to run any attention kernel on GQA is to
`repeat_interleave` K and V up to `H` heads, which:

* inflates KV memory traffic by the group factor (e.g. 4× for a 32:8 config),
  and attention is **memory-bound**, so that traffic is the bottleneck;
* wastes HBM materialising the broadcast tensors.

This kernel instead maps each query head to its shared KV head with pure
pointer arithmetic (`off_hkv = off_h // GQA_GROUP`), reading each KV tile
**once**. The backward pass does the mirror image: gradients from all query
heads in a group are **reduced back onto the single shared KV head** via
atomic accumulation — the subtle part that a broadcast-based approach gets for
free but a fused GQA kernel must handle explicitly.

## Design

### Forward (`_fwd_kernel`)
* One program per `(batch·head, query-block)`. Streams over key blocks keeping
  the running max `m_i` and denominator `l_i` (online softmax); the
  `N×N` score matrix is never written to HBM.
* `exp2` + a `log2(e)`-folded scale instead of `exp` (matches hardware fast
  path).
* Causal masking uses the standard bound `hi = (start_m+1)·BLOCK_M`, so fully
  masked key blocks are never visited.
* Saves the log-sum-exp `L` (natural log) for a cheap, recomputation-based
  backward.

### Backward (`_bwd_preprocess` + `_bwd_kernel`)
* `delta = rowsum(dO ∘ O)` is precomputed.
* One program per `(batch·head, key-block)`; iterates query blocks and
  **recomputes** `P` from the saved `L` (Flash-Attention v2 recomputation —
  trades FLOPs for memory).
* `dK`/`dV` accumulate locally then scatter (atomic) onto the shared KV head;
  `dQ` is scattered with atomics because many key-blocks contribute to the
  same query rows.

## Correctness

Verified against a plain PyTorch reference for **MHA, MQA and GQA**, causal and
non-causal, for both the forward output and all three gradients:

```
$ TRITON_INTERPRET=1 pytest tests/ -v
16 passed
```

Max observed error vs. fp32 PyTorch reference: forward `~4e-7`, gradients
`~1e-6`. The test suite runs on **CPU** via Triton's interpreter, so
correctness is reproducible without a GPU.

## Benchmark

`bench/bench_flash_attn.py` reports forward TFLOP/s vs. `torch.nn.functional.
scaled_dot_product_attention` on a Llama-3-8B-style head config (H=32, H_KV=8,
D=128, causal). Run it on a CUDA GPU:

```
python bench/bench_flash_attn.py
```

> Note: performance numbers require a GPU and are intentionally **not**
> reported here since they were not measured on target hardware in this
> environment. The benchmark harness is provided so they can be reproduced.

## Files

| File | Purpose |
|------|---------|
| `flash_attn.py` | Forward + backward Triton kernels and the autograd wrapper |
| `tests/test_flash_attn.py` | Correctness tests (fwd + gradients), CPU-runnable |
| `bench/bench_flash_attn.py` | TFLOP/s benchmark vs. PyTorch SDPA (GPU) |

## Limitations / future work

* Head dim is a `constexpr` tile; non-power-of-two dims are padded by the
  caller.
* `dQ` uses global atomics; a split-K + deterministic reduction would remove
  atomic contention on very long sequences.
* FP8 (E4M3) QK path and TMA-based loads (Hopper) are natural extensions of
  the same structure.
