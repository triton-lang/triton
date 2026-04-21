---
owner: ""
created: 2026-04-20T23:55:53Z
updated: 2026-04-21T07:05:45Z
---

# Gluon Attention Forward Tuning

## Rationale

Tune `python/examples/gluon/01-attention-forward.py` with a shape-aware kernel configuration selector so the example can improve across the benchmark surface without cloning kernel entrypoints.

## Invariants

- Keep one Gluon attention kernel entrypoint and source implementation.
- Preserve correctness against PyTorch scaled dot product attention for the existing pytest matrix.
- Treat causal masking, TMEM reductions, and boundary cases as correctness-sensitive fast paths.
- Promote tuning changes only after build, correctness, and benchmark evidence.

## Current Understanding

### Systems Involved

- `python/examples/gluon/01-attention-forward.py`: target kernel, Python wrapper, tests, and benchmark harness.
- `python/examples/gluon/05-moe-bmm1-fused-gather.py`: local selector pattern with frozen dataclass configs and occupancy-specific helpers.
- Kernel optimization skill attention notes: prior attention tuning found D128 noncausal split-by-4 and D64 noncausal large-context split-by-8 useful, and identified causal diagonal TMEM row-max as correctness-sensitive.
- Same-input local benchmarking on this machine confirmed D128 split-by-4 for fp16/bf16 and rejected D64 split-by-8 for the tested fp16 noncausal sweep.
- Causal D64 at N=1024 showed a repeatable small win from split-by-2; larger D64 causal contexts stayed best at split-by-4.
- A narrow tile/occupancy sweep did not find a second promotion: `BLOCK_M=128` fails to compile in the current warp-specialized body, and `OCCUPANCY=2` was slower on representative fp16 rows.
- A 4-GPU autotune pass with 32 compile-only warmup workers searched split factor, max register cap, and noncausal TMEM policy. Warmup compiled 576/768 candidates; all 192 failures used `MAXNREG=160`, so that cap was pruned. The warmed 576-candidate benchmark pass finished in about 75 seconds wall-clock.
- The warmed search found automatic noncausal TMEM reduction wins on GB300 for every fp16/bf16 shape searched. D64 noncausal pairs best with `SPLIT_EXP_FACTOR=1`; D128 remains best at split 4.
- A schedule search over `BLOCK_N` and `GROUP_SIZE_N` warmed 336 candidates with 32 workers; `BLOCK_N=256` failed, `BLOCK_N=64` did not win, and causal `GROUP_SIZE_N=8` was a strong win except for large D128 contexts.
- A buffer search exposed `NUM_KV_BUFFERS` and `USE_EXP2_TURNSTILE` as selector parameters. D128 only compiled with 2 or 3 KV buffers and stayed best at the existing 3. D64 causal improved with 2 KV buffers; for N>=2048 it also improved when the exp2 turnstile was disabled.
- A fixed-GPU finalist pass was run because cross-GPU performance varied. Existing 4-GPU results narrowed to 312 candidates, then the per-case top 3 plus current/legacy configs were rerun on GPU 0 only with rep=1000. A final focused repeat confirmed D64 noncausal N=8192 prefers `MAXNREG=112`, and bf16 D128 noncausal N<=2048 prefers `GROUP_SIZE_N=4`.
- The same bounded search/finalist workflow was extended to fp8 by setting `ATTN_SEARCH_DTYPES=fp8`. The existing benchmark already had `triton-fp8`; the search harness now supports dtype selection and fp8 result roots. Fp8 broad/schedule/buffer searches found coherent wins across all 16 fp8 benchmark-surface cases.

### Expected Change Surface

- Files/directories likely to change: `python/examples/gluon/01-attention-forward.py`; this initiative file.
- Data models/APIs affected: optional Python-side kernel config and selector for `attention_forward`.
- Operational or rollout touchpoints: local `make`, pytest, and benchmark runs on Blackwell GPUs.

## Assumptions and Risks

| Type | Item | Confidence (Low/Med/High) | Validation Plan |
|---|---|---:|---|
| Assumption | Existing tile geometry is broadly correct, and initial wins should focus on split factors and safe path selection. | Med | Compare broad benchmark output before/after. |
| Assumption | D128 split-by-4 is applicable to this source revision for causal and noncausal fp16/bf16 rows. | High | Full pytest matrix passed; paired A/B benchmark showed about 4.5-6.7% fp16 gains and about 5-6.5% bf16 spot-check gains. |
| Risk | TMEM reduction on causal diagonal uses pre-mask maxima and can be semantically invalid. | High | Disable selected TMEM reduction for causal until a masked row-max path is implemented and tested. |
| Risk | GPU access or long build time may limit validation in this session. | Med | Record exact commands and blockers in this file and handoff notes. |

## Plan

### Phase 1: Selector Wiring

- [x] Add a frozen Python kernel config dataclass and selector for attention.
  - Artifact: patch in `python/examples/gluon/01-attention-forward.py`
- [x] Route `attention_forward` through the selector while preserving explicit config override for A/B tests.
  - Artifact: patch and focused correctness run

### Phase 2: Measurement Loop

- [x] Rebuild Triton with `make` before running tests.
  - Artifact: build result
- [x] Run focused correctness gates for representative causal/noncausal, D64/D128, dtype, and TMEM cases.
  - Artifact: pytest output summary
- [x] Run broad benchmark and identify next Pareto frontier candidates.
  - Artifact: benchmark summary with wins/losses

### Phase 3: Next Tuning Slice

- [ ] Use benchmark evidence to test one additional selector axis at a time.
  - Artifact: patch or dead-end note

## Execution Log

- `2026-04-20` Started initiative after inspecting the attention kernel and MoE selector pattern.
  - Artifact: `.codex/initiatives/artifacts/gluon-attention-forward-tuning.md`
  - Validation: discovery only
  - Learnings: current attention wrapper hard-codes `BLOCK_M=256`, `BLOCK_N=128`, `SPLIT_EXP_FACTOR=256 // HEAD_DIM`, `num_warps=4`, and `maxnreg=128`.
  - Plan updates: first slice will add selector plumbing with conservative known tuning axes.
- `2026-04-21` Completed selector wiring and first measured tuning slice.
  - Artifact: `python/examples/gluon/01-attention-forward.py`
  - Validation: `make`; `python3 -m py_compile python/examples/gluon/01-attention-forward.py`; focused pytest subset `4 passed`; full `pytest -s --tb=short python/examples/gluon/01-attention-forward.py::test_op` reported `128 passed in 21.29s`.
  - Learnings: D128 split-by-4 wins across fp16 causal/noncausal N=1024, 2048, 4096, 8192 with paired speedups from about 1.045x to 1.067x. D128 bf16 N=2048 spot checks gained 1.065x noncausal and 1.054x causal. D64 split-by-8 regressed versus split-by-4 on noncausal N=4096 and N=8192, so it was not promoted. D64 causal N=1024 split-by-2 held a small repeatable 1.007x win.
  - Plan updates: next tuning slice should explore tile geometry, occupancy, TMEM noncausal policy, or register caps one axis at a time.
- `2026-04-21` Checked tile/occupancy as the next selector axis.
  - Artifact: no code promotion; dead-end evidence recorded here.
  - Validation: paired microbenchmark sweep over fp16 representative rows.
  - Learnings: `BLOCK_M=128` produced a Gluon compilation error at `gl.warp_specialize` for tested rows, while `BLOCK_M=256, OCCUPANCY=2` was consistently slower than occupancy 1.
  - Plan updates: deprioritize tile/occupancy until the kernel body is restructured; TMEM noncausal policy or register/buffer tuning are better next axes.
- `2026-04-21` Ran compile-warmed autotuning/search across 4 GB300 GPUs.
  - Artifact: `.codex/attn_search.py`, `.codex/attn_search_results/`
  - Validation: warmup and benchmark CSVs for broad and schedule searches; focused pytest subset `6 passed`; full `pytest -s --tb=short python/examples/gluon/01-attention-forward.py::test_op` reported `128 passed in 12.65s`.
  - Learnings: 32 compile-only warmup workers leveraged CPU parallelism and made the 576-candidate warmed benchmark pass complete in about 75 seconds. Noncausal TMEM reduction is a consistent GB300 win: high-rep confirmation showed D64 fp16 gains of 1.041x at N=1024, 1.034x at N=2048, and 1.029x at N=8192; D128 fp16 gains of 1.016x at N=2048 and 1.024x at N=8192; bf16 N=2048 gains of 1.036x for D64 and about 1.010x for D128. Causal group size 8 is strong for D64 and D128 through N=2048; high-rep D128 N=8192 regressed, so large D128 keeps group size 4.
  - Plan updates: next search should look at buffer counts or a kernel-body change to make alternate tile shapes compile; avoid retrying `MAXNREG=160`, `BLOCK_N=256`, or `BLOCK_M=128` without source changes.
- `2026-04-21` Promoted D64 buffer-count and exp2-turnstile selector rules.
  - Artifact: `python/examples/gluon/01-attention-forward.py`, `.codex/attn_search_results/buffer*`
  - Validation: buffer warmup and benchmark CSVs; high-rep confirmation; focused pytest subset `5 passed`; full `pytest -s --tb=short python/examples/gluon/01-attention-forward.py::test_op` reported `128 passed in 22.43s`.
  - Learnings: D64 causal high-rep confirmation showed `NUM_KV_BUFFERS=2` with turnstile disabled for N>=2048 gained 1.065x at N=2048 fp16, 1.051x at N=4096 fp16, 1.049x at N=8192 fp16, and 1.067x at N=2048 bf16. D64 causal N=1024 keeps turnstile enabled and gains 1.039x from two KV buffers. D64 noncausal N=1024 gains 1.020x from two KV buffers. D128 stayed unchanged.
  - Plan updates: remaining selector-only gains look smaller; next meaningful frontier is likely kernel-body work for alternate tile shapes or profiling the best D64 causal path to understand why turnstile flips by context size.
- `2026-04-21` Reproduced top candidates on one fixed GPU and promoted only same-GPU finalists.
  - Artifact: `.codex/attn_finalist.py`, `.codex/attn_finalist_results/`, `python/examples/gluon/01-attention-forward.py`
  - Validation: 4-GPU top-K narrowing (`312` rows); GPU 0 finalist rerun (`133` rows, rep=1000); focused repeat over four challenger cases; focused pytest subset `4 passed`; full `pytest -s --tb=short python/examples/gluon/01-attention-forward.py::test_op` reported `128 passed in 6.73s`.
  - Learnings: The current selector was already best in most fixed-GPU cases. Same-GPU finalists promoted two small but repeatable rules: D64 noncausal N=8192 fp16/bf16 uses `MAXNREG=112`, and bf16 D128 noncausal N=1024/2048 uses `GROUP_SIZE_N=4`. Focused repeats averaged fp16 D64 N=8192 current 2.08046 ms vs max112 2.04902 ms, bf16 D64 N=8192 current 2.02396 ms vs max112 2.00187 ms, bf16 D128 N=1024 current 0.06185 ms vs G4 0.06120 ms, and bf16 D128 N=2048 current 0.19488 ms vs G4 0.19346 ms.
  - Plan updates: Future promotions should use this same fixed-GPU finalist pattern after any broad parallel search; do not trust mixed-GPU top-1 ordering for marginal wins.
- `2026-04-21` Tuned and promoted fp8 selector configs.
  - Artifact: `.codex/attn_search_results_fp8/`, `.codex/attn_finalist_results_fp8/`, `python/examples/gluon/01-attention-forward.py`
  - Validation: fp8 broad/schedule/buffer compile-warmed searches; 4-GPU fp8 narrowing (`172` rows); GPU 0 finalist rerun (`69` rows, rep=1000); all 16 fp8 benchmark-surface invocations compiled and ran; same-GPU A/B against old fp8 policy; full fp16/bf16 pytest matrix reported `128 passed in 6.72s`.
  - Learnings: Noncausal fp8 should enable TMEM reduction. D64 noncausal uses split 2; D128 noncausal uses split 4 for N<=2048 and split 8 for N>=4096. Causal D64 uses group 8/split 4 for N<=2048 and group 4/split 2/maxnreg112 for N>=4096. Causal D128 uses group 8/split 2 for N<=2048 and group 4/split 8 for N>=4096. Same-GPU A/B speedups versus old fp8 policy ranged from about 1.026x to 1.139x across the 16 fp8 benchmark cases.
  - Plan updates: fp8 now has its own validated selector branch; future fp8 work should add correctness tolerances or reference comparisons if the example starts testing fp8 numerics, because current pytest coverage remains fp16/bf16.

## Next Up

- [ ] Profile the best D64 causal path or change the kernel body to make alternate tile shapes viable; for marginal selector wins, always rerun finalists on one fixed GPU. Add fp8 correctness coverage if a reliable fp8 reference tolerance is established.

## Open Questions

- Question: Which exact Blackwell SKU and benchmark repeat count should be used for final promotion?
  - Owner: ""
  - Resolution path: infer from local GPU properties and benchmark stability once validation runs.

## Deferred / Out of Scope

- New kernel entrypoints:
  - Reason: user requested config and selector tuning, and the current kernel is structured for constexpr variation.
