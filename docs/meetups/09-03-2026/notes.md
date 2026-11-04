# Agenda:
* Ian Barber (Meta) presented a mixed-radix extension to Triton's linear layout system to support non-power-of-two (NPOT) shapes natively, currently gated behind TRITON_ALLOW_NPOT=1 in Meta's fbtriton fork with exact product layouts landing in the next few weeks
* Alexey Loginov (Meta) Triton Developer Summit is October 19th at the San Jose Convention Center — free to attend, registration open now
* Avik Chaudhuri (Meta) presented TritonPPM, a static analysis + XGBoost model that predicts Triton kernel performance without compiling — mean regret under 3%, zero regret on 82.5% of problems at top-10, with predictions running in under 1 second cold; a paper is forthcoming and open-source release is planned
* Keren Zhou (OpenAI) walked through 5 new Proton profiler features: CUDA graph profiling, continuous profiling, async profiling events, third-party device support, and PC sampling on AMD GPUs

# Minutes:

* Non power-of-two shapes in Triton
  * The core problem is that Triton's linear layout system is built on powers of two, so shapes like 96 (= 32 × 3) currently require padding (e.g. 96 → 128, wasting 25%) or manual decomposition (e.g. 192 → 3 × 64)
  * Ian's approach extends linear layouts to mixed-radix arithmetic — instead of all-binary bases, a register dimension can use radix 3 (or 5, 7, 9), letting the layout represent shapes like 96 exactly with no wasted slots
  * The main implementation challenge is proving layout validity all the way through the transform pipeline, since the mixed-radix algebra isn't closed in general — the fallback is always padding
  * Current status in fbtriton: tl.arange, elementwise ops, loads/stores, and wave-quant autotuning candidates work today; exact product layouts, reductions/scans, WGMMA, and Blackwell (TCGen05/MMAv5) support are coming in the next few weeks
  * Ian noted the win is more often reduced memory movement than compute savings, and the autotuner will still include power-of-two variants since they're sometimes faster
* Triton Developer Summit (Oct 19)
  * Triton Developer Summit is October 19th at the San Jose Convention Center, the day before PyTorch Conference (Oct 20–21)
  * Attendance and registration are free — breakfast, lunch, and a happy hour are included, put on by NVIDIA in partnership with the organizers
  * Agenda is being finalized and early drafts were expected out September 3rd or 4th (the day after or two days after this meeting); roughly 8 full talks, ~5 lightning talks, and posters including work from Ian and Avik
* Predicting Triton kernel performance
  * TritonPPM works by running a static analysis on the kernel source once per kernel to produce symbolic formulas for work-per-block (bytes loaded/stored, flops, register/spill estimates, shared memory, dependency/pipelining), then evaluating those formulas cheaply per config and passing the features through XGBoost
  * Training used >100 kernels from Triton tutorials and TritonBench, >250K launch instances benchmarked on B200, with leave-one-kernel-out cross-validation to test generalization to unseen kernels
  * Key results: Spearman correlation 0.96–0.99 across kernel types; k* (min candidates for zero regret) has mean 9.8 and p50 of 1; at top-10, mean regret is 2.8% and zero regret on 82.5% of problems; at 5% of config space, zero regret on 83.1%
  * Speed: formula generation is ~500ms per kernel (once), feature evaluation is under 100ms per 1K configs, and cold-start for ~5K configs is under 1 second
  * Retraining for a new GPU requires new benchmark data but only ~10 minutes of training time; the static analysis formulas themselves don't need to change much across hardware
  * Next steps: paper with full static analysis details, TLX extension for explicit register/local memory controls, integrations with Inductor and TritonParse, and potential upstreaming to Triton
* Proton profiler new features
  * CUDA graph profiling: Proton's scope API lets users annotate replay call sites so each kernel launch can be attributed back to its capture point and carry per-call-site metrics — something PyTorch Profiler can't do
  * Continuous profiling: targets long-running jobs (hours to weeks) at 1–2% overhead (sub-1% on recent NVIDIA GPUs) by structuring execution into user-defined phases and flushing data to disk per phase rather than accumulating in memory; supports both background auto-flush and manual async/sync data copy
  * Async profiling events: opaque events that can be passed across loops, functions, and warp-specialized regions to measure async ops like cp.async and tcgen05, with producer-consumer relationships linked in the trace view
  * Third-party device support: out-of-tree Triton backends can now register a Proton plugin with a CMake file and a custom profiler backend — no changes needed to the rest of Proton
  * PC sampling on AMD GPUs: PC sampling (previously well-known on NVIDIA) is now available on AMD GPUs in Proton; Keren worked with AMD over the summer to stabilize the ROCm software stack, and per-line instruction sample counts and stall breakdowns are available

# Recording
* Recording link [here](https://youtu.be/KcjctHAtyRQ)
