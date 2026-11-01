# Agenda:
* FlyDSL: A Python-native DSL and MLIR-based compiler for high-performance AMD GPU kernels - Felix Li, AMD
* Triton Extensions: status update on symbol conflicts, distributing extensions as packages, and the triton-dev wheel proposal - Andrew Brown, Kernelize
* Triton Developer Conference announcement - Alexey Loginov, Meta

# Minutes:
* FlyDSL - Felix Li, AMD
  * Architecture and design
    * FlyDSL is a Python DSL and MLIR-based stack that gives developers full control from cluster → block → warp → thread → instruction level, unlike Triton which abstracts at block level.
    * The architecture has two layers: a hardware-agnostic Fly dialect (layout algebra, partition ops, tiled copy/MMA) and a vendor-specific FlyROCDL dialect (MFMA, WMMA, BufferCopy for AMD), with other backends possible.
    * The key design abstractions are layout algebra (CuTe-style shape+stride layouts, ported from NVIDIA) and an atom model (CopyAtom, MmaAtom) — switching GPU architecture means swapping atom types while kernel logic stays the same.
    * A GEMM kernel takes ~20 lines of Python using `@flyc.kernel`, and compile-to-GPU-binary takes under 5 seconds with no C++ rebuild.
  * Kernel library and performance
    * FlyDSL ships a ready-to-use kernel library covering GEMM (FP16/BF16/FP8, preshuffle, blockscale), attention (paged decode, Flash Attention/GQA, MLA), MoE, and fusions/comms (FusedNorm, Fused RoPE, custom AllReduce).
    * Flash Attention on MI355X (gfx950) hits 1319 peak TFLOPS (bf16), 1.20x vs CK and 1.03x vs hand-written ASM across causal FMHA shapes.
    * DeepSeek MoE kernel outperforms AITER HIP MoE by ~1.2x on average, up to 1.54x at 32K tokens and 1.38x at 1 token (CDNA4, ROCm 7.2, Mxfp4).
    * FP8 GEMM is on par with HipBLAS ASM overall, with a 1.24x peak speedup at M=N=K=10K.
    * FlyDSL kernels are in production via AITER → vLLM, SGLang, and ATOM; PyTorch Inductor integration is in progress, TileLang and FlashAttention/JAX support are planned.
    * The project is Apache 2.0 on GitHub at github.com/ROCm/FlyDSL.
  * Q&A
    * Q> Does FlyDSL expose low-level scheduling controls?
    * A> Yes — schedule barriers, schedule group barriers, and wait priority via AMD's ROCDL dialect for instruction interleaving. Inline assembly is also supported, though Felix prefers higher-level APIs where possible.
    * Q> How much automatic optimization does FlyDSL do?
    * A> Limited to basic MLIR passes (CSE, constant folding); most optimization relies on LLVM/backend for low-level code paths.
    * Q> When is thread-level control necessary vs. tile-level?
    * A> For compute-bound kernels like prefill attention, thread-level control is necessary to interleave instructions and hide register latency behind MFMAs; for memory-bound kernels, tile-level APIs are sufficient.
    * Q> How does FlyDSL compare to Gluon?
    * A> Felix thinks Gluon is good for memory-bound kernels but can't do instruction-level interleaving or register control needed for compute-bound cases — that's where FlyDSL has an edge.
    * Q> (Whitney) Do different shapes require separate kernels?
    * A> You can mix pipelines in one kernel but it makes code harder to read; in practice they tend to write separate kernels per shape regime.
    * Q> (Simon) Can you mix thread-level and block-level instructions?
    * A> Yes, via `call_to_index` and `index_to_call` conversion APIs.

* Triton Extensions - Andrew Brown, Kernelize
  * Overview
    * Triton extensions let developers build out-of-tree passes, dialects, and backends as shared libraries loaded into Triton at runtime via `TRITON_PLUGIN_PATHS`, without shipping a custom Triton build.
    * Puyan confirmed the plugin extension interface shipped in Triton 3.7, which is bundled with the latest PyTorch pip install — no special builds needed to start writing plugins.
    * Current extensions in the triton-ext repo include a loop splitting pass and arithmetic intensity pass (Simon Waters), uTLX (Corbin Robeck and Puyan Lotfi, planned for PyTorch), an Apple GPU backend (Darko, unmerged), and a CPU backend (Kernelize, looking for collaborators).
  * Symbol conflicts
    * The core problem is that LLVM/MLIR use static registries and TypeID initializers that must have exactly one definition — multiple LLVM instances in the same process cause hard-to-debug conflicts.
    * The current workaround is building Triton with `TRITON_EXT_ENABLED=1`, which exports all Triton/LLVM/MLIR symbols publicly from `libtriton.so` so extensions can reference them — but this breaks `triton-opt` lit testing and risks conflicts with other same-process libraries.
    * Windows compounds the problem because there's a hard limit on the number of exportable symbols, which LLVM+MLIR+Triton exceeds.
    * Alternatives under consideration include splitting LLVM/MLIR and Triton into separate libraries, passing MLIR as text between stages, and namespacing LLVM/MLIR symbols — none adopted yet.
    * Near-term guidance: test extensions via the Python API rather than `triton-opt`, and avoid loading multiple LLVM instances in the same process.
  * Distributing extensions as packages
    * PR #10775 proposes registering extensions from Python using `extend_with()` functions rather than the `TRITON_PLUGIN_PATHS` environment variable, which eliminates the timing problem of ensuring the env var is set before Triton loads.
    * The pattern: each extension ships an `__init__.py` that calls `passes.plugin.extend_with(LIB)`, `ir.extend_dialects_with(LIB)`, and `ir.builder.extend_with(LIB)` — so `import triton_foo` is all a user needs.
    * This also enables out-of-tree backends to be distributed as pip-installable wheels that register themselves as `triton.backends` entry points.
    * Andrew is looking for review consensus on PR #10775 before moving forward.
  * triton-dev wheel proposal
    * Proposed a `triton-dev` wheel containing the shared library, headers, TableGen definitions, and Python files, compiled with `TRITON_EXT_ENABLED=1` — looking for community feedback before implementing.

* Triton Developer Conference - Alexey Loginov, Meta
  * The Triton Developer Conference is October 19th at the San Jose Convention Center, co-located with the PyTorch Conference.
  * Proposals for presentations, posters, and tutorials are open through end of July — the deadline was recently extended.
  * Contact Bill or check the Triton Slack channel for the submissions page.
  * Bill suggested Felix submit a proposal about FlyDSL work for the conference; Felix said he'd think about it.

# Recording
* Recording link [here](https://youtu.be/6H8sbWhKCfI?si=LkFz22qlYLiVFqzK)
