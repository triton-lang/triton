# Agenda:
* TileLens: Visual program analysis for tile-based programming models - Taihua He, George Mason University
* Triton Extensions on Windows: investigation and findings - Quinn Pham, Intel
* Community questions (block pointer deprecation, TMA naming, Triton releases) - Whitney Tsang, Intel

# Minutes:
* TileLens - Taihua He, George Mason University
  * Motivation
    * Kernel observability is increasingly important (determinism, mega kernels, asynchrony).
    * Fragmented tooling ecosystem: Nvidia, AMD, and Tranium each ship their own tools.
    * Software side is also fragmented: many tile-based DSLs (Triton, Tilelang, CuTile, Amazon NKI). Some projects mix multiple DSLs (e.g. Mamba3 uses CuTeDSL, Triton, and Tilelang in the same project).
    * Goal: a general-purpose analyzer for tile-based DSLs.
  * TileLens design
    * Frontends (Triton, NKI, etc.) lower common ops (e.g. `tl.load`, `nki.load`) into a common IR.
    * Trace kernels via a function decorator and emit a list of IR records.
    * IR records are consumed by clients: tracer, profiler, sanitizer, visualizer.
    * Usage: `import triton_viz`, decorate the kernel, swap clients (tracer/sanitizer/profiler) by changing one config value.
  * Implementation
    * Decorator wraps the kernel with a kernel interface (Triton-style `[grid]` syntax) and an SPMD run function.
    * DSL is patched to use an interpreter (Triton already had one; a custom NKI interpreter was added).
    * Operations are double-patched to add instrumentation hooks that map DSL ops to common TileLens ops.
  * Adding a new DSL frontend (NKI case study)
    * Write NumPy implementations of all DSL functions (e.g. `nki.ndarray`, `nki.matmul`).
    * Patch the DSL namespace to point to those implementations; provide an `unpatch` to restore.
    * Provide an interpreter that runs the kernel over an SPMD grid on CPU.
    * Add custom IR ops where DSL semantics differ (e.g. NKI's `dma_copy` / `tensor_copy` mapped to a new `transfer` op).
    * Provide adapter functions and op-name maps, then register as a frontend.
  * Adding a new client
    * Implement the client abstract class and override hooks (pre/post run, arg callback, `register_op_callback`, etc.).
    * `register_op_callback` lets clients define per-op behavior (e.g. tracer stores outputs after `tl.dot`, `tl.load`, `tl.store`).
  * Notable features
    * Eager and symbolic execution (sanitizer/profiler): build a graph of address computations and only evaluate what is needed for analysis (e.g. bounds checks). Much faster than full eager execution.
    * Concurrent grid launch: simulates concurrent SM execution by running PIDs in multiple threads, so atomic races behave like they would on GPU. Configured via `triton_viz.config.num_sms`.
  * Live demo
    * Visualizer: matmul kernel, shows input arrays, masks, per-PID load regions, value heatmaps, dot/load/store records, transposes.
    * Sanitizer: runs cleanly on a correct matmul; flags out-of-bounds access (e.g. an injected `+ 1`) with the offending line and a trace of address computation.
    * Profiler: shows load/store efficiency, mask metrics, and potential issues.
  * Limitations / future work
    * Sanitizer still in active development (data-race checkers in progress).
    * Tracer is slow (Python) and memory-heavy (stores all activations); activation-checkpointing could help.
    * Doesn't model memory allocation or concurrency in detail today.
    * More DSL support needed; CuTile is on the roadmap.
  * Q&A
    * Q> How does this compare to CUTracer / tritonparse / tl.parse?
    * A> TileLens is higher level - takes any high-level DSL and squishes it into a common interface. Focused on semantic correctness and intermediate activations, not extracting peak performance.
    * Q> (Simon) Any plans for shared-memory allocation visualization, register/occupancy roofline, or lifetime analysis (e.g. for TLX where users explicitly allocate shared memory)?
    * A> Wanted to do this for NKI but it's hard without compiler internals; out of scope today, but a great extension point for users to add custom hardware-specific clients.

* Triton Extensions on Windows - Quinn Pham, Intel
  * Background: Triton extensions framework (Corbin, Puyan, Thomas, Simon - January 2026 meetup) lets you add passes/dialects/ops without modifying core Triton. Built on the Triton plugin infrastructure. See the `triton-ext` repo.
  * Intel's investigation: use pass extensions to upstream target-independent passes that benefit multiple backends.
  * Implemented 3 passes as extensions: hoist layout conversions, fuse reshape, remove boundary checks.
    * Hoist layout conversions is used by both Intel and AMD - good upstream candidate.
  * What worked
    * Minimal changes required - just hook into the pass infrastructure.
    * Works great on Linux.
  * Windows problems
    * Extensions are shared libraries dynamically loaded at runtime. Need a single-source definition of LLVM/MLIR/Triton symbols for consistent type identity (e.g. dynamic casts).
    * On Linux: disable `-fvisibility=hidden` to export everything from `libtriton.so`. Works fine.
    * On Windows: equivalent `WINDOWS_EXPORT_ALL_SYMBOLS` hits the 65,000 symbol limit per DLL. There's an MSVC linker RFC to lift this, but it has been under review for over a year.
  * Path forward
    * LLVM/MLIR symbols: no supported way to distribute LLVM/MLIR shared libs on Windows today. `BUILD_SHARED_LIBS` is dev-only; `LLVM_BUILD_DYLIB` is the recommended distribution option but isn't supported on Windows. There is upstream effort to annotate LLVM's public interface with an `LLVM_ABI` macro so the DyLib build stays under the 65K limit.
    * Triton symbols: Triton can do the same - annotate the public Triton interface and only export those symbols. This is feasible and would give us cleaner ABI control even on Linux.
  * Q&A
    * Q> (Andrew) How much effort is the `LLVM_BUILD_DYLIB` Windows work?
    * A> A lot. CI for annotation has been disabled and many sub-tasks don't even have issues yet.
    * Q> Will this require splitting libraries (LLVM, MLIR, Triton, extensions)?
    * A> Yes - extensions would link against `libLLVM`, `libMLIR`, and one (or more) Triton libs.

* Community questions - Whitney Tsang, Intel
  * Block pointer deprecation
    * Q> (Whitney, Intel) The recent OpenAI change deprecates block pointer and lowers it to tensor pointer in the frontend. Block pointer carries structural info that is critical for Intel; we have a downstream change that lowers to tensor descriptor instead. Why was tensor pointer chosen, and would lowering to tensor descriptor be welcome upstream?
    * A> (Ettore, Intel) Logically, replacing a deprecated language feature should preserve semantic info - tensor descriptor carries similar info to block pointer, while tensor pointer loses it.
    * A> (Thomas Raoux, OpenAI - joined late) Goal is to remove block pointer entirely. Mapping to pointers is straightforward via a language wrapper; mapping to tensor descriptor is more work and doesn't translate 1:1. Open to a clean standard-library-style wrapper that emulates block pointer via tensor descriptor where the target supports it. Backends already differ today (some Nvidia versions fall back to tensor pointer, others use TMA). Whitney to send a PR for review.
  * `use_tma` naming for tensor descriptors in inductor
    * Q> An issue was filed a few meetups ago to rename `use_tma` since nothing forces tensor descriptors to use TMA. No progress yet. https://github.com/pytorch/pytorch/issues/163536
    * A> (Bill) No update; will follow up offline.
  * Triton release cadence vs PyTorch
    * Q> PyTorch is releasing more frequently - how does that affect Triton releases?
    * A> (Andrey) Plan is to keep aligning Triton releases just before PyTorch releases - aiming for every PyTorch release, but in practice it may be every other release. Gating step is moving the Triton pin in PyTorch CI green; if that succeeds, we cut a Triton release. Will invest more effort around releases that introduce new architectures or important features.

* Deferred
  * Felix Lee (AMD) - fly DSL talk pushed to the July 2026 meetup.
  * Andrew Brown's team work on Triton extensions also targeted for July.

# Recording
* Recording link [here](https://youtu.be/RwqdPXBSMCE)
