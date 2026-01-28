# Agenda:
* Update on triton-shared (Haishan Zhu and Nhat Nguyen, Meta)
* Update on the plugin system infrastructure - what's upstream today and roadmap  (Corbin Robeck and Puyan Lotfi, Meta)
* Standing up a repo with useful plugins (testing, deployment, etc). (Simon Waters, kernelize.ai)

# Minutes:
* Update on triton-shared (Haishan Zhu and Nhat Nguyen, Meta)
  * Haishan and Nhat work on MTIA's Triton compiler.
  * triton-shared is a subset of dialects and compiler passes that perform architecture agnostic lowerings of Triton dialects.
  * Microsoft Maya team maintained it but is stopping now. They're currently working on passing the reins to the Meta MTIA team.
  * Contributors were confused with Microsoft's post, but rest assured, triton-shared is still alive and will continue to thrive with Meta's maintenance and hosting (update coming soon to triton-language slack channel).
  * Questions? Reach out to Haishan or Nhat on slack.
  * Landed enhancments over the last year
    * Standardized handling of Triton pointer type
      * Lower pointer types and ops to MLIR's PtrDialect
    * Widened pointer analysis coverage
      * Lowering of `tt.atomic_raw` to `tts.atomic_ram` (for atomics) - supports atomic operations on structured memory regions.
      * Added control-flow support - detects control-flow and correctly generates correct pointer arithmetic.
        * Q> Ettiore, Intel: Do you rely on compiler being able to convert `scf.if` into `select` operation?
        * A> Yes. But in complicated cases, `scf.if` will still be there.
      * More agressive constant folding for `tts` ops.
    * Added `TensorDescriptor` support (add `TensorDescriptorToPointerPass` pass to your compilation pipeline)
  * Q> Why is there a triton-shared repo?
  * A> Lots of analysis that don't apply to GPUs, e.g. lowerings that benefit non-GPU architectures (memory/arithmetic ops).  Gives us a place to put it because its not useful for GPU-oriented triton.
  * Q> How about contributing this to MLIR repo? Maybe to linalg?
  * A> Will discuss it off-line.
* Triton Extensions, Plugins and Custom Ops (Corbin Robeck and Puyan Lotfi, Meta, Thomas Raoux, OpenAI and Simon Waters, kernelize.ai)
  * Background
    * OSS Triton priorities aligned with OpenAI’s internal use cases. Does utility justify upstream maintenance is cost?
  * Use cases:
    * Some features are bleeding edge, useful to only a handful of users (but they’re needed for cutting edge models.)
    * Some features are experimental, developed and maintained on a fork.
    * In both cases, developers are expected to do fork and maintain until accepted (large burden to keep in sync with head.). This is a big problem with customers that use a pinned version, external fork, of Triton for production (upstream not obligated to fix breakages to your fork.)
    * Model/kernel/hardware specific passes (e.g. for warp specialization. Allow for experimenting without recompiling Triton.  How to move it from my fork back upstream?) (Same as above)
    * Giving LLMs/autotuners access to passes, knobs without recompiling Triton.
  * Overview of existing pass pipeline
    * Transformation passes - (within the same IR)
    * Conversion passes - (rewrite from one IR to another. e.g. TTGIR->LLVM)
    * Current, new passes require recompiling Triton compiler (slow, cumbersome to iterate!)
  * New Plugin Framework
    * New API
    * Enforces layerings
    * Full featured: dialects, custom operations
    * Don’t need to work on a fork!  Don’t need to recompile compiler to experiment.
    * See https://github.com/triton-lang/triton/tree/main/plugins for examples.
  * Concept: overrideable pipeline
    1. Hooks: embedded in backend compiler.py (allows Python to insert the plugin to compiler) Overrides passes in add_stages table.
    2. Native code: invoked by hook.
    * Kernel or library provides hint to invoke pass by setting hook.
    * Inspect_stages hook will override TTIR entry in stages table, invoking wrapper before rest of TTIR pass setup.
    * Examples:
      * Override existing passes
      * Run custom out of tree passes
      * Run custom ops, dialects, lowering passes
      * Import entire out-of-tree backend (at runtime!)
  * Plugin interface
    * Uses PyBind names.
    * See demo for how to register plugin with API
    * See demo for creating a transformation pass (e.g. like invoking pass after a loop unrolled.)
  * Custom dialects
    * New operations!
    * Registered similarly to passes.
  * Custom out-of-tree targets (in progress)
    * Alternate path for loading backends (backend is something like AMD, Nvidia, etc.)
    * Current backends are loaded with macros.
    * New way, load from external shared object, don’t need to statically link in with Triton at compile time.
  * Custom DSL Ops (in progress)
    * Higher level
    * Top level DSL ops, at the Python level.
    * Triton rewrites it to dialects it understands and lowers and can run it.
    * Use case: proton instrumentation passes could be rewritten to use it (may not decide to go this route.). Create sanitizers that complement existing performance tooling.
* Triton-ext Repository (Simon Waters, kernelize.ai)
  * Overview
    * Public location to check in our out-of-tree passes
    * Backends, dialects, language extensions, passes, etc.
    * Expecting lots of folks to need common passes and infrastructure.
    * Triton-distributed interested in adding their stuff
  * Example: LoopSplit pass in triton-ext repository
    * Why we did it.  FlashAttention called inner loop twice for causal case (why not a single loop and have it split it automatically.).  Get 5% perf improvement by doing automatic splitting (if original loop written with causal branch). Helps if user has causal branches and doesn’t realize they can make it more efficient this way.
    * Passes only (implementation still in flux)
    * Definition: splits a loop into two loops based on a condition (and can optimize out the condition in both loops later.)
    * Triton-ext provides a database of passes, linked to libTritonExtPassInfra.so.
    * Simplified boilerplate code to point to API entry points, Only need to set 2 values: extension name and class and its registered.
    * Note: Simon wrote this version of LoopSplit before CUDA tileIR wrote theirs. (About 1 year ago).
    * Demo LoopSplit pass.  Not a lot of overhead to create it.
* Questions
  * Q> Ettiore, Intel.  We’ve got generic TTIR passes that modify X that could have been upstream of or added to this plugin infrastructure.
    * A> Use this as a proving ground for passes to be upstreamed.  LoopSplit was a prototype.
  * Q> Ettiore, Intel. You could have landed this in regular triton repo to start with.
    * A> Yes. Absolutely.  Use this as a route to get your plugin vetted before merging upstream.
    * A> Corbin, Meta. Lots of passes we depend on for auto tuning but not useful for more broadly. Maybe a 5% chance it work (very kernel specific things we want to be available to the auto tuner.). But other folks might make use of them too.
    * A> Corbin. Meta.  Hard to get enough data to prove utility without it existing in the first place.
  * Q> Two types of plugins: generic and specific (maybe to a particular architecture).  Don’t mix them.
    * A> Yea. We’ve got a backend directory to put arch specific passes.
    * A> Corbin, Meta. Exposing pass heuristics you want to tune dynamically (without recompiling). Development velocity speed up by being able to dynamically modify the pipeline.
    * A> Simon, kernelize.ai, Shared repo will facilitate collaboration. Make it easier to add new things and collaborate with the community.
    * A> Puyan, Meta, You don’t need to create a plugin before you go into triton core. This is a lighter weight to try things out. E.g. a concurrency sanitizer
  * Q> Ettiore, Intel, What if your transformation pass in your pipeline relies on different layouts? Different systolic array vs MMA and you want a custom layout.
    * A> Corbin, Meta, Linear layouts.  Add another linear layout to meet you custom layout.
  * Q> Example in repo?
    * A> No, But we’ve done some experiment to combine two linear layouts.
    * A> Puyan, Meta, At least one use case we’ve thought of.

# Minutes
* Recording link [here](https://youtu.be/JnFFwBB6Dhk)
