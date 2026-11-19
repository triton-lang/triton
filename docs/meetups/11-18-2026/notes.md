# Agenda:
* Update on triton-ext for Intel backend - Mieszko Dziadowiec
* Composable kernels with aggregates - Simon Waters
* Gaps in Triton extensibility - Saagar

# Minutes:
* Update on triton-ext for Intel backend
  * Upstream changes for out-of-tree XPU
    * The same XPU backend was built in tree and out of tree on an unmodified Triton release, as a feasibility check and not a decision to switch.
    * It needed 11 small generic changes to core: 6 are merged upstream, 2 are in review, and 3 are ready to file. The full list with reasons is in RFC #12439.
    * Merged: layout interfaces for dot-operand and shared-memory lowering, a warp size check, two Proton fixes, and an FPSan fix that limits the NVIDIA-specific layout to NVIDIA. The first three are in 3.9.0; the other three are on main, so 3.10 at the earliest.
    * In review: core tracking binary stages instead of hard-coding cubin/hsaco (#12265), and backend specialization on tensor descriptor arguments (#11837). The latter matters because host-side descriptors couldn't use the fast load path, which cost about 80% of performance.
    * Ready to file once the two in review land (contributors without write access can have only 2 PRs open): a Gluon layout converter registry, Proton headers in the wheel, and Proton instrumentation lowering from other backends.
  * Out-of-tree test and performance parity
    * 87,204 tests from Triton, PyTorch, vLLM and Liger-Kernel were compared one by one with 0 unexplained differences.
    * TTGIR was identical for all 15,368 kernels.
    * 22 of 24 benchmarks are within ±1.5%, and the other two are treated as parity: softmax varied by about 3% between runs (noise), and fused GEMM picked a slightly different config but matched one to one with the same config.
    * The comparison used upstream main from October 8th; tests and kernels were compared on October 22nd and benchmarks on October 9th. The AOT tools aren't covered yet.
  * Windows export limit workaround
    * A Windows DLL exports at most 65,535 symbols, while libtriton in the official 3.9.0 wheel exports 137,294, about 85,000 of them from MLIR. Even if LLVM's symbol count drops (hoped for around the start of next year), MLIR remains the main blocker.
    * The backend uses only about 2,200 symbols, so core can export just that list and fit under the limit.
    * The workaround has 4 steps, about 38 minutes in total on an i9-13900:
      1. Build LLVM with fallback type IDs (23 min)
      2. Build core with Triton extensions enabled and a 12-line CMake patch (9 min)
      3. Build the backend and list the symbols it uses (3 min)
      4. Relink core to export that list (3 min)
    * An LLVM contributor opened a follow-up issue about MLIR as a DLL about two weeks ago, so it looks planned.
  * Keeping up with Triton core changes
    * Open question from Mieszko: how to handle a core change that breaks an out-of-tree backend, since nobody can guarantee the current changes are everything needed.
    * Thomas thinks the changes are too fine-grained and suggests larger extension boundaries, such as Intel having its own lower-level dot op and layouts, even at the cost of duplicating some code. He also noted upstream is more sensitive about core changes than people realize, so where the line sits for small generic changes is subjective.
    * Intel's fork merges from upstream almost daily, since falling behind for a few months early in the project was painful to catch up on. Whitney would rather learn early that an extension approach needs a common code change than find out after a release.
    * Ettore noted that duplicated code means NVIDIA bug fixes can be missed even without merge conflicts, so they use tools to find fixes to port back.
    * Simon has noticed that core development has slowed down a lot at the TTIR level.
* Composable kernels with aggregates
  * Simon's PR lets aggregates have methods and tensor or constexpr members, and be passed as kernel parameters, where they expand like tuples and specialize the kernel.
  * It replaces ad hoc fusion, like the activation hard-coded in the matmul tutorial, with chainable epilogue objects such as BiasAdd followed by a scale.
  * Method calls are first generated as function calls in MLIR and are normally inlined, but a backend could keep them separate.
  * Not yet tested on Gluon, though aggregates already work there in theory; Saagar said they use aggregates heavily in Gluon and took the change right away.
  * Ettore likes the feature but doesn't control what goes upstream. Simon sees lambdas or closures as a possible later extension.
* Gaps in Triton extensibility
  * Saagar said current extensions mainly cover the backend (passes and lowering) and the driver; the frontend isn't well covered.
  * He wants support for new DSLs, since long kernels need functions, classes and data structures, and more interop, like calling Gluon from Triton.
  * Wants short Python micro-passes that can inspect finished IR (e.g. counting FMAs or dots, checking shared memory use) without a vendored LLVM dependency.
  * Wants execution hooks to replay compiled kernels and edit arguments at launch, rather than just compile and run; today they edit kernel source with a decorator.
  * These are ideas from local patches, not formal proposals, and Saagar asked whether others care about this area.

# Next meeting
* Late January or early February, away from the holidays. Possible topics: AI-generated kernels (how little kernel code is handwritten now) and a review of this year's Triton conference topics.
