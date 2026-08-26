# Working on Triton

## Block Programming Model
- Triton and Gluon programs operate on logical blocks, not independent hardware threads. Scalar values, scalar control flow, and memory descriptor origins are uniform within a logical block or warp-specialized execution region. Per-element variation must be represented by tensors with valid layouts. Membar inserts the required intra-CTA barriers so shared-memory operations obey this block-level model from the user's perspective.
- Do not introduce hidden lane- or warp-varying scalars through inline assembly, hardware IDs, or other backdoors. Such programs violate the programming model and must not be used as reproducers or as justification for compiler guards. Hardware IDs used by compiler lowerings to implement layouts are different from source-level scalar values.
- Block-uniform does not mean loop-invariant or identical across warp-specialized regions. Preserve physical alias, execution-region, and iteration-boundary checks when reasoning about synchronization, as well as the documented waits for asynchronous operations.
- Regression tests must be expressible as valid Triton/Gluon block programs. Targeted PTX delays may expose an existing race, but must not change values, memory effects, synchronization, or thread participation, or introduce divergence.

## Descriptor Memory Effects
- Shared-memory read/write effects access only the logical elements of their attached memory descriptor's view. This also applies to gathers, scatters, shared-memory atomic operations, and asynchronous copies; these operations do not access memory outside their descriptors.
- FP4 tensor copies do not touch the padding bytes introduced by their shared-memory layout. Padding affects the physical addresses of logical elements, not which elements an operation accesses.

## Build and Testing Guidelines
- Before running tests for native/compiler changes, run `make` in the triton directory to rebuild triton. DO NOT RUN `make` if you only changed Python code or code in `python/triton_kernels`.
- For compiler changes, add tests in `python/test/` (pytest) or test (lit). Keep GPU-only tests in `python/test/unit/` or `python/test/gluon/`, name them `test_<feature>_<condition>`, and avoid creating new test files unless requested.
- Run pytest with `-s --tb=short`. Run a single test with `pytest file.py::test_name`.
- The build dir is given by `BUILD_DIR := $(shell PYTHONPATH="./python" python3 -c 'from build_helpers import get_cmake_dir; print(get_cmake_dir())')`
- Run lit from the build dir:  `cd BUILD_DIR; ninja triton-opt; lit -v test/<path>.mlir` (example: `lit -v test/TritonNvidiaGPU/tmem_layouts.mlir`).
- Lit tests can be run locally (no GPU required).
- Compiler crashes sometimes print an MLIR reproducer (external_resources / mlir_reproducer). Save the full MLIR + {-# ... #-} metadata to `/tmp/<file>.mlir`, then run `triton-opt /tmp/<file>.mlir --run-reproducer` to reproduce locally.

## C++ Guidelines
- In C++, never put side-effecting code in `assert`. Assertions may be compiled out, so perform mutations and other required computation before the assertion and assert only the resulting condition. This guideline does not apply to Python `assert` statements.

## Lowering Guidelines
- Triton IR uses fixed-width integer types, not MLIR's `index` type. Do not report missing `index` support as a Triton bug.
- Lowerings must inspect only the operation they lower. Do not inspect other operations or follow loop-carried values; perform cross-operation reasoning in a separate analysis or transformation pass.
