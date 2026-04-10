# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is Triton

Triton is a language and compiler for writing highly efficient GPU kernels. It exposes a Python DSL (`triton.language`) that gets compiled through an MLIR-based pipeline down to PTX (NVIDIA) or AMDGCN (AMD).

## Build and Installation

```shell
# First-time setup (installs deps + builds via pip/CMake)
pip install -r python/requirements.txt
pip install -e .

# Or use make:
make dev-install       # install deps + build triton
make dev-install-llvm  # build LLVM from source then install triton
```

Speed up builds:
- `TRITON_BUILD_WITH_CLANG_LLD=true` — faster linking with lld
- `TRITON_BUILD_WITH_CCACHE=true` — use ccache
- `MAX_JOBS=N` — cap parallel jobs (avoid OOM)
- `pip install -e . --no-build-isolation` — faster incremental rebuilds

After source changes, rebuild with:
```shell
make        # runs: ninja -C <BUILD_DIR>
```

The build directory is computed via:
```shell
PYTHONPATH=./python python3 -c 'from build_helpers import get_cmake_dir; print(get_cmake_dir())'
```

## Running Tests

```shell
# No-GPU tests (lit + C++ unit tests + some frontend tests)
make test-nogpu

# All tests (requires GPU)
make test

# Subsets
make test-lit          # MLIR lit tests only (no GPU needed)
make test-cpp          # C++ unit tests (no GPU needed)
make test-unit         # Python/pytest GPU tests
make test-interpret    # Run GPU tests via interpreter (no GPU needed)
make test-gluon        # Gluon frontend tests
make test-proton       # Profiler tests
```

Run a single pytest:
```shell
pytest python/test/unit/language/test_core.py::test_name -s --tb=short
```

Run a single lit test:
```shell
cd <BUILD_DIR>
ninja triton-opt
lit -v test/TritonGPU/accelerate-matmul.mlir
```

Reproduce a compiler crash from an MLIR reproducer:
```shell
# Save the full MLIR + {-# ... #-} metadata block to /tmp/repro.mlir
triton-opt /tmp/repro.mlir --run-reproducer
```

## Architecture Overview

### Compiler Pipeline

The compilation path for a `@triton.jit` kernel:

1. **Python AST → TTIR** (`python/triton/compiler/code_generator.py`) — `ast_to_ttir` walks the Python AST and emits Triton IR (MLIR dialect `tt`).
2. **TTIR → TTGIR** (`lib/Conversion/TritonToTritonGPU/`) — lowers to the GPU-specific dialect (`ttg`) which carries layout annotations.
3. **TTGIR optimization passes** (`lib/Dialect/TritonGPU/Transforms/`) — coalescing, pipelining, layout propagation, matmul acceleration, warp specialization, etc.
4. **Backend-specific lowering** — `lib/Conversion/TritonGPUToLLVM/` plus backend-specific paths in `third_party/nvidia/` or `third_party/amd/`.
5. **LLVM IR → PTX / AMDGCN** — via LLVM; assembled and loaded at runtime.

The backend pipeline stages are registered in `python/triton/backends/` via `BaseBackend.add_stages()` in `python/triton/backends/compiler.py`.

### MLIR Dialects

| Dialect | Location | Purpose |
|---|---|---|
| `tt` (Triton) | `include/triton/Dialect/Triton/`, `lib/Dialect/Triton/` | Core ops: `tt.load`, `tt.store`, `tt.dot`, `tt.func`, etc. |
| `ttg` (TritonGPU) | `include/triton/Dialect/TritonGPU/`, `lib/Dialect/TritonGPU/` | Layout-annotated ops; handles shared memory, warp/CTA tiling. |
| `ttng` (TritonNvidiaGPU) | `lib/Dialect/TritonNvidiaGPU/` | NVIDIA-specific ops (TMA, TMEM, warp group MMA). |
| Gluon | `lib/Dialect/Gluon/` | Experimental lower-level frontend for explicit tiling. |

### Python Layer

- `python/triton/language/` — the user-facing DSL (`tl.*` functions). `core.py` is the main module; `semantic.py` implements type-checking/promotion semantics.
- `python/triton/runtime/` — JIT machinery (`jit.py`), autotuner, driver abstraction, cache.
- `python/triton/compiler/` — Python-side compiler driver (`compiler.py`) and AST→IR codegen (`code_generator.py`).
- `python/triton/backends/` — pluggable backend interface; NVIDIA and AMD backends live in `third_party/`.
- `python/triton/knobs.py` — all runtime configuration knobs (also settable via environment variables).

### Backend Plugins (third_party/)

- `third_party/nvidia/` — CUDA backend: Hopper-specific code (`hopper/`), LLVM lowering, PTX assembly.
- `third_party/amd/` — ROCm/HIP backend: AMD-specific passes and AMDGCN generation.
- `third_party/proton/` — Triton profiler (separate CMake target, own tests).

### Tests Layout

- `test/` — lit (filecheck) tests organized by dialect (`test/Triton/`, `test/TritonGPU/`, `test/TritonNvidiaGPU/`, etc.).
- `python/test/unit/` — pytest GPU tests (`language/`, `runtime/`, `cuda/`, etc.).
- `python/test/regression/` — regression tests.
- `unittest/` — C++ GoogleTest unit tests.

## Key Debugging Environment Variables

| Variable | Effect |
|---|---|
| `MLIR_ENABLE_DUMP=1` | Dump IR before every MLIR pass (use `=kernelName` to filter) |
| `MLIR_DUMP_PATH=<dir>` | Where `MLIR_ENABLE_DUMP` writes (default: stderr) |
| `LLVM_IR_ENABLE_DUMP=1` | Dump IR before every LLVM pass |
| `TRITON_INTERPRET=1` | Run kernels via Python interpreter (no GPU; supports breakpoints) |
| `TRITON_REPRODUCER_PATH=<path>` | Save MLIR reproducer before each compiler stage |
| `TRITON_ALWAYS_COMPILE=1` | Skip cache; always recompile |
| `TRITON_KERNEL_DUMP=1` + `TRITON_DUMP_DIR=<dir>` | Dump all IR stages and final PTX/AMDGCN |
| `TRITON_ENABLE_LLVM_DEBUG=1` | Pass `-debug` to LLVM |
| `TRITON_LLVM_DEBUG_ONLY=<names>` | Limit LLVM debug to specific pass names |
| `TRITON_FRONT_END_DEBUGGING=1` | Show full frontend stack traces |
| `USE_IR_LOC={ttir,ttgir}` | Remap source locations to IR line numbers |
| `MLIR_ENABLE_DIAGNOSTICS=remarks,operations` | Control MLIR diagnostic verbosity |

## Adding Tests

- **Lit tests**: add `.mlir` files under `test/<DialectName>/`. Run via `lit -v test/<path>` from the build dir. No GPU required.
- **Pytest tests**: add under `python/test/unit/`. Prefer existing test files over new ones. Name tests `test_<feature>_<condition>`. Run GPU-only tests in `python/test/unit/` or `python/test/gluon/`.
- Run pytest with `-s --tb=short`; single test: `pytest file.py::test_name`.
