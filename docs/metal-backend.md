# Metal backend — quick reference

## Purpose

Documents the Metal (Apple GPU) backend implementation, test locations, local smoke checks, and common CI failure modes.

## Key files and locations

- Runtime / compiler / driver (Python + C):

  - `third_party/metal/backend/runtime.py` — Python runtime, MetalLibraryHandle, launch helpers. Includes a non-darwin stub fallback.

  - `third_party/metal/backend/compiler.py` — .metal generation + calls out to `metal`/`metallib`; includes a small reflection parser for kernel argument indices.

  - `third_party/metal/backend/driver.py` and `third_party/metal/backend/driver.c` — native helper for device properties and runtime binding used by tests.

- C++ MLIR/transform glue (Triton dialects & passes):

  - `lib/Dialect/Triton/*` — Triton dialect sources. Recent fixes touched `lib/Dialect/Triton/IR/Ops.cpp` and `lib/Dialect/Triton/Transforms/LoopAwareCSE.cpp` for MLIR API/iterator compatibility.

- Tests:

  - lit tests: `test/` (many Metal tests are REQUIRES: darwin and are guarded)

  - C++ unit tests: `unittest/` (contains Metal-specific tests under `unittest/Metal/`)

  - Python tests: `python/test/backend/` (end-to-end and guarded macOS smoke tests)

## Local smoke-run helper (added)

- `scripts/run_macos_smoke_locally.sh` — convenience script that:

  - checks for macOS and essential tools (`xcrun`, `metal`, `metallib`)

  - locates Homebrew LLVM (`llvm@20` preferred) and exports `MLIR_DIR`/`LLVM_DIR`/`CMAKE_PREFIX_PATH`

  - configures CMake (Ninja) into `build_metal`, builds `all`, runs `ctest -R Metal` (minimal C++ Metal tests) and a guarded Python pytest.

## How to run locally (macOS)

1) Install Homebrew and llvm@20 (recommended):

   brew install llvm@20

   # note: llvm is keg-only; CMake needs MLIR_DIR/LLVM_DIR (script sets these automatically if brew is present)

2) Use the helper script (preferred):

   ./scripts/run_macos_smoke_locally.sh

3) Manual steps (if debugging):

   mkdir -p build_metal && cd build_metal

   cmake -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DMLIR_DIR=/opt/homebrew/opt/llvm/lib/cmake/mlir -DLLVM_DIR=/opt/homebrew/opt/llvm/lib/cmake/llvm ..

   cmake --build . -j$(sysctl -n hw.ncpu)

   ctest -R Metal -V

## Known CI failure modes and quick fixes

- CMake can't find MLIR/LLVM on macOS runners:

  - Install Homebrew LLVM in the workflow and set `MLIR_DIR`/`LLVM_DIR`/`CMAKE_PREFIX_PATH` to the brewed prefix (workflow already attempts this).

- MLIR API/namespace drift (compile errors):

  - Small utility functions sometimes moved between namespaces across MLIR versions (e.g., `call_interface_impl` → `function_interface_impl`). Fix by updating callsites or adding thin compatibility wrappers.

  - Example fixes applied in this branch: `lib/Dialect/Triton/IR/Ops.cpp` and `lib/Dialect/Triton/Transforms/LoopAwareCSE.cpp`.

- Missing optional subdirectories on CI (causes add_subdirectory failure):

  - Guard `add_subdirectory(TritonMetalGPU)` with an `if(EXISTS ...)` check to avoid configure failure when a dir is absent (done in this branch).

- llvm-lit path mismatch: brewed `llvm` is keg-only; `llvm-lit` may be under `/opt/homebrew/opt/llvm/bin/llvm-lit`. If `AddLLVM.cmake` warns, ensure the brewed llvm prefix is added to `CMAKE_PREFIX_PATH` and `PATH` as needed.

## CI debugging tips

- Use the GitHub Actions UI to open the failed job log and search for the first ERROR/FATAL line; it often points to the root cause.

- For deeper inspection, install and use the `gh` CLI locally: `gh run watch --repo <owner>/<repo> <run-id>` and `gh run view --log <run-id>`.

- When a C++ compile error appears, search for the file/line in the repo and instrument small reproducer builds by limiting the Ninja targets (e.g., `cmake --build . --target <single-target>`).

## If you want

- I can open a short `docs/` page inside the repo (this file) — already added and pushed.

- I can also create a short CONTRIBUTING-style section specifically for macOS/Metal developer setup and CI triage; tell me if you'd like that expanded.


## Notes

- Metal backend requires macOS (darwin) and Apple developer tools (`xcrun`, `metal`, `metallib`). Python macOS tests require `pyobjc`.

- The repo's CI workflow on this branch is `ci/macos-metal-validate` and will run a lightweight configure/build/test sequence against brewed llvm to detect MLIR mismatches early.

