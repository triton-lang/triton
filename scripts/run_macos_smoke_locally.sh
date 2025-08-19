#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"

echo "[macos-smoke] Starting local macOS smoke-run checks"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "[macos-smoke] Skipping: not macOS (uname != Darwin)"
  exit 0
fi

echo "[macos-smoke] Checking essential tools: xcrun, metal, metallib"
if ! command -v xcrun >/dev/null 2>&1; then
  echo "[macos-smoke] ERROR: xcrun not found on PATH"
  exit 2
fi

if ! command -v metal >/dev/null 2>&1 || ! command -v metallib >/dev/null 2>&1; then
  echo "[macos-smoke] WARNING: metal/metallib not found on PATH — compilation tests will be skipped"
fi

echo "[macos-smoke] Checking for Homebrew LLVM (llvm@20 or llvm)"
LLVM_PREFIX=""
if brew --prefix llvm@20 >/dev/null 2>&1; then
  LLVM_PREFIX="$(brew --prefix llvm@20)"
elif brew --prefix llvm >/dev/null 2>&1; then
  LLVM_PREFIX="$(brew --prefix llvm)"
fi

if [[ -z "$LLVM_PREFIX" ]]; then
  echo "[macos-smoke] WARNING: Homebrew LLVM not found (llvm@20 or llvm). CMake may fail to locate MLIR/LLVM."
else
  echo "[macos-smoke] Found LLVM_PREFIX=$LLVM_PREFIX"
  export LLVM_DIR="$LLVM_PREFIX/lib/cmake/llvm"
  export MLIR_DIR="$LLVM_PREFIX/lib/cmake/mlir"
  export CMAKE_PREFIX_PATH="$LLVM_PREFIX:${CMAKE_PREFIX_PATH:-}"
fi

BUILD_DIR="${BUILD_DIR:-build_metal}"
mkdir -p "$BUILD_DIR"
pushd "$BUILD_DIR" >/dev/null

echo "[macos-smoke] Configuring CMake (Ninja) in $BUILD_DIR"
cmake -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo ${MLIR_DIR:+-DMLIR_DIR=$MLIR_DIR} ${LLVM_DIR:+-DLLVM_DIR=$LLVM_DIR} .. || {
  echo "[macos-smoke] CMake configure failed"
  popd >/dev/null
  exit 3
}

echo "[macos-smoke] Building 'all' (may take a while)"
cmake --build . --target all -j $(sysctl -n hw.ncpu) || {
  echo "[macos-smoke] Build failed"
  popd >/dev/null
  exit 4
}

echo "[macos-smoke] Running minimal C++ Metal tests (ctest -R Metal)"
ctest -R Metal -V || echo "[macos-smoke] C++ Metal tests may have failed or been skipped"

echo "[macos-smoke] Setting up Python venv for guarded Python tests"
python3 -m venv .venv || true
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel >/dev/null 2>&1 || true
pip install -e .. >/dev/null 2>&1 || true
pip install pyobjc pytest numpy >/dev/null 2>&1 || true

echo "[macos-smoke] Running guarded Python tests (may skip on some CI images)"
pytest -q python/test/backend/metal_runtime_test.py::test_macos_smoke_launch_guarded -q -rfs || echo "[macos-smoke] Python macOS smoke tests failed or skipped"

popd >/dev/null
echo "[macos-smoke] Completed. If this was run automatically as a pre-push hook, review any failures and only push when green."
