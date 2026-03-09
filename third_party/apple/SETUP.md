# Apple MPS Backend — Build Setup

## Prerequisites

- macOS with Apple Silicon
- Xcode + Metal Toolchain: `xcodebuild -downloadComponent MetalToolchain`
- Full Triton LLVM tarball extracted to `~/.triton/llvm-full/`

```bash
mkdir -p ~/.triton/llvm-full
tar -xzf /tmp/llvm-triton.tar.gz -C ~/.triton/llvm-full
```

## Build

```bash
bash /tmp/build_triton.sh   # cmake configure + ninja
```

`/tmp/build_triton.sh`:
```bash
LLVM=~/.triton/llvm-full/llvm-20902f0b-macos-arm64
PB=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())")
BUILD=~/projects/oss/triton/build/cmake.macosx-26.0-arm64-cpython-3.14

LLVM_SYSPATH=$LLVM cmake ~/projects/oss/triton \
  -GNinja -B $BUILD \
  -DLLVM_DIR=$LLVM/lib/cmake/llvm \
  -DMLIR_DIR=$LLVM/lib/cmake/mlir \
  -Dpybind11_DIR=$PB \
  -DLLVM_SYSPATH=$LLVM \
  -DTRITON_BUILD_PYTHON_MODULE=ON \
  -DTRITON_CODEGEN_BACKENDS="apple;nvidia;amd" \
  -DTRITON_BUILD_PROTON=OFF \
  -DCMAKE_BUILD_TYPE=Release

ninja -C $BUILD -j$(sysctl -n hw.logicalcpu)
```

## Install (editable)

```bash
cd ~/projects/oss/triton
LLVM=$HOME/.triton/llvm-full/llvm-20902f0b-macos-arm64
LLVM_SYSPATH=$LLVM \
TRITON_BUILD_DIR=$HOME/projects/oss/triton/build/cmake.macosx-26.0-arm64-cpython-3.14 \
pip install -e . --no-build-isolation
```

`TRITON_BUILD_DIR` reuses the existing cmake build — no recompile.

## Patches applied to this repo

| File | Change |
|------|--------|
| `setup.py:375` | backends `["apple", "nvidia", "amd"]` (was `["apple"]`) |
| `third_party/apple/python/triton_apple.cc` | `PYBIND11_MODULE` → `void init_triton_apple(py::module &&m)` + `DialectRegistry` move capture |
| `lib/TritonAppleGPUToLLVM/Dialect.cpp` | Added missing headers + implement `verify`/`getRepOrder`/`getRepOrderForOperand` |
| `lib/TritonAppleGPUToLLVM/DotOpToLLVM.cpp` | Modern MLIR API (`Op::create`, `LLVM::getVectorType`) |
| `lib/TritonAppleGPUToLLVM/AppleMmaLayoutConversions.cpp` | Fixed namespace to `mlir::triton::applegpu` |
| `lib/TritonAppleGPUTransforms/AccelerateAppleMatmul.cpp` | Modern MLIR API + `::impl::` namespace + `applyPatternsGreedily` |
