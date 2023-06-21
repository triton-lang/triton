# Build

From clone root `TRITON_SRC_DIR`

```shell
LLVM_INSTALL_DIR=...
TRITON_BUILD_DIR=build

cmake -DTRITON_BUILD_MLIR_PYTHON_BINDINGS=1 \
  -DTRITON_BUILD_TUTORIALS=OFF \
  -DPython3_EXECUTABLE:FILEPATH=$(which python3) \
  -DLLVM_INCLUDE_DIRS=${LLVM_INSTALL_DIR}/include \
  -DLLVM_LIBRARY_DIR=${LLVM_INSTALL_DIR}/lib \
  -DCMAKE_INSTALL_PREFIX=/home/mlevental/dev_projects/triton/python/triton/_C \
  -S $TRITON_SRC_DIR \
  -B $TRITON_BUILD_DIR

cmake --build $TRITON_BUILD_DIR --target install
```
