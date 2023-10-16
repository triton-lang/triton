python3 -m pip install lit
cd python
LIT_TEST_DIR="build/$(ls build | grep -i cmake)/test"
if [ ! -d "${LIT_TEST_DIR}" ]; then
    echo "Could not find '${LIT_TEST_DIR}'"
    exit -1
fi

lit -v "$LIT_TEST_DIR"
# lit -v "$LIT_TEST_DIR/Conversion/AMDGPU/load_store.mlir"
# lit -v "$LIT_TEST_DIR/Conversion/tritongpu_to_llvm.mlir"
# lit -v "$LIT_TEST_DIR/Target/mlir_to_amdgcn_float16_vectorized_load_store.mlir"
# lit -v "$LIT_TEST_DIR/Target/tritongpu_to_llvmir.mlir"
# lit -v "$LIT_TEST_DIR/Target/mlir_to_amdgcn_float16_vectorized_load_store.mlir"
# lit -v "$LIT_TEST_DIR/Target/mlir_to_amdgcn_int16_vectorized_load_store.mlir"
# lit -v "$LIT_TEST_DIR/Target/tritongpu_to_amdgcn.mlir"
# lit -v "$LIT_TEST_DIR/Target/tritongpu_to_hsaco.mlir"
# lit -v "$LIT_TEST_DIR/Target/tritongpu_to_llvmir.mlir"
# lit -v "$LIT_TEST_DIR/Target/tritongpu_to_ptx.mlir?

# triton-opt %s -split-input-file --convert-triton-gpu-to-llvm | FileCheck --check-prefixes=CHECK,GCN %s
