python3 -m pip install lit
cd python
LIT_TEST_DIR="build/$(ls build)/test"
if [ ! -d "$LIT_TEST_DIR" ]; then
    echo "Not found $($LIT_TEST_DIR).  Did you change an installation method?"
    exit -1
fi

lit -v "$LIT_TEST_DIR"
# lit -v "$LIT_TEST_DIR/Conversion/AMDGPU/load_store.mlir"
# lit -v "$LIT_TEST_DIR/Conversion/tritongpu_to_llvm.mlir"
# lit -v "$LIT_TEST_DIR/Target/mlir_to_amdgcn_float16_vectorized_load_store.mlir"
# lit -v "$LIT_TEST_DIR/Target/tritongpu_to_llvmir.mlir"

# triton-opt %s -split-input-file --convert-triton-gpu-to-llvm | FileCheck --check-prefixes=CHECK,GCN %s
