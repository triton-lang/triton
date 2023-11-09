export TRITON_CODEGEN_TRITON_SHARED=1
export TRITON_BUILD_WITH_CLANG_LLD=true
export LLVM_BUILD_DIR=$HOME/github/llvm-project/build

cd /home/nhat/github/triton/python
# python3 -m pip install --upgrade pip
# python3 -m pip install cmake==3.24
# python3 -m pip install ninja
# python3 -m pip uninstall -y triton

LLVM_INCLUDE_DIRS=$LLVM_BUILD_DIR/include \
LLVM_LIBRARY_DIR=$LLVM_BUILD_DIR/lib \
LLVM_SYSPATH=$LLVM_BUILD_DIR \
    python3 setup.py build

# LLVM_INCLUDE_DIRS=$LLVM_BUILD_DIR/include \
# LLVM_LIBRARY_DIR=$LLVM_BUILD_DIR/lib \
# LLVM_SYSPATH=$LLVM_BUILD_DIR \
#     python3 -m pip install --no-build-isolation -vvv '.[tests]'
