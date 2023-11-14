cd $HOME/github/llvm-project  # your clone of LLVM.
mkdir build
cd build
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON  ../llvm -DLLVM_ENABLE_PROJECTS="mlir;llvm"
ninja
