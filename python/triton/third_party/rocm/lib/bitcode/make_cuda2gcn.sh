#!/bin/bash

set -e

pushd .

git clone https://github.com/dfukalov/ROCm-Device-Libs.git
cd ROCm-Device-Libs
git apply ../cuda2gcn.patch
mkdir build
cd build
cmake .. -DCMAKE_PREFIX_PATH=$HOME/.triton/llvm/clang+llvm-14.0.0-x86_64-linux-gnu-ubuntu-18.04
make -j4

popd
cp ROCm-Device-Libs/build/amdgcn/bitcode/cuda2gcn.bc .
rm -rf ROCm-Device-Libs
