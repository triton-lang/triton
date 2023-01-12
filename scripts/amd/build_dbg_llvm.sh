#!/bin/bash

# The script gets latest LLVM 14 release, builds it and
# installs to the `build_dbg/install` folder. After that
# one can use the installation with `LLVM_SYSPATH` env
# variable to override default paths used in
# `triton/python/setup.py`

# Notes:
# 1. The whole folder will temporary get ~95GB disk space.
# After the last clean command it will be schrinked to ~38GB.
#
# 2. Default ld linker gets a lot of memory for debug build,
# so number of parallel linker jobs reduced to 4 with
# `LLVM_PARALLEL_LINK_JOBS` option. It works with ninja
# build system only. Additionally `LLVM_PARALLEL_COMPILE_JOBS`
# can be used to reduce parallel compilers processes.

git clone https://github.com/llvm/llvm-project.git llvm14
cd llvm14
git checkout llvmorg-14.0.6
mkdir build_dbg
cd build_dbg
cmake ../llvm -G Ninja \
   -DCMAKE_BUILD_TYPE=Debug \
   -DCMAKE_INSTALL_PREFIX=${PWD}/install \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_BUILD_RUNTIME=OFF \
   -DLLVM_PARALLEL_LINK_JOBS=4 \
   -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU"

cmake --build .
cmake --install .
cmake --build . --target clean
