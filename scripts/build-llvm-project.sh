#!/usr/bin/env bash

REPO_ROOT="$(git rev-parse --show-toplevel)"

LLVM_TARGETS=${LLVM_TARGETS:-Native;NVPTX;AMDGPU}
LLVM_PROJECTS=${LLVM_PROJECTS:-mlir;llvm;lld}
LLVM_BUILD_TYPE=${LLVM_BUILD_TYPE:-RelWithDebInfo}
LLVM_COMMIT_HASH=${LLVM_COMMIT_HASH:-$(cat "$REPO_ROOT/cmake/llvm-hash.txt")}
LLVM_PROJECT_PATH=${LLVM_PROJECT_PATH:-"$REPO_ROOT/llvm-project"}
LLVM_BUILD_PATH=${LLVM_BUILD_PATH:-"$LLVM_PROJECT_PATH/build"}
LLVM_INSTALL_PATH=${LLVM_INSTALL_PATH:-"$LLVM_PROJECT_PATH/install"}
LLVM_PROJECT_URL=${LLVM_PROJECT_URL:-"https://github.com/triton-lang/llvm-project"}

if [ -z "$CMAKE_ARGS" ]; then
    if [ "$#" -eq 0 ]; then
        CMAKE_ARGS=(
            -G Ninja
              -DCMAKE_BUILD_TYPE="$LLVM_BUILD_TYPE"
              -DLLVM_CCACHE_BUILD=OFF
              -DLLVM_ENABLE_ASSERTIONS=ON
              -DCMAKE_C_COMPILER=clang
              -DCMAKE_CXX_COMPILER=clang++
              -DLLVM_ENABLE_LLD=ON
              -DLLVM_OPTIMIZED_TABLEGEN=ON
              -DLLVM_TARGETS_TO_BUILD="$LLVM_TARGETS"
              -DCMAKE_EXPORT_COMPILE_COMMANDS=1
              -DLLVM_ENABLE_PROJECTS="$LLVM_PROJECTS"
              -DCMAKE_INSTALL_PREFIX="$LLVM_INSTALL_PATH"
              -B"$LLVM_BUILD_PATH" "$LLVM_PROJECT_PATH/llvm"
        )
    else
        CMAKE_ARGS=("$@")
    fi
fi

if [ -n "$LLVM_CLEAN" ] && [ -e "$LLVM_PROJECT_PATH" ]; then
    rm -rf "$LLVM_PROJECT_PATH"
fi

if [ ! -e "$LLVM_PROJECT_PATH" ]; then
    echo "Cloning from $LLVM_PROJECT_URL"
    git clone "$LLVM_PROJECT_URL" "$LLVM_PROJECT_PATH"
fi
echo "Resetting to $LLVM_COMMIT_HASH"
git -C "$LLVM_PROJECT_PATH" fetch origin "$LLVM_COMMIT_HASH"
git -C "$LLVM_PROJECT_PATH" reset --hard "$LLVM_COMMIT_HASH"
echo "Configuring with ${CMAKE_ARGS[@]}"
cmake "${CMAKE_ARGS[@]}"
echo "Building LLVM"
ninja -C "$LLVM_BUILD_PATH"
