#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AMD_LLVM_BUILD_INFO="$REPO_ROOT/cmake/amd-llvm-build-info.json"
LLVM_COMMIT_HASH="$(jq -er '.llvm_hash' "$AMD_LLVM_BUILD_INFO")"
LLVM_REPOSITORY="$(jq -er '.repository' "$AMD_LLVM_BUILD_INFO")"
LLVM_BUILD_NUMBER="$(jq -er '.build_number' "$AMD_LLVM_BUILD_INFO")"

LLVM_PROJECT_PATH="${AMD_LLVM_PROJECT_PATH:-"$REPO_ROOT/llvm-project-amd"}"
LLVM_BUILD_PATH="${AMD_LLVM_BUILD_PATH:-"$REPO_ROOT/.llvm-project-amd/build"}"
AMD_BACKEND_PATH="$REPO_ROOT/third_party/amd/backend"

if [[ ! -e "$LLVM_PROJECT_PATH/.git" ]]; then
    git init --quiet "$LLVM_PROJECT_PATH"
fi
if ! git -C "$LLVM_PROJECT_PATH" cat-file -e "$LLVM_COMMIT_HASH^{commit}" 2>/dev/null; then
    git -C "$LLVM_PROJECT_PATH" fetch --depth=1 "https://github.com/$LLVM_REPOSITORY.git" "$LLVM_COMMIT_HASH"
fi
git -C "$LLVM_PROJECT_PATH" checkout --detach "$LLVM_COMMIT_HASH"

mkdir -p "$LLVM_BUILD_PATH"
LLVM_BUILD_PATH="$(cd "$LLVM_BUILD_PATH" && pwd)"
LLVM_INSTALL_PATH="$LLVM_BUILD_PATH/install"

# Build+link with clang+lld only when explicitly requested (the macOS toolchain
# does not ship lld). Otherwise fall back to the default compiler and linker.
CLANG_LLD_ARGS=()
CLANG_ARGS=()
case "$(printf '%s' "${TRITON_BUILD_WITH_CLANG_LLD:-}" | tr '[:upper:]' '[:lower:]')" in
    1 | on | true) TRITON_BUILD_WITH_CLANG_LLD=1 ;;
esac
if [ "${TRITON_BUILD_WITH_CLANG_LLD:-}" = "1" ]; then
    CLANG_ARGS=(
        -DCMAKE_C_COMPILER=clang
        -DCMAKE_CXX_COMPILER=clang++
    )
    CLANG_LLD_ARGS=("${CLANG_ARGS[@]}" -DLLVM_ENABLE_LLD=ON)
fi

cmake -G Ninja -S "$LLVM_PROJECT_PATH/llvm" -B "$LLVM_BUILD_PATH" \
    "${CLANG_LLD_ARGS[@]}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$LLVM_INSTALL_PATH" \
    -DCMAKE_INSTALL_LIBDIR=lib \
    -DLLVM_TARGETS_TO_BUILD=AMDGPU \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_INCLUDE_TESTS=OFF \
    -DLLVM_ENABLE_PIC=ON \
    -DBUILD_SHARED_LIBS=OFF \
    -DLLVM_BUILD_LLVM_DYLIB=OFF \
    -DLLVM_LINK_LLVM_DYLIB=OFF \
    -DLLVM_ENABLE_ZSTD=OFF \
    -DLLVM_ENABLE_LZMA=OFF \
    '-DLLVM_DISTRIBUTION_COMPONENTS=llvm-headers;llvm-libraries;cmake-exports'
cmake --build "$LLVM_BUILD_PATH" --target install-distribution

cmake -G Ninja -S "$AMD_BACKEND_PATH/codegen" -B "$LLVM_BUILD_PATH/codegen" \
    "${CLANG_ARGS[@]}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_DIR="$LLVM_INSTALL_PATH/lib/cmake/llvm" \
    -DCMAKE_INSTALL_PREFIX="$AMD_BACKEND_PATH" \
    -DTRITON_AMD_LLVM_REVISION="$LLVM_COMMIT_HASH-$LLVM_BUILD_NUMBER"
cmake --build "$LLVM_BUILD_PATH/codegen" --target install
