#ifndef TRITON_TOOLS_UTILS_H
#define TRITON_TOOLS_UTILS_H

#include "triton/Tools/LinearLayout.h"

#if defined(_MSC_VER) && !defined(__clang__)
// from https://gist.github.com/pps83/3210a2f980fd02bb2ba2e5a1fc4a2ef0
#include <intrin.h>

static int __builtin_ctz(unsigned x) {
  unsigned long r;
  _BitScanForward(&r, x);
  return static_cast<int>(r);
}

static int __builtin_ctzll(unsigned long long x) {
  unsigned long r;
  _BitScanForward64(&r, x);
  return static_cast<int>(r);
}

#endif

namespace mlir::triton {

// Check inDims = ["register", "lane", "warp", "block"]
bool hasRegisterInDims(const LinearLayout &layout);
// Check outDims = ["dim0", "dim1", ...]
bool hasCanonicalOutDims(const LinearLayout &layout);
// Check all bases either have a single non-zero value and it is a power of two
// or are all zeros
bool hasPowerOfTwoBases(const LinearLayout &layout);
// Shortcut for hasRegisterInDims() && hasCanonicalOutDims() &&
// hasPowerOfTwoBases()
bool isDistributedLayout(const LinearLayout &layout);
// Check inDims = ["offset", "iteration"]
bool hasSharedMemoryInDims(const LinearLayout &layout);
// Shortcut for hasSharedMemoryInDims() && hasCanonicalOutDims()
bool isSharedMemoryLayout(const LinearLayout &layout);

// TODO REWRITE
// Convert from `from` to `to`
// Inverts or pseudo-inverts `outer` and composes it with `this`.
//
// Formally, if C = A.invertAndCompose(B), then for all x, C(x) = y implies
// A(x) = B(y), or in other words A(x) = B(C(x)).  If B is invertible, then
// C(x) = B^-1(A(x)), which is how this function gets its name.
//
// For example, suppose you have the following two LLs.
//
//   - R is an LL representing registers, mapping (lane, warp) to a 2D index.
//   - S is an LL representing shared memory, mapping offset to a 2D index.
//
// Suppose you want to store tensor values from registers into shared memory.
// That is, given a (lane, warp), you want to know the corresponding shared
// memory offset to store into.
//
// This is equivalent to converting a (lane, warp) into a 2D index (i.e.
// applying R), then converting a 2D index into a shmem offset (i.e. applying
// the inverse of S).  R.invertAndCompose(S) computes this transformation.
//
// Notice the following requirements in order for this to work.
//
//   - R and S must have the same output dimension names (different order is
//     allowed).
//   - S must be surjective, i.e. there must be some offset for each output
//     dimension of S.  This way when we compose S^-1 with R, every possible
//     2D index that we might get from R has some shmem offset.
//   - The codomain of S must be at least as large as the codomain of R.
//     Otherwise, R could map some tensor index that is not stored in S.
//
// One requirement we *don't* have is that S is injective; we allow two shmem
// offsets to hold the same 2D index.  If S is not injective, there's
// ambiguity in which offset we choose for a given (lane, warp).  For now we
// don't place any guarantees on the choices made by this function.
[[nodiscard]] LinearLayout convertToFrom(const LinearLayout &to,
                                         const LinearLayout &from);
} // namespace mlir::triton

#endif
