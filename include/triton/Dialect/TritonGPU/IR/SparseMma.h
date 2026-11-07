#ifndef TRITON_DIALECT_TRITONGPU_IR_SPARSEMMA_H
#define TRITON_DIALECT_TRITONGPU_IR_SPARSEMMA_H

#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include <array>

namespace mlir::triton::gpu {
// K extents are in operand elements, including individual FP4 values.
struct SparseMmaKShape {
  int64_t storedA;
  int64_t logical;

  bool matches(int64_t a, int64_t b) const {
    return a == storedA && b == logical;
  }
};

inline constexpr SparseMmaKShape kF16Sparse2To4{16, 32};
inline constexpr SparseMmaKShape kFp8Sparse2To4{32, 64};
inline constexpr SparseMmaKShape kNvfp4Sparse4To8{64, 128};

// Register layouts for A, B, sparsity metadata and D of a warp-level sparse
// MMA.
std::array<Attribute, 4> getWarpSparseMmaLayouts(MLIRContext *ctx,
                                                 unsigned elementBitWidth,
                                                 ArrayRef<int64_t> shape,
                                                 int numWarps);
Value buildWarpSparseDot(OpBuilder &builder, Location loc, Value a, Value b,
                         Value sparsityMetadata, Value acc, int numWarps);
} // namespace mlir::triton::gpu
#endif
