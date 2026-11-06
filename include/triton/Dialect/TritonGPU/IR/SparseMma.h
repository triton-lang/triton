#ifndef TRITON_DIALECT_TRITONGPU_IR_SPARSEMMA_H
#define TRITON_DIALECT_TRITONGPU_IR_SPARSEMMA_H

#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include <array>

namespace mlir::triton::gpu {
// Register layouts for A, B, metadata and D of a warp-level sparse MMA.
std::array<Attribute, 4> getWarpSparseMmaLayouts(MLIRContext *ctx, int width,
                                                 ArrayRef<int64_t> shape,
                                                 int numWarps);
Value buildWarpSparseDot(OpBuilder &builder, Location loc, Value a, Value b,
                         Value metadata, Value acc, int numWarps);
} // namespace mlir::triton::gpu
#endif
