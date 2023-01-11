#ifndef TRITON_LIB_DIALECT_TRITONGPU_TRANSFORMS_UTILITY_H_
#define TRITON_LIB_DIALECT_TRITONGPU_TRANSFORMS_UTILITY_H_
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {

void addFixupLoopPattern(MLIRContext *ctx, RewritePatternSet &patterns);

} // namespace mlir

#endif // TRITON_LIB_DIALECT_TRITONGPU_TRANSFORMS_UTILITY_H_
