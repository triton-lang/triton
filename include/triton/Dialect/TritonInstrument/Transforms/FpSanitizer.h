#ifndef TRITONINSTRUMENT_FPSANITIZER_H
#define TRITONINSTRUMENT_FPSANITIZER_H

#include "mlir/IR/PatternMatch.h"

namespace mlir::triton::instrument {

// Casts the floating-point payload v to the payload type targetTy.
// homomorphicCasts truncates instead of folding the discarded bits when
// narrowing. isUnsigned treats payloads as zero-extended, such as the raw
// nibbles unpacked by Fp4ToFpOp.
Value castFloatPayloadToType(PatternRewriter &rewriter, Location loc, Value v,
                             Type targetTy, bool homomorphicCasts,
                             bool isUnsigned = false);

} // namespace mlir::triton::instrument

#endif // TRITONINSTRUMENT_FPSANITIZER_H
