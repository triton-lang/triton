#include "Conversion/ProtonGPUToLLVM/TargetInfoBase.h"
#include "llvm/Support/MathExtras.h"

namespace mlir::triton::proton {

Value TargetInfoBase::clock(ConversionPatternRewriter &rewriter, Location loc,
                            bool isClock64) const {
  llvm_unreachable("Not implemented");
  return Value();
}

} // namespace mlir::triton::proton
