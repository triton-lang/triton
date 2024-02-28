#ifndef TRITON_CONVERSION_TRITONGPU_TO_LLVM_TARGETINFOBASE_H
#define TRITON_CONVERSION_TRITONGPU_TO_LLVM_TARGETINFOBASE_H

#include "triton/Conversion/MLIRTypes.h"

using namespace mlir;
namespace mlir::triton {
class TargetInfoBase {
public:
  virtual bool isSupported() const = 0;
  virtual Value callBallotOp(ConversionPatternRewriter &rewriter, Location loc,
                             Value threadMask, Value cmp) const = 0;
  virtual ~TargetInfoBase() {}
};
} // namespace mlir::triton
#endif // TRITON_CONVERSION_TRITONGPU_TO_LLVM_TARGETINFOBASE_H
