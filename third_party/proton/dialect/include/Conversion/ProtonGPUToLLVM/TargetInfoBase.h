#ifndef CONVERSION_PROTONGPU_TO_LLVM_TARGETINFOBASE_H
#define CONVERSION_PROTONGPU_TO_LLVM_TARGETINFOBASE_H

#include "triton/Conversion/MLIRTypes.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"

namespace mlir::triton::proton {

class TargetInfoBase {
public:
  explicit TargetInfoBase(const mlir::triton::TargetInfoBase &helper)
      : helper(helper) {}

  virtual const mlir::triton::TargetInfoBase &getTritonTargetInfo() const {
    return helper;
  }

  virtual Value clock(ConversionPatternRewriter &rewriter, Location loc,
                      bool isClock64) const;

  virtual ~TargetInfoBase() {}

protected:
  const mlir::triton::TargetInfoBase &helper;
};
} // namespace mlir::triton::proton
#endif // CONVERSION_PROTONGPU_TO_LLVM_TARGETINFOBASE_H
