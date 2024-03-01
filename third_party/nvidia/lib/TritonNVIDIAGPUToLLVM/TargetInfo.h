#ifndef TRITON_CONVERSION_TRITONGPU_TO_LLVM_TARGETINFONVIDIA_H
#define TRITON_CONVERSION_TRITONGPU_TO_LLVM_TARGETINFONVIDIA_H
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
namespace mlir::triton::NVIDIA {
class TargetInfo : public mlir::triton::TargetInfoBase {
public:
  TargetInfo(int computeCapability) : computeCapability(computeCapability) {}
  bool supportMaximumMinimum() const override;
  Value callBallotOp(ConversionPatternRewriter &rewriter, Location loc,
                     Type type, Value cmp) const override;

private:
  int computeCapability;
};
} // namespace mlir::triton::NVIDIA
#endif // TRITON_CONVERSION_TRITONGPU_TO_LLVM_TARGETINFONVIDIA_H
