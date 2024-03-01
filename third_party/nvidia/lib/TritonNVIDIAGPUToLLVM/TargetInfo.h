#ifndef TRITON_CONVERSION_TRITONGPU_TO_LLVM_TARGETINFONVIDIA_H
#define TRITON_CONVERSION_TRITONGPU_TO_LLVM_TARGETINFONVIDIA_H
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
namespace mlir::triton::NVIDIA {
class TargetInfo : public mlir::triton::TargetInfoBase {
public:
  TargetInfo(int computeCapability) : computeCapability(computeCapability) {}
  // TODO: delete these, we are already inheriting from TargetInfoBase
  bool supportMaximumMinimum() const override;
  Value callBallotOp(ConversionPatternRewriter &rewriter, Location loc,
                     Type type, Value cmp) const override;
  Value storeShared(ConversionPatternRewriter &rewriter, Location loc,
                    Value ptr, Value val, Value pred) const override;
  Value loadShared(ConversionPatternRewriter &rewriter, Location loc, Value ptr,
                   Type elemTy, Value pred) const override;
  Value shflSync(Location loc, ConversionPatternRewriter &rewriter, Value val,
                 int i) const override;
  Value shflUpSync(Location loc, ConversionPatternRewriter &rewriter, Value val,
                   int i) const override;
  Value shflIdxSync(Location loc, ConversionPatternRewriter &rewriter,
                    Value val, int i) const override;
  Value shflIdxSync(Location loc, ConversionPatternRewriter &rewriter,
                    Value val, Value i) const override;
  bool warpReduce(ConversionPatternRewriter &rewriter, Location loc,
                  SmallVector<Value> &acc, triton::ReduceOp op,
                  unsigned numLaneToReduce) const override;

private:
  int computeCapability;
};
} // namespace mlir::triton::NVIDIA
#endif // TRITON_CONVERSION_TRITONGPU_TO_LLVM_TARGETINFONVIDIA_H
