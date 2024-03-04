#ifndef TRITON_CONVERSION_TRITONGPU_TO_LLVM_TARGETINFOAMD_H
#define TRITON_CONVERSION_TRITONGPU_TO_LLVM_TARGETINFOAMD_H
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include <string>
namespace AMD {
class TargetInfo : public mlir::triton::TargetInfoBase {
public:
  TargetInfo(std::string arch) : arch(arch) {}
  bool supportMaximumMinimum() const override;
  Value ballot(ConversionPatternRewriter &rewriter, Location loc, Type type,
               Value cmp) const override;
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
  unsigned getShuffleIndex(triton::ReduceOp op, unsigned N,
                           unsigned numLaneToReduce) const override;

private:
  std::string arch;
};
} // namespace AMD
#endif // TRITON_CONVERSION_TRITONGPU_TO_LLVM_TARGETINFOAMD_H