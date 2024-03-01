#ifndef TRITON_CONVERSION_TRITONGPU_TO_LLVM_TARGETINFOAMD_H
#define TRITON_CONVERSION_TRITONGPU_TO_LLVM_TARGETINFOAMD_H
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include <string>
namespace AMD {
class TargetInfo : public mlir::triton::TargetInfoBase {
public:
  TargetInfo(std::string arch) : arch(arch) {}
  bool supportMaximumMinimum() const override;
  Value callBallotOp(ConversionPatternRewriter &rewriter, Location loc,
                     Type type, Value cmp) const override;
  virtual void storeShared(ConversionPatternRewriter &rewriter, Location loc,
                           Value ptr, Value val, Value pred) const = 0;
  virtual Value loadShared(ConversionPatternRewriter &rewriter, Location loc,
                           Value ptr, Type elemTy, Value pred) const = 0;
  virtual Value shflSync(Location loc, ConversionPatternRewriter &rewriter,
                         Value val, int i) const = 0;
  virtual Value shflUpSync(Location loc, ConversionPatternRewriter &rewriter,
                           Value val, int i) const = 0;
  virtual Value shflIdxSync(Location loc, ConversionPatternRewriter &rewriter,
                            Value val, int i) const = 0;
  virtual Value shflIdxSync(Location loc, ConversionPatternRewriter &rewriter,
                            Value val, Value i) const = 0;
private:
  std::string arch;
};
} // namespace AMD
#endif // TRITON_CONVERSION_TRITONGPU_TO_LLVM_TARGETINFOAMD_H