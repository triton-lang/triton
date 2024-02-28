#ifndef TRITON_CONVERSION_TRITONGPU_TO_LLVM_TARGETINFOAMD_H
#define TRITON_CONVERSION_TRITONGPU_TO_LLVM_TARGETINFOAMD_H
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include <string>
namespace AMD {
class TargetInfo : public mlir::triton::TargetInfoBase {
public:
  TargetInfo(std::string arch) : arch(arch) {}
  bool isSupported() const override;
  Value callBallotOp(ConversionPatternRewriter &rewriter, Location loc,
                     Type type, Value threadMask, Value cmp) const override;

private:
  std::string arch;
};
} // namespace AMD
#endif // TRITON_CONVERSION_TRITONGPU_TO_LLVM_TARGETINFOAMD_H