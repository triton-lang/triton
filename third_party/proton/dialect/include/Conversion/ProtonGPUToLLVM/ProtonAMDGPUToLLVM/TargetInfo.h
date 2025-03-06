#ifndef CONVERSION_PROTONGPU_TO_LLVM_TARGETINFOAMD_H
#define CONVERSION_PROTONGPU_TO_LLVM_TARGETINFOAMD_H

#include "Conversion/ProtonGPUToLLVM/TargetInfoBase.h"
#include "third_party/amd/lib/TritonAMDGPUToLLVM/TargetInfo.h" // TODO(fywkevin): move amd TargetInfo.h to include/
#include <string>

namespace mlir::triton::proton::AMD {
class TargetInfo : public mlir::triton::proton::TargetInfoBase {
public:
  explicit TargetInfo(const mlir::triton::AMD::TargetInfo &helper)
      : mlir::triton::proton::TargetInfoBase(helper) {}

  const mlir::triton::AMD::TargetInfo &getTritonTargetInfo() const override {
    return static_cast<const mlir::triton::AMD::TargetInfo &>(helper);
  }

  Value clock(ConversionPatternRewriter &rewriter, Location loc,
              bool isClock64) const override;

  ~TargetInfo() {}
};
} // namespace mlir::triton::proton::AMD

#endif // CONVERSION_PROTONGPU_TO_LLVM_TARGETINFOAMD_H
