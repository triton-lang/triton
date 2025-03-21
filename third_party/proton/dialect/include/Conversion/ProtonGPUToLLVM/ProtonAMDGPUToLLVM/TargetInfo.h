#ifndef PROTONGPU_TO_LLVM_TARGETINFO_AMD_H
#define PROTONGPU_TO_LLVM_TARGETINFO_AMD_H

#include "Conversion/ProtonGPUToLLVM/TargetInfoBase.h"
#include "third_party/amd/lib/TritonAMDGPUToLLVM/TargetInfo.h" // TODO(fywkevin): move amd TargetInfo.h to include/
#include <string>

namespace mlir::triton::proton::gpu::AMD {
class TargetInfo : public mlir::triton::proton::gpu::TargetInfoBase {
public:
  explicit TargetInfo(const mlir::triton::AMD::TargetInfo &helper)
      : mlir::triton::proton::gpu::TargetInfoBase(helper) {}

  const mlir::triton::AMD::TargetInfo &getTritonTargetInfo() const override {
    return static_cast<const mlir::triton::AMD::TargetInfo &>(helper);
  }

  Value clock(ConversionPatternRewriter &rewriter, Location loc,
              bool isClock64) const override;

  ~TargetInfo() = default;
};
} // namespace mlir::triton::proton::gpu::AMD

#endif // PROTONGPU_TO_LLVM_TARGETINFO_AMD_H
