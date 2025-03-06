#ifndef CONVERSION_PROTONGPU_TO_LLVM_TARGETINFONVIDIA_H
#define CONVERSION_PROTONGPU_TO_LLVM_TARGETINFONVIDIA_H

#include "Conversion/ProtonGPUToLLVM/TargetInfoBase.h"
#include "third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TargetInfo.h" // TODO(fywkevin): move nvidia TargetInfo.h to include/

namespace mlir::triton::proton::NVIDIA {
class TargetInfo : public mlir::triton::proton::TargetInfoBase {
public:
  explicit TargetInfo(const mlir::triton::NVIDIA::TargetInfo &helper)
      : mlir::triton::proton::TargetInfoBase(helper) {}

  const mlir::triton::NVIDIA::TargetInfo &getTritonTargetInfo() const override {
    return static_cast<const mlir::triton::NVIDIA::TargetInfo &>(helper);
  }

  Value clock(ConversionPatternRewriter &rewriter, Location loc,
              bool isClock64) const override;

  ~TargetInfo() {}
};
} // namespace mlir::triton::proton::NVIDIA

#endif // CONVERSION_PROTONGPU_TO_LLVM_TARGETINFONVIDIA_H
