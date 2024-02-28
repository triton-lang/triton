#ifndef TRITON_CONVERSION_TRITONGPU_TO_LLVM_TARGETINFONVIDIA_H
#define TRITON_CONVERSION_TRITONGPU_TO_LLVM_TARGETINFONVIDIA_H
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
namespace mlir::triton::NVIDIA {
class TargetInfo : public mlir::triton::TargetInfoBase {
public:
  TargetInfo(int computeCapability) : computeCapability(computeCapability) {}
  bool isSupported() const override;

private:
  int computeCapability;
};
} // namespace mlir::triton::NVIDIA
#endif // TRITON_CONVERSION_TRITONGPU_TO_LLVM_TARGETINFONVIDIA_H
