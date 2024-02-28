#include "TargetInfo.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

namespace mlir::triton::NVIDIA {
bool TargetInfo::isSupported() const { return computeCapability >= 80; }
Value TargetInfo::callBallotOp(ConversionPatternRewriter &rewriter,
                               Location loc, Type type, Value threadMask,
                               Value cmp) const {
  return rewriter.create<NVVM::VoteBallotOp>(loc, type, threadMask, cmp);
}
} // namespace mlir::triton::NVIDIA
