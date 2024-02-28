#include "TargetInfo.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

namespace AMD {
bool TargetInfo::isSupported() const { return false; }
Value TargetInfo::callBallotOp(ConversionPatternRewriter &rewriter,
                               Location loc, Value threadMask,
                               Value cmp) const {
  return i32_val(0);
}
} // namespace AMD