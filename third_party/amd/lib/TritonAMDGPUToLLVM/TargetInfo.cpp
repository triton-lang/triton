#include "TargetInfo.h"
#include "amd/include/TritonAMDGPUToLLVM/GCNAsmFormat.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

namespace AMD {
bool TargetInfo::isSupported() const { return false; }
Value TargetInfo::callBallotOp(ConversionPatternRewriter &rewriter,
                               Location loc, Value threadMask,
                               Value cmp) const {
  auto stringAttr = rewriter.getStringAttr("llvm.amdgcn.ballot");
  SmallVector<Value> operands = {threadMask, cmp};
  Value asmResult =
      rewriter.create<LLVM::CallIntrinsicOp>(loc, i32_ty, stringAttr, operands)
          ->getResult(0);
  return asmResult;
}
} // namespace AMD