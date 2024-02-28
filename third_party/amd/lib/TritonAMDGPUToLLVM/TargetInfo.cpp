#include "TargetInfo.h"
#include "amd/include/TritonAMDGPUToLLVM/GCNAsmFormat.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

namespace AMD {
bool TargetInfo::isSupported() const { return false; }
Value TargetInfo::callBallotOp(ConversionPatternRewriter &rewriter,
                               Location loc, Type type, Value threadMask,
                               Value cmp) const {
  auto stringAttr = rewriter.getStringAttr("llvm.amdgcn.ballot");
  SmallVector<Value> operands = {cmp};
  Value asmResult =
      rewriter.create<LLVM::CallIntrinsicOp>(loc, type, stringAttr, operands)
          ->getResult(0);
  return asmResult;
}
} // namespace AMD