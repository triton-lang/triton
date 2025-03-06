#include "Conversion/ProtonGPUToLLVM/ProtonAMDGPUToLLVM/TargetInfo.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "llvm/Support/MathExtras.h"

namespace mlir::triton::proton::AMD {

Value TargetInfo::clock(ConversionPatternRewriter &rewriter, Location loc,
                        bool isClock64) const {

  llvm_unreachable("AMDGPU clock not implemented yet");
  return Value();
}

} // namespace mlir::triton::proton::AMD
