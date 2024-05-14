#ifndef TRITON_CONVERSION_TRITONCPU_TO_LLVM_TARGETINFOBASE_H
#define TRITON_CONVERSION_TRITONCPU_TO_LLVM_TARGETINFOBASE_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Conversion/MLIRTypes.h"

namespace mlir::triton::cpu {
class CPUTargetInfo {
public:
  // Note: we may revisit for different CPU ISAs like AVX and Neon.
  CPUTargetInfo() {}

  Value programId(ConversionPatternRewriter &rewriter, Location loc,
                  LLVM::LLVMFuncOp funcOp, int axis) const;

  void printf(ConversionPatternRewriter &rewriter, Value formatStrStart,
              int formatStrByteCount, ValueRange args) const;

  ~CPUTargetInfo() {}
};
} // namespace mlir::triton::cpu
#endif // TRITON_CONVERSION_TRITONCPU_TO_LLVM_TARGETINFOBASE_H
