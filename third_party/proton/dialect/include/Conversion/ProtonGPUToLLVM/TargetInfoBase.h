#ifndef PROTONGPU_TO_LLVM_TARGETINFO_BASE_H
#define PROTONGPU_TO_LLVM_TARGETINFO_BASE_H

#include "mlir/IR/Attributes.h"
#include "triton/Conversion/MLIRTypes.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::triton::proton::gpu {

class TargetInfoBase {
public:
  explicit TargetInfoBase(const mlir::triton::TargetInfoBase &helper)
      : helper(helper) {}

  virtual const mlir::triton::TargetInfoBase &getTritonTargetInfo() const {
    return helper;
  }

  virtual Value clock(ConversionPatternRewriter &rewriter, Location loc,
                      bool isClock64) const = 0;

  virtual Value processorId(ConversionPatternRewriter &rewriter,
                            Location loc) const = 0;

  virtual int getAddressSpace(Attribute addressSpace) const = 0;

  virtual int getIndexPtrAddrSpace() const = 0;

  virtual ~TargetInfoBase() = default;

protected:
  const mlir::triton::TargetInfoBase &helper;
};
} // namespace mlir::triton::proton::gpu

#endif // PROTONGPU_TO_LLVM_TARGETINFO_BASE_H
