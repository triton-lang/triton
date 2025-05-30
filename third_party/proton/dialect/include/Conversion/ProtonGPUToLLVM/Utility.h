#ifndef PROTONGPU_TO_LLVM_UTILITY_H
#define PROTONGPU_TO_LLVM_UTILITY_H

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::LLVM {

struct SegmentObject {
  Value base;
  Value segmentBase;
  Value indexPtr;

  SegmentObject(Value base, Value segmentBase, Value indexPtr)
      : base(base), segmentBase(segmentBase), indexPtr(indexPtr) {}

  Value getStruct(Location loc, ConversionPatternRewriter &rewriter);

  static LLVMStructType getStructType(MLIRContext *ctx, int memorySpace,
                                      int indexPtrAddrSpace);

  static SegmentObject fromStruct(Location loc, Value segmentStruct,
                                  ConversionPatternRewriter &rewriter);
};

} // namespace mlir::LLVM

#endif // PROTONGPU_TO_LLVM_UTILITY_H
