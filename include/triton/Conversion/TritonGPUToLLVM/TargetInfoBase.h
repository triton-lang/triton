#ifndef TRITON_CONVERSION_TRITONGPU_TO_LLVM_TARGETINFOBASE_H
#define TRITON_CONVERSION_TRITONGPU_TO_LLVM_TARGETINFOBASE_H

#include "triton/Conversion/MLIRTypes.h"

namespace mlir::triton {
class TargetInfoBase {
public:
  virtual bool supportMaximumMinimum() const = 0;

  virtual Value getClusterCTAId(RewriterBase &rewriter, Location loc) const = 0;

  virtual Value ballot(ConversionPatternRewriter &rewriter, Location loc,
                       Type type, Value cmp) const = 0;

  virtual void storeShared(ConversionPatternRewriter &rewriter, Location loc,
                           Value ptr, Value val, Value pred) const = 0;
  virtual Value loadShared(ConversionPatternRewriter &rewriter, Location loc,
                           const TypeConverter *converter, Value ptr,
                           Type elemTy, Value pred) const = 0;

  virtual Value shuffleXor(ConversionPatternRewriter &rewriter, Location loc,
                           Value val, int i) const = 0;
  virtual Value shuffleUp(ConversionPatternRewriter &rewriter, Location loc,
                          Value val, int i) const = 0;
  virtual Value shuffleIdx(ConversionPatternRewriter &rewriter, Location loc,
                           Value val, int i) const = 0;
  virtual Value shuffleIdx(ConversionPatternRewriter &rewriter, Location loc,
                           Value val, Value i) const = 0;

  virtual Value programId(ConversionPatternRewriter &rewriter, Location loc,
                          ModuleOp moduleOp, int axis) const = 0;

  virtual bool warpReduce(ConversionPatternRewriter &rewriter, Location loc,
                          SmallVector<Value> &acc, triton::ReduceOp op,
                          unsigned numLaneToReduce) const = 0;

  virtual bool processReplicaUsingStMatrix(
      ConversionPatternRewriter &rewriter, Location loc, Value smemBase,
      SmallVector<Value> &vals, RankedTensorType srcTy, Type elemTy,
      ArrayRef<unsigned> paddedRepShape, ArrayRef<unsigned> origRepShape,
      ArrayRef<unsigned> outOrd, unsigned accumNumReplicates,
      int swizzleByteWidth = 0) const = 0;

  virtual std::string getMulhiFuncName(Type resultElementTy) const = 0;
  // Emits LLVM code with |rewriter| to print a message following the given
  // format from the device. |formatStrStart| is the pointer to the start of
  // the format string global variable; |args| are the arguments to fill
  // placeholders in the format string.
  virtual void printf(ConversionPatternRewriter &rewriter, Value formatStrStart,
                      int formatStrByteCount, ValueRange args) const = 0;
  // Emits LLVM code with |rewriter| to perform assertion failure with the given
  // |message| from the given |func| in |file|.
  virtual void assertFail(ConversionPatternRewriter &rewriter, Location loc,
                          StringRef message, StringRef file, StringRef func,
                          int line) const = 0;

  virtual ~TargetInfoBase() {}
};
} // namespace mlir::triton
#endif // TRITON_CONVERSION_TRITONGPU_TO_LLVM_TARGETINFOBASE_H
