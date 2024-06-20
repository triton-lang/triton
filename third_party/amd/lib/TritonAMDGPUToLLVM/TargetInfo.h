#ifndef TRITON_CONVERSION_TRITONGPU_TO_LLVM_TARGETINFOAMD_H
#define TRITON_CONVERSION_TRITONGPU_TO_LLVM_TARGETINFOAMD_H

#include "TritonAMDGPUToLLVM/TargetUtils.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include <string>

namespace mlir::triton::AMD {
class TargetInfo : public mlir::triton::TargetInfoBase {
public:
  explicit TargetInfo(std::string arch) : arch(std::move(arch)) {}

  ISAFamily getISAFamily() const { return deduceISAFamily(arch); }

  int getSharedMemorySize() const;

  bool supportMaximumMinimum() const override;

  Value getClusterCTAId(RewriterBase &rewriter, Location loc) const override;

  Value ballot(ConversionPatternRewriter &rewriter, Location loc, Type type,
               Value cmp) const override;

  void storeShared(ConversionPatternRewriter &rewriter, Location loc, Value ptr,
                   Value val, Value pred) const override;
  Value loadShared(ConversionPatternRewriter &rewriter, Location loc,
                   const TypeConverter *converter, Value ptr, Type elemTy,
                   Value pred) const override;

  Value shuffleXor(ConversionPatternRewriter &rewriter, Location loc, Value val,
                   int i) const override;
  Value shuffleUp(ConversionPatternRewriter &rewriter, Location loc, Value val,
                  int i) const override;
  Value shuffleIdx(ConversionPatternRewriter &rewriter, Location loc, Value val,
                   int i) const override;
  Value shuffleIdx(ConversionPatternRewriter &rewriter, Location loc, Value val,
                   Value i) const override;

  Value programId(ConversionPatternRewriter &rewriter, Location loc,
                  ModuleOp moduleOp, int axis) const override;

  bool warpReduce(ConversionPatternRewriter &rewriter, Location loc,
                  SmallVector<Value> &acc, triton::ReduceOp op,
                  unsigned numLaneToReduce) const override;

  bool processReplicaUsingStMatrix(
      ConversionPatternRewriter &rewriter, Location loc, Value smemBase,
      SmallVector<Value> &vals, RankedTensorType srcTy, Type elemTy,
      ArrayRef<unsigned> paddedRepShape, ArrayRef<unsigned> origRepShape,
      ArrayRef<unsigned> outOrd, unsigned accumNumReplicates,
      int swizzleByteWidth) const override;

  std::string getMulhiFuncName(Type resultElementTy) const override;

  void printf(ConversionPatternRewriter &rewriter, Value formatStrStart,
              int formatStrByteCount, ValueRange args) const override;
  void assertFail(ConversionPatternRewriter &rewriter, Location loc,
                  StringRef message, StringRef file, StringRef func,
                  int line) const override;

  bool enableLinearLayout() const override { return false; }

private:
  void printfImpl(Value formatStrStart, int formatStrByteCount, ValueRange args,
                  ConversionPatternRewriter &rewriter, bool useStdErr) const;

  std::string arch;
};
} // namespace mlir::triton::AMD

#endif // TRITON_CONVERSION_TRITONGPU_TO_LLVM_TARGETINFOAMD_H
