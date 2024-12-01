#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using namespace mlir;
using namespace mlir::triton;

namespace {
class GatherOpConversion : public ConvertOpToLLVMPattern<GatherOp> {
public:
  GatherOpConversion(LLVMTypeConverter &typeConverter,
                     const TargetInfoBase &targetInfo, PatternBenefit benefit)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo) {
  }

  LogicalResult
  matchAndRewrite(GatherOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;

private:
  const TargetInfoBase &targetInfo;
};

LogicalResult
GatherOpConversion::matchAndRewrite(GatherOp op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  RankedTensorType srcType = op.getSrc().getType();

  // Compute the src subtensor shape owned by this CTA.
  SmallVector<unsigned> srcShapePerCTA =
      convertType<unsigned>(triton::gpu::getShapePerCTA(srcType));

  // Grab the src values in this thread.
  SmallVector<Value> srcValues =
      unpackLLElements(loc, adaptor.getSrc(), rewriter);

  // Emit the indices of the src values owned by this thread.
  SmallVector<SmallVector<Value>> srcIndices =
      emitIndices(loc, rewriter, targetInfo, srcType.getEncoding(),
                  op.getSrc().getType(), /*withCTAOffset=*/true);

  // Store the src values owned by the thread into their respective location in
  // the scratch memory.
  assert(srcValues.size() == srcIndices.size());

  // Get the base pointer to the scratch memory.
  Value smemBase = LLVM::getSharedMemoryBase(loc, rewriter, targetInfo, op);

  // For each src element owned by the thread, index into the scratch memory and
  // then store it.
  Type elemType = getTypeConverter()->convertType(srcType.getElementType());
  for (auto [value, indices] : llvm::zip(srcValues, srcIndices)) {
    // Convert the index at each dim into a single offset given the shape of the
    // tensor.
    Value offset = LLVM::linearize(rewriter, loc, indices, srcShapePerCTA);
    // Emit the offset into the shared memory and then store the value.
    Value ptr = gep(smemBase.getType(), elemType, smemBase, offset);
    store(value, ptr);
  }

  // Synchronize the whole CTA.
  barrier();

  // Grab the index values owned by this thread.
  SmallVector<Value> idxValues =
      unpackLLElements(loc, adaptor.getIndices(), rewriter);

  // Apply the layout of the destination tensor to obtain the indices of the
  // column to gather along, then for each column, replace the index along the
  // gather axis with the appropriate index value.
  //
  // I = LL(pid)
  // idx = indices[I]
  // I_gather = [I[d] if d != axis else idx for d in range(len(I))]
  // out[I] = src[I_gather]
  RankedTensorType dstType = op.getType();
  SmallVector<SmallVector<Value>> dstIndices =
      emitIndices(loc, rewriter, targetInfo, dstType.getEncoding(), dstType,
                  /*withCTAOffset=*/true);

  unsigned idxWidth = op.getIndices().getType().getElementTypeBitWidth();
  unsigned axis = op.getAxis();
  SmallVector<Value> results(dstIndices.size());
  for (auto [i, idx, indices] : llvm::enumerate(idxValues, dstIndices)) {
    // The LL index computations are performed with 32 bit integers. If the
    // indices are something else, cast them to i32.
    if (idxWidth > 32) {
      idx = trunc(i32_ty, idx);
    } else if (idxWidth < 32) {
      // Negative indices don't make sense, so zero-extend.
      idx = zext(i32_ty, idx);
    }
    indices[axis] = idx;
    Value offset = LLVM::linearize(rewriter, loc, indices, srcShapePerCTA);
    Value ptr = gep(smemBase.getType(), elemType, smemBase, offset);
    results[i] = load(elemType, ptr);
  }

  Value packed =
      packLLElements(loc, getTypeConverter(), results, rewriter, dstType);
  rewriter.replaceOp(op, packed);
  return success();
}

} // namespace

void triton::populateGatherOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                            RewritePatternSet &patterns,
                                            const TargetInfoBase &targetInfo,
                                            PatternBenefit benefit) {
  patterns.insert<GatherOpConversion>(typeConverter, targetInfo, benefit);
}
