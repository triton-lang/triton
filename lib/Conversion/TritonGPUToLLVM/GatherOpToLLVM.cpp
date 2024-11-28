#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

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
  // Codegen the gather by storing the source tensor into shared memory and then
  // gathering directly from shared memory.
  void emitGatherInShared(GatherOp op, OpAdaptor adaptor,
                          ConversionPatternRewriter &rewriter) const;
  // Codegen a warp-local gather by shuffling elements across the warp and
  // selecting from them.
  void emitWarpLocalGather(GatherOp op, OpAdaptor adaptor,
                           ConversionPatternRewriter &rewriter) const;

  const TargetInfoBase &targetInfo;
};

LogicalResult
GatherOpConversion::matchAndRewrite(GatherOp op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
  GatherLoweringHelper helper(op);
  // Specialize the lowering based on the source layout.
  if (helper.isWarpLocal()) {
    emitWarpLocalGather(op, adaptor, rewriter);
  } else {
    emitGatherInShared(op, adaptor, rewriter);
  }
  return success();
}

void GatherOpConversion::emitGatherInShared(
    GatherOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
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
}

void GatherOpConversion::emitWarpLocalGather(
    GatherOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
  MLIRContext *ctx = op.getContext();
  Location loc = op.getLoc();
  RankedTensorType srcType = op.getSrc().getType();
  RankedTensorType idxType = op.getIndices().getType();

  llvm::errs() << getLayoutStr(srcType, false) << "\n";
  llvm::errs() << getLayoutStr(idxType, false) << "\n";

  StringAttr kLane = str_attr("lane");
  StringAttr kRegister = str_attr("register");
  StringAttr kGatherDim = rewriter.getStringAttr("dim" + Twine(op.getAxis()));

  SmallVector<Value> srcValues =
      unpackLLElements(loc, adaptor.getSrc(), rewriter);
  SmallVector<Value> idxValues =
      unpackLLElements(loc, adaptor.getIndices(), rewriter);

  // For a warp-local gather, a couple things are true:
  // - Each warp owns 2^N columns of the source tensor along the gather axis
  //   and the same columns of the index tensor, which may be shorter or longer
  //   than those in the source tensor.
  // - Columns may be owned by multiple warps if the layout is oversubscribed.
  // - In a particular column, each thread owns at least one element of the
  //   source tensor and at least one element of the index tensor.

  // Organize the source and index values into columns.
  SmallVector<StringAttr> otherDims;
  for (unsigned dim = 0, rank = srcType.getRank(); dim < rank; ++dim) {
    if (dim != op.getAxis()) {
      otherDims.push_back(str_attr("dim" + Twine(dim)));
    }
  }

  LinearLayout srcLayout =
      *toLinearLayout(srcType.getShape(), srcType.getEncoding());
  LinearLayout idxLayout =
      *toLinearLayout(idxType.getShape(), idxType.getEncoding());

  LinearLayout srcColLayout = srcLayout.sublayout({kRegister}, otherDims);
  LinearLayout idxColLayout = idxLayout.sublayout({kRegister}, otherDims);
  LinearLayout srcThreadLayout = srcLayout.sublayout({kRegister}, kGatherDim);
  LinearLayout idxThreadLayout = idxLayout.sublayout({kRegister}, kGatherDim);

  // Sanity check the layouts.
  assert(srcColLayout.getInDimSize(kRegister) == srcValues.size());
  assert(idxColLayout.getInDimSize(kRegister) == idxValues.size());
  assert(srcThreadLayout.getInDimSize(kRegister) == srcValues.size());
  assert(idxThreadLayout.getInDimSize(kRegister) == idxValues.size());

  SmallVector<SmallVector<Value>> srcValuesCol, idxValuesCol;
  for (auto [i, srcVal] : llvm::enumerate(srcValues)) {
    SmallVector<std::pair<StringAttr, int32_t>> colIdx =
        srcColLayout.apply({{kRegister, i}});
  }

  SmallVector<Value> tmpResults(idxValues.size(), f32_val(0.0));
  rewriter.replaceOp(op, packLLElements(loc, getTypeConverter(), tmpResults,
                                        rewriter, op.getType()));
}

} // namespace

void triton::populateGatherOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                            RewritePatternSet &patterns,
                                            const TargetInfoBase &targetInfo,
                                            PatternBenefit benefit) {
  patterns.insert<GatherOpConversion>(typeConverter, targetInfo, benefit);
}
