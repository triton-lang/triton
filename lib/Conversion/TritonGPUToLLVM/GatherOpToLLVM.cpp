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

  // Layout dimension names.
  StringAttr kBlock = str_attr("block");
  StringAttr kWarp = str_attr("warp");
  StringAttr kLane = str_attr("lane");
  StringAttr kRegister = str_attr("register");
  StringAttr kGatherDim = rewriter.getStringAttr("dim" + Twine(op.getAxis()));
  SmallVector<StringAttr> allDims, otherDims;
  for (unsigned dim = 0, rank = srcType.getRank(); dim < rank; ++dim) {
    allDims.push_back(str_attr("dim" + Twine(dim)));
    if (dim != op.getAxis()) {
      otherDims.push_back(allDims.back());
    }
  }

  // Compute the src and idx layouts.
  LinearLayout srcLayout =
      *toLinearLayout(srcType.getShape(), srcType.getEncoding());
  LinearLayout idxLayout =
      *toLinearLayout(idxType.getShape(), idxType.getEncoding());

  // Let `ll_src` be the source layout and `ll_idx` be the index layout.
  // Let `src_col` be a tuple of dimensions except the gather dimension,
  // representing a specific column in the source tensor. Likewise for
  // `idx_col`. Let `src_idx` be the index into gather dimension in the source
  // tensor.
  //
  // `(src_lane, src_reg) = ll_src^-1(src_col, src_idx)`, where `src_lane` is
  // the thread that contains the required element and `src_reg` is the register
  // within that thread.
  //
  // Because `ll_src(block=0, warp=0, lane=0)[otherDims] ==
  // idx_src(0, 0, 0)[otherDims]`, we know given any `idx_reg` (element in the
  // index tensor) the thread will need to read from the same column in the
  // source tensor.
  //
  // Thus, we can obtain
  //
  //   (src_lane, src_reg) = (ll_src^-1)(
  //       ll_idx(black, warp, lane, idx_reg)[otherDims],
  //       idxValues[idx_reg]
  //   )[{"lane", "register"}]
  //
  // And the mapping will be the correct for each thread.
  //
  // Given `src_reg \in [0, N)`, we just need to emit N index shuffles for each
  // `idx_reg` (the number of index shuffles is quadratic!) and `llvm.select`
  // using `src_reg` to get the right one.

  // Fully invert the source layout. We know it is invertible because
  // `isWarpLocal` checked this (subpermutation matrix, no broadcasting).
  LinearLayout invSrcLayout = srcLayout.invert();

  // Sanity check: the warp must be invariant to the index because otherwise the
  // gather would need to read across warps!
  assert(invSrcLayout.sublayoutIsZero(kGatherDim, {kBlock, kWarp}) &&
         "expected a warp-local gather");
  invSrcLayout = invSrcLayout.sublayout(allDims, {kLane, kRegister});

  LinearLayout idxColLayout =
      idxLayout.sublayout({kBlock, kWarp, kLane, kRegister}, otherDims);

  SmallVector<Value> srcValues =
      unpackLLElements(loc, adaptor.getSrc(), rewriter);
  SmallVector<Value> idxValues =
      unpackLLElements(loc, adaptor.getIndices(), rewriter);

  auto [blockId, warpId, laneId] =
      emitHardwareTuple(loc, rewriter, targetInfo, /*withCTAOffset=*/true,
                        srcLayout.getInDimSize(kLane));

  unsigned /*N=*/srcRegsPerThread = srcLayout.getInDimSize(kRegister);
  assert(srcRegsPerThread == srcValues.size());
  SmallVector<Value> results;
  for (auto [idxReg, idxVal] : llvm::enumerate(idxValues)) {
    SmallVector<std::pair<StringAttr, Value>> column =
        applyLinearLayout(loc, rewriter, idxColLayout,
                          {{kBlock, blockId},
                           {kWarp, warpId},
                           {kLane, laneId},
                           {kRegister, i32_val(idxReg)}});
    assert(column.size() == otherDims.size());

    column.emplace_back(kGatherDim, idxVal);
    SmallVector<std::pair<StringAttr, Value>> srcLaneAndReg =
        applyLinearLayout(loc, rewriter, invSrcLayout, column);

    auto [srcLaneName, srcLane] = srcLaneAndReg.back();
    auto [srcRegName, srcReg] = srcLaneAndReg.front();
    assert(srcLaneName == kLane && srcRegName == kRegister);

    assert(!srcValues.empty() && "can't gather from an empty tensor");
    Value result = undef(srcValues.front().getType());
    for (unsigned i = 0; i != srcRegsPerThread; ++i) {
      Value value = targetInfo.shuffleIdx(rewriter, loc, srcValues[i], srcLane);
      result = select(icmp_eq(i32_val(i), srcReg), value, result);
    }
    results.push_back(result);
  }

  rewriter.replaceOp(op, packLLElements(loc, getTypeConverter(), results,
                                        rewriter, op.getType()));
}

} // namespace

void triton::populateGatherOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                            RewritePatternSet &patterns,
                                            const TargetInfoBase &targetInfo,
                                            PatternBenefit benefit) {
  patterns.insert<GatherOpConversion>(typeConverter, targetInfo, benefit);
}
