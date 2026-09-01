#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

namespace {
class GatherOpConversion : public ConvertOpToLLVMPattern<GatherOp> {
public:
  GatherOpConversion(ModuleOp module, LLVMTypeConverter &typeConverter,
                      const TargetInfoBase &targetInfo, PatternBenefit benefit)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo) {
    // Analyze tensor expressions before conversion rewrites their producers.
    module.walk([&](GatherOp op) { plans.try_emplace(op, op); });
  }

  LogicalResult
  matchAndRewrite(GatherOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;

private:
  // Codegen the gather by storing the source tensor into shared memory and then
  // gathering directly from shared memory.
  void emitGatherInShared(GatherOp op, OpAdaptor adaptor,
                          const GatherLoweringHelper &helper,
                          ConversionPatternRewriter &rewriter) const;
  // Codegen a warp-local gather by shuffling elements across the warp and
  // selecting from them.
  void emitWarpLocalGather(GatherOp op, OpAdaptor adaptor,
                           const LinearLayout &plan,
                           ConversionPatternRewriter &rewriter) const;

  const TargetInfoBase &targetInfo;
  DenseMap<Operation *, GatherLoweringHelper> plans;
};

LogicalResult
GatherOpConversion::matchAndRewrite(GatherOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter) const {
  const GatherLoweringHelper &helper = plans.find(op)->second;
  // Specialize the lowering based on the source layout. Given that the cost of
  // a warp shuffle is approximately half the cost of a roundtrip to shared
  // memory with zero bank conflicts, we will need a more precise heuristic to
  // choose between the two codegen paths and rely on the middle end to pick the
  // right layout.
  if (helper.isWarpLocal()) {
    emitWarpLocalGather(op, adaptor, *helper.getWarpLocalLayout(), rewriter);
  } else {
    emitGatherInShared(op, adaptor, helper, rewriter);
  }
  return success();
}

static Value convertIndexToI32(Location loc, Value index,
                              ConversionPatternRewriter &rewriter) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  unsigned idxWidth = index.getType().getIntOrFloatBitWidth();
  // The LL index computations are performed with 32 bit integers. If the
  // indices are something else, cast them to i32.
  if (idxWidth > 32) {
    index = b.trunc(i32_ty, index);
  } else if (idxWidth < 32) {
    // Negative indices don't make sense, so zero-extend.
    index = b.zext(i32_ty, index);
  }
  return index;
}

void GatherOpConversion::emitGatherInShared(
    GatherOp op, OpAdaptor adaptor, const GatherLoweringHelper &helper,
    ConversionPatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  auto *ctx = op.getContext();
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  RankedTensorType srcType = op.getSrc().getType();
  bool crossCTA = !helper.isCTALocal();
  ArrayRef<unsigned> sourceToIndex = helper.getSourceToIndex();
  bool selectedOnly = !sourceToIndex.empty();
  unsigned axis = op.getAxis();

  // Keep logical offsets, including the CTA coordinate. Each CTA populates
  // its own portion of this scratch; remote reads use the source's owning CTA.
  SmallVector<unsigned> scratchShape = convertType<unsigned>(
      selectedOnly ? op.getType().getShape() : srcType.getShape());

  // Grab the src values in this thread.
  SmallVector<Value> srcValues =
      unpackUniqueTensorElements(loc, adaptor.getSrc(), rewriter);
  SmallVector<Value> idxValues =
      unpackUniqueTensorElements(loc, adaptor.getIndices(), rewriter);

  // Emit the indices of the src values owned by this thread.
  auto srcLayout =
      toLinearLayout(srcType).removeZeroBasesAlongDim(str_attr("register"));
  SmallVector<SmallVector<Value>> srcIndices = emitIndices(
      loc, rewriter, targetInfo, srcLayout, srcType, /*withCTAOffset=*/true);
  auto freeVarMasks = srcLayout.getFreeVariableMasks();
  // Even replicated CTAs must populate their own scratch. Within each CTA,
  // choose one writer per logical element, including dependent register bases.
  freeVarMasks[str_attr("block")] = 0;
  Value writePredicate =
      emitRedundantThreadPredicate(freeVarMasks, rewriter, loc, targetInfo);
  if (!writePredicate)
    writePredicate = b.true_val();

  // Store the src values owned by the thread into their respective location in
  // the scratch memory.
  assert(srcValues.size() == srcIndices.size());

  // Get the base pointer to the scratch memory.
  Value smemBase = LLVM::getSharedMemoryBase(loc, rewriter, targetInfo, op);

  // For each src element owned by the thread, index into the scratch memory and
  // then store it.
  Type elemType = getTypeConverter()->convertType(srcType.getElementType());
  for (auto [reg, value, indices] : llvm::enumerate(srcValues, srcIndices)) {
    if (reg & freeVarMasks.lookup(str_attr("register")))
      continue;
    Value predicate = writePredicate;
    if (selectedOnly) {
      Value selector =
          convertIndexToI32(loc, idxValues[sourceToIndex[reg]], rewriter);
      predicate = b.and_(predicate, b.icmp_eq(indices[axis], selector));
      indices[axis] = b.i32_val(0);
    }
    // Convert the index at each dim into a single offset given the shape of the
    // tensor.
    Value offset = LLVM::linearize(rewriter, loc, indices, scratchShape);
    // Emit the offset into the shared memory and then store the value.
    Value ptr = b.gep(smemBase.getType(), elemType, smemBase, offset);
    targetInfo.storeShared(rewriter, loc, ptr, value, predicate);
  }

  if (crossCTA)
    targetInfo.clusterBarrier(loc, rewriter, op);
  else
    b.barrier(triton::gpu::AddrSpace::Local);

  // Apply the layout of the destination tensor to obtain the indices of the
  // column to gather along, then for each column, replace the index along the
  // gather axis with the appropriate index value.
  //
  // I = LL(pid)
  // idx = indices[I]
  // I_gather = [I[d] if d != axis else idx for d in range(len(I))]
  // out[I] = src[I_gather]
  RankedTensorType dstType = op.getType();
  auto dstLayout =
      toLinearLayout(dstType).removeZeroBasesAlongDim(str_attr("register"));
  SmallVector<SmallVector<Value>> dstIndices = emitIndices(
      loc, rewriter, targetInfo, dstLayout, dstType, /*withCTAOffset=*/true);

  std::optional<LinearLayout> sourceCTA;
  if (crossCTA)
    sourceCTA = srcLayout.pseudoinvert().sublayout(
        llvm::to_vector(srcLayout.getOutDimNames()), {str_attr("block")});
  SmallVector<Value> results(dstIndices.size());
  for (auto [i, idx, indices] : llvm::enumerate(idxValues, dstIndices)) {
    auto gatherIndices = indices;
    gatherIndices[axis] = convertIndexToI32(loc, idx, rewriter);
    Value ctaId;
    if (sourceCTA) {
      SmallVector<std::pair<StringAttr, Value>> coordinates;
      for (auto [dim, value] : llvm::zip(srcLayout.getOutDimNames(), gatherIndices))
        coordinates.emplace_back(dim, value);
      ctaId = applyLinearLayout(loc, rewriter, *sourceCTA, coordinates)
                  .front()
                  .second;
    }
    Value offset = LLVM::linearize(rewriter, loc,
                                   selectedOnly ? indices : gatherIndices,
                                   scratchShape);
    Value ptr = b.gep(smemBase.getType(), elemType, smemBase, offset);
    results[i] =
        targetInfo.loadDShared(rewriter, loc, ptr, ctaId, elemType, b.true_val());
  }
  // A remote CTA must finish reading before any producer reuses its scratch.
  if (crossCTA)
    targetInfo.clusterBarrier(loc, rewriter, op);

  Value packed = packUniqueTensorElements(loc, getTypeConverter(), results,
                                         rewriter, dstType);
  rewriter.replaceOp(op, packed);
}

// The analysis fixes the source warp (and, for thread-local gathers, lane)
// to the receiver's. Only the unresolved register and lane coordinates remain.
void GatherOpConversion::emitWarpLocalGather(
    GatherOp op, OpAdaptor adaptor, const LinearLayout &plan,
    ConversionPatternRewriter &rewriter) const {
  MLIRContext *ctx = op.getContext();
  Location loc = op.getLoc();
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  StringAttr kRegister = str_attr("register");
  StringAttr kLane = str_attr("lane");
  StringAttr kWarp = str_attr("warp");
  StringAttr kBlock = str_attr("block");
  StringAttr kIndex = str_attr("index");
  StringAttr kConstant = str_attr("constant");
  bool needsShuffle = plan.hasOutDim(kLane);
  auto outDims = llvm::to_vector(plan.getOutDimNames());

  SmallVector<Value> srcValues =
      unpackUniqueTensorElements(loc, adaptor.getSrc(), rewriter);
  SmallVector<Value> idxValues =
      unpackUniqueTensorElements(loc, adaptor.getIndices(), rewriter);
  auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
  Value blockId = plan.sublayoutIsZero(kBlock, outDims)
                      ? b.i32_val(0)
                      : targetInfo.getClusterCTAId(rewriter, loc);

  // Enumerate the image of the runtime inputs in the source-register space.
  // This prunes known index bits and dependent bases without assuming a
  // particular layout or visiting every register in the source tensor.
  SmallVector<int32_t> registerOffsets = {0};
  unsigned registerDim = plan.getOutDimIndex(kRegister);
  for (const auto &[dim, bases] : plan.getBases()) {
    if (dim == kRegister || dim == kConstant)
      continue;
    for (const auto &basis : bases) {
      int32_t offset = basis[registerDim];
      if (llvm::is_contained(registerOffsets, offset))
        continue;
      unsigned size = registerOffsets.size();
      for (unsigned i = 0; i < size; ++i)
        registerOffsets.push_back(registerOffsets[i] ^ offset);
    }
  }
  auto registerBase = plan.sublayout({kRegister, kConstant}, {kRegister});

  SmallVector<Value> results;
  for (auto [idxReg, idxValue] : llvm::enumerate(idxValues)) {
    SmallVector<std::pair<StringAttr, Value>> inputs =
        {{kRegister, b.i32_val(idxReg)}, {kLane, laneId}, {kWarp, warpId},
         {kBlock, blockId}, {kIndex, convertIndexToI32(loc, idxValue, rewriter)},
         {kConstant, b.i32_val(1)}};
    for (auto &[dim, value] : inputs)
      if (plan.sublayoutIsZero(dim, outDims))
        value = b.i32_val(0);
    auto coordinates = applyLinearLayout(loc, rewriter, plan, inputs);
    Value srcReg = coordinates[registerDim].second;
    Value srcLane = needsShuffle
                        ? coordinates[plan.getOutDimIndex(kLane)].second
                        : Value();
    int32_t base = registerBase.apply({{kRegister, idxReg}, {kConstant, 1}})
                       .front()
                       .second;
    Value result;
    for (int32_t offset : registerOffsets) {
      int32_t reg = base ^ offset;
      Value value = srcValues[reg];
      if (needsShuffle)
        value = targetInfo.shuffleIdx(rewriter, loc, value, srcLane);
      // Selection must happen at the receiver: each lane may request a
      // different register from the same source lane.
      result = result ? b.select(b.icmp_eq(srcReg, b.i32_val(reg)), value, result)
                      : value;
    }
    results.push_back(result);
  }

  rewriter.replaceOp(op,
                     packUniqueTensorElements(loc, getTypeConverter(), results,
                                              rewriter, op.getType()));
}

} // namespace

void triton::populateGatherOpToLLVMPatterns(ModuleOp module,
                                          LLVMTypeConverter &typeConverter,
                                          RewritePatternSet &patterns,
                                          const TargetInfoBase &targetInfo,
                                          PatternBenefit benefit) {
  patterns.insert<GatherOpConversion>(module, typeConverter, targetInfo, benefit);
}
