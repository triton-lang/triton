#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

static void atomicAddOne(Value ptr, Location loc,
                         ConversionPatternRewriter &rewriter) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  LLVM::AtomicRMWOp::create(rewriter, loc, LLVM::AtomicBinOp::add, ptr,
                            b.i32_val(1), LLVM::AtomicOrdering::monotonic);
}

static SmallVector<Value>
computeWarpHistogram(Location loc, ConversionPatternRewriter &rewriter,
                     const SmallVector<Value> &srcValues,
                     const SmallVector<Value> &maskValues, int numBins,
                     int numThreadsPerWarp, Value threadPred,
                     const TargetInfoBase &targetInfo) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  SmallVector<Value> counts(numBins, b.i32_val(0));
  auto ballotTy = int_ty(numThreadsPerWarp);
  for (int i = 0; i < srcValues.size(); ++i) {
    Value pred = threadPred;
    if (!maskValues.empty())
      pred = maybeAnd(rewriter, loc, pred, maskValues[i]);
    for (int bin = 0; bin < numBins; ++bin) {
      Value binPred = b.icmp_eq(srcValues[i], b.i32_val(bin));
      binPred = maybeAnd(rewriter, loc, binPred, pred);
      Value ballot = targetInfo.ballot(rewriter, loc, ballotTy, binPred);
      Value count = LLVM::CtPopOp::create(rewriter, loc, ballotTy, ballot);
      if (numThreadsPerWarp > 32)
        count = b.trunc(i32_ty, count);
      counts[bin] = b.add(counts[bin], count);
    }
  }

  return counts;
}

static SmallVector<Value>
reduceWarpHistograms(HistogramOp op, ArrayRef<Value> counts,
                     ConversionPatternRewriter &rewriter,
                     const LLVMTypeConverter *typeConverter) {
  int numWarps = lookupNumWarps(op);
  if (numWarps == 1)
    return llvm::to_vector(counts);

  auto loc = op.getLoc();
  auto *ctx = op.getContext();
  int numBins = counts.size();
  int threadsPerWarp =
      TritonGPUDialect::getThreadsPerWarp(op->getParentOfType<ModuleOp>());
  auto kWarp = str_attr("dim0");
  auto kBin = str_attr("dim1");
  // Each warp contributes one count per bin, including zero from redundant
  // warps. Reducing dim0 combines those partial histograms.
  auto layout =
      LinearLayout::identity1D(numBins, str_attr("register"), kBin) *
      LinearLayout::zeros1D(threadsPerWarp, str_attr("lane"), kBin) *
      LinearLayout::identity1D(numWarps, str_attr("warp"), kWarp) *
      LinearLayout::zeros1D(lookupNumCTAs(op), str_attr("block"), kBin);
  auto type = RankedTensorType::get(
      {numWarps, numBins}, rewriter.getI32Type(),
      LinearEncodingAttr::get(ctx, layout.transposeOuts({kWarp, kBin})));
  Value packed =
      packUniqueTensorElements(loc, typeConverter, counts, rewriter, type);
  Value partials =
      UnrealizedConversionCastOp::create(rewriter, loc, type, packed)
          .getResult(0);
  auto reduce = ReduceOp::create(rewriter, loc, ValueRange{partials}, 0);
  {
    OpBuilder::InsertionGuard guard(rewriter);
    auto *body = rewriter.createBlock(
        &reduce.getCombineOp(), {},
        {rewriter.getI32Type(), rewriter.getI32Type()}, {loc, loc});
    Value sum = arith::AddIOp::create(rewriter, loc, body->getArgument(0),
                                      body->getArgument(1));
    ReduceReturnOp::create(rewriter, loc, ValueRange{sum});
  }
  // The reduction reuses the histogram scratch for its inter-warp transfer.
  reduce->setAttr("allocation.offset", op->getAttr("allocation.offset"));
  assert(ReduceOpHelper(reduce).getScratchSizeInBytes() <=
         defaultAllocationAnalysisScratchSizeFn(op));
  Value result = reduce.getResult().front();
  Value reduced =
      UnrealizedConversionCastOp::create(
          rewriter, loc, typeConverter->convertType(result.getType()), result)
          .getResult(0);
  return unpackUniqueTensorElements(loc, reduced, rewriter);
}

static SmallVector<Value>
distributeHistogram(Location loc, ConversionPatternRewriter &rewriter,
                    ArrayRef<Value> counts, ArrayRef<Value> indices) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  SmallVector<Value> histogramValues;
  for (Value index : indices) {
    Value count = counts.back();
    for (int bin = 0; bin + 1 < counts.size(); ++bin)
      count = b.select(b.icmp_eq(index, b.i32_val(bin)), counts[bin], count);
    histogramValues.push_back(count);
  }
  return histogramValues;
}

static void accumulateSharedHistogram(
    Location loc, ConversionPatternRewriter &rewriter, Value baseSharedMemPtr,
    ArrayRef<Value> srcValues, ArrayRef<Value> maskValues, int numBins,
    int numThreadPerWarp, Value threadId, Value threadPred, int numWarps,
    const TargetInfoBase &targetInfo) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  // Initialize the shared memory with zeros.
  int64_t numElementPerThread =
      ceil<int64_t>(numBins, numThreadPerWarp * numWarps);
  for (int i = 0; i < numElementPerThread; ++i) {
    Value offset =
        b.add(threadId, b.i32_val((i * numWarps * numThreadPerWarp)));
    offset = b.urem(offset, b.i32_val(numBins));
    Value sharedMemPtr =
        b.gep(baseSharedMemPtr.getType(), i32_ty, baseSharedMemPtr, offset);
    targetInfo.storeShared(rewriter, loc, sharedMemPtr, b.i32_val(0),
                           b.true_val());
  }
  b.barrier(triton::gpu::AddrSpace::Local);

  // Apply atomic add to update the histogram in shared memory.
  Value numBinsValue = b.i32_val(numBins);
  for (int i = 0; i < srcValues.size(); ++i) {
    Value pred = threadPred;
    if (!maskValues.empty())
      pred = maybeAnd(rewriter, loc, pred, maskValues[i]);
    Value updatePred = b.icmp_ult(srcValues[i], numBinsValue);
    updatePred = maybeAnd(rewriter, loc, updatePred, pred);

    auto [prevBlock, ifBlock, thenBlock] =
        createIfBlock(rewriter, loc, updatePred);
    (void)prevBlock;
    rewriter.setInsertionPointToStart(ifBlock);
    Value sharedMemPtr = b.gep(baseSharedMemPtr.getType(), i32_ty,
                               baseSharedMemPtr, srcValues[i]);
    atomicAddOne(sharedMemPtr, loc, rewriter);
    rewriter.setInsertionPointToStart(thenBlock);
  }
}

static SmallVector<Value>
loadSharedHistogram(Location loc, ConversionPatternRewriter &rewriter,
                    Value baseSharedMemPtr, ArrayRef<Value> indices,
                    int numCTAsToCombine, int ctaBroadcastMask,
                    HistogramOp sourceOp, const TargetInfoBase &targetInfo) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  if (numCTAsToCombine > 1)
    targetInfo.clusterBarrier(loc, rewriter, sourceOp);
  else
    b.barrier(triton::gpu::AddrSpace::Local);
  SmallVector<Value> histogramValues;
  for (Value index : indices) {
    Value sharedMemPtr =
        b.gep(baseSharedMemPtr.getType(), i32_ty, baseSharedMemPtr, index);
    Value val = b.i32_val(0);
    for (int cta = 0; cta < numCTAsToCombine; ++cta) {
      // Replicated CTAs do not accumulate inputs and contribute only zeros.
      if (cta & ctaBroadcastMask)
        continue;
      Value ctaId = numCTAsToCombine > 1 ? b.i32_val(cta) : Value();
      Value partial = targetInfo.loadDShared(rewriter, loc, sharedMemPtr, ctaId,
                                             i32_ty, b.true_val());
      val = b.add(val, partial);
    }
    histogramValues.push_back(val);
  }
  return histogramValues;
}

namespace {
struct HistogramOpConversion
    : public ConvertOpToLLVMPattern<triton::HistogramOp> {
public:
  using ConvertOpToLLVMPattern<triton::HistogramOp>::ConvertOpToLLVMPattern;

  explicit HistogramOpConversion(LLVMTypeConverter &typeConverter,
                                 const TargetInfoBase &targetInfo,
                                 PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo) {
  }

  LogicalResult
  matchAndRewrite(triton::HistogramOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto *ctx = op.getContext();
    Value input = adaptor.getSrc();
    auto typeConverter = getTypeConverter();
    SmallVector<Value> srcValues =
        unpackUniqueTensorElements(loc, input, rewriter);

    Value llMask = adaptor.getMask();
    SmallVector<Value> maskValues;
    if (llMask)
      maskValues = unpackUniqueTensorElements(loc, llMask, rewriter);

    int numBins = op.getType().getDimSize(0);
    auto mod = op->getParentOfType<ModuleOp>();
    int numThreadsPerWarp =
        triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
    assert((numThreadsPerWarp == 32 || numThreadsPerWarp == 64) &&
           "Only supports 32 or 64 threads per warp");
    int numWarps = triton::gpu::lookupNumWarps(op);
    Value threadId = getThreadId(rewriter, loc);
    auto srcType = op.getSrc().getType();
    auto freeVarMasks = getFreeVariableMasks(srcType);
    bool crossCTA = hasCrossCTAScratch(op);
    // Each CTA computes its own histogram.
    if (!crossCTA)
      freeVarMasks[str_attr("block")] = 0;
    Value threadPred =
        emitRedundantThreadPredicate(freeVarMasks, rewriter, loc, targetInfo);

    auto dstType = op.getType();
    auto dstLayout =
        toLinearLayout(dstType).removeZeroBasesAlongDim(str_attr("register"));
    auto indices = emitIndices(op.getLoc(), rewriter, targetInfo, dstLayout,
                               dstType, true);
    SmallVector<Value> innerDimIndices;
    for (int i = 0; i < indices.size(); ++i)
      innerDimIndices.push_back(indices[i][0]);
    SmallVector<Value> histogramValue;
    if (canUseWarpBallotHistogram(op)) {
      auto counts =
          computeWarpHistogram(loc, rewriter, srcValues, maskValues, numBins,
                               numThreadsPerWarp, threadPred, targetInfo);
      counts = reduceWarpHistograms(op, counts, rewriter, typeConverter);
      histogramValue =
          distributeHistogram(loc, rewriter, counts, innerDimIndices);
    } else {
      Value baseSharedMemPtr =
          LLVM::getSharedMemoryBase(loc, rewriter, targetInfo, op);
      accumulateSharedHistogram(loc, rewriter, baseSharedMemPtr, srcValues,
                                maskValues, numBins, numThreadsPerWarp,
                                threadId, threadPred, numWarps, targetInfo);
      histogramValue = loadSharedHistogram(
          loc, rewriter, baseSharedMemPtr, innerDimIndices,
          crossCTA ? lookupNumCTAs(op) : 1,
          freeVarMasks.lookup(str_attr("block")), op, targetInfo);
    }

    Value results = packUniqueTensorElements(loc, typeConverter, histogramValue,
                                             rewriter, op.getType());
    rewriter.replaceOp(op, results);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};
} // namespace

void mlir::triton::populateHistogramOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfoBase &targetInfo, PatternBenefit benefit) {
  patterns.add<HistogramOpConversion>(typeConverter, targetInfo, benefit);
}
