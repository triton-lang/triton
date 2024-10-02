#include "ReduceScanCommon.h"
#include "mlir/Support/LLVM.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::LLVM::delinearize;
using ::mlir::LLVM::linearize;
using ::mlir::triton::gpu::getTotalElemsPerThread;

// apply combine region to acc and cur and accumulate it into acc
static SmallVector<Value> accumulate(ScanLoweringHelper &helper,
                                     ConversionPatternRewriter &rewriter,
                                     ValueRange acc, ValueRange cur,
                                     Value pred = {}) {
  auto loc = helper.getLoc();
  auto &combineOp = helper.getCombineOp();
  return applyCombineOp(loc, rewriter, combineOp, acc, cur, pred);
}

// Scan a contiguous elements within a thread and update `srcValues` in place.
static void
scanThreadContiguousElements(SmallVector<SmallVector<Value>> &srcValues,
                             ConversionPatternRewriter &rewriter,
                             ScanLoweringHelper &helper) {
  // Depending on layout contiguous elements along axis dim may not be
  // contiguous in srcValues. Keep track of what elements belong to the same
  // chunk of contiguous elements.
  unsigned scanElementsPerThreads = helper.getAxisNumElementsPerThread();
  unsigned numChunks = srcValues.size() / scanElementsPerThreads;
  unsigned stride = helper.getAxisElementStride();
  SmallVector<SmallVector<Value>> accs(numChunks);
  for (unsigned srcIndex = 0; srcIndex < srcValues.size(); srcIndex++) {
    // Change this into emitOffsetForLayout?
    unsigned accIndex = (srcIndex % stride) +
                        ((srcIndex / stride) / scanElementsPerThreads) * stride;

    accs[accIndex] =
        accumulate(helper, rewriter, accs[accIndex], srcValues[srcIndex]);
    srcValues[srcIndex] = accs[accIndex];
  }
}

// Apply a scan across threads of the warp for the last element of each
// contiguous group of elements.
static void warpScan(SmallVector<SmallVector<Value>> &srcValues,
                     ConversionPatternRewriter &rewriter,
                     const TargetInfoBase &targetInfo,
                     ScanLoweringHelper &helper, Value laneIdAxis) {
  Location loc = helper.getLoc();
  unsigned scanElementsPerThreads = helper.getAxisNumElementsPerThread();
  unsigned elementStride = helper.getAxisElementStride();
  unsigned threadStride = helper.getAxisThreadStride();
  unsigned scanDim = helper.getAxisNumThreadsPerWarpWithUniqueData();
  for (unsigned srcIndex = 0; srcIndex < srcValues.size(); srcIndex++) {
    unsigned elementIdx = (srcIndex / elementStride) % scanElementsPerThreads;
    // Only consider the last element of each contiguous chunk of elements.
    if (elementIdx != scanElementsPerThreads - 1)
      continue;
    // Reduce within warps.
    SmallVector<Value> acc = srcValues[srcIndex];
    for (unsigned i = 1; i <= scanDim / 2; i <<= 1) {
      SmallVector<Value> shfl(acc.size());
      for (unsigned j = 0; j < acc.size(); ++j) {
        shfl[j] = targetInfo.shuffleUp(rewriter, loc, acc[j], i * threadStride);
      }
      Value mask = icmp_sge(laneIdAxis, i32_val(i));
      SmallVector<Value> tempAcc =
          accumulate(helper, rewriter, shfl, acc, mask);
      for (unsigned j = 0; j < acc.size(); ++j) {
        acc[j] = select(mask, tempAcc[j], acc[j]);
      }
    }
    srcValues[srcIndex] = acc;
  }
}

// For each set of contiguous elements within a thread we store the partial
// reduction into shared memory. Each parallel scan and each warp will store its
// own partial reductions. The shared memory is organized as follow:
//          -----------------------------------------------------------------
// chunk 0: | acc[0] warp 0 | acc[1] warp 0 | acc[0] warp 1 | acc[1] warp 1 |
// chunk 1: | acc[0] warp 0 | acc[1] warp 0 | acc[0] warp 1 | acc[1] warp 1 |
static void storeWarpAccumulator(SmallVector<SmallVector<Value>> &srcValues,
                                 ConversionPatternRewriter &rewriter,
                                 ScanLoweringHelper &helper, Value laneId,
                                 Value warpId, SmallVector<Value> smemBases,
                                 SmallVector<Type> smemTypes,
                                 Value parallelLaneId,
                                 const TargetInfoBase &targetInfo) {
  Location loc = helper.getLoc();
  unsigned scanElementsPerThreads = helper.getAxisNumElementsPerThread();
  unsigned scanDim = helper.getAxisNumThreadsPerWarpWithUniqueData();
  unsigned numParallelLane = helper.getNonAxisNumThreadsPerCTA();
  unsigned axisNumWarps = helper.getAxisNumWarpsWithUniqueData();
  unsigned chunkId = 0;
  unsigned elementStride = helper.getAxisElementStride();

  for (unsigned srcIndex = 0; srcIndex < srcValues.size(); srcIndex++) {
    unsigned elementIdx = (srcIndex / elementStride) % scanElementsPerThreads;
    // Only consider the last element of each contiguous chunk of elements.
    if (elementIdx != scanElementsPerThreads - 1)
      continue;
    auto lastElement = srcValues[srcIndex];
    Value mask = icmp_eq(laneId, i32_val(scanDim - 1));
    Value index = add(parallelLaneId, mul(warpId, i32_val(numParallelLane)));
    index = add(index, i32_val(chunkId * numParallelLane * axisNumWarps));
    for (unsigned i = 0; i < lastElement.size(); ++i) {
      Value writePtr =
          gep(smemBases[i].getType(), smemTypes[i], smemBases[i], index);
      targetInfo.storeShared(rewriter, loc, writePtr, lastElement[i], mask);
    }
    chunkId++;
  }
}

// Read the partial reductions from shared memory from each chunk of contiguous
// elements for each warp and parallel scan. Then combine the partial reduction
// with the right elements. Within a given contiguous element chunk we update
// all the elements by accumulating the value from the last element of the
// reduced value from the previous lane.
static void AddPartialReduce(SmallVector<SmallVector<Value>> &srcValues,
                             ConversionPatternRewriter &rewriter,
                             const TargetInfoBase &targetInfo,
                             ScanLoweringHelper &helper,
                             SmallVector<Value> smemBases,
                             SmallVector<Type> smemTypes, Value warpId,
                             Value laneIdAxis, Value parallelLaneId) {
  Location loc = helper.getLoc();
  unsigned numParallelLane = helper.getNonAxisNumThreadsPerCTA();
  unsigned scanElementsPerThreads = helper.getAxisNumElementsPerThread();
  unsigned parallelElementsPerThread = helper.getNonAxisNumElementsPerThread();
  unsigned elementStride = helper.getAxisElementStride();
  unsigned threadStride = helper.getAxisThreadStride();
  unsigned axisNumWarps = helper.getAxisNumWarpsWithUniqueData();
  Value maskNotFirstWarp = icmp_ne(warpId, i32_val(0));
  Value maskNotFirstLane = icmp_ne(laneIdAxis, i32_val(0));
  Value maskNotFirstThread = or_(maskNotFirstWarp, maskNotFirstLane);
  struct Accumulator {
    SmallVector<Value> acc;
    SmallVector<Value> maskedAcc;
  };
  unsigned numScanBlocks = helper.getAxisNumBlocks();
  unsigned numParallelBlocks = helper.getNonAxisNumBlocks();
  assert(numScanBlocks * numParallelBlocks * parallelElementsPerThread *
             scanElementsPerThreads ==
         srcValues.size());
  SmallVector<Accumulator> accumulators(numParallelBlocks *
                                        parallelElementsPerThread);
  unsigned chunkId = 0;
  unsigned blockStride = helper.getAxisBlockStride();
  for (unsigned srcIndex = 0; srcIndex < srcValues.size(); srcIndex++) {
    unsigned elementIdx = (srcIndex / elementStride) % scanElementsPerThreads;
    // Only consider the last element of each contiguous chunk of elements.
    if (elementIdx != scanElementsPerThreads - 1)
      continue;
    // Accumulate the partial reduction from shared memory. Decide which
    // accumulator to combine based on whether the elements belong to the same
    // dimension along axis.
    unsigned blockId = chunkId / parallelElementsPerThread;
    unsigned parallelBlockId =
        blockId % blockStride +
        ((blockId / blockStride) / numScanBlocks) * blockStride;
    unsigned accumulatorIndex = chunkId % parallelElementsPerThread +
                                parallelBlockId * parallelElementsPerThread;
    Accumulator &accumulator = accumulators[accumulatorIndex];
    unsigned axisBlockId = (blockId / blockStride) % numScanBlocks;
    for (unsigned i = 0; i < axisNumWarps; ++i) {
      Value index = add(parallelLaneId, i32_val(numParallelLane *
                                                (i + chunkId * axisNumWarps)));
      SmallVector<Value> partialReduce(helper.getNumOperands());
      for (unsigned j = 0; j < helper.getNumOperands(); ++j) {
        auto elemTy = smemTypes[j];
        Value ptr = gep(smemBases[j].getType(), elemTy, smemBases[j], index);
        partialReduce[j] = load(elemTy, ptr);
      }

      if (accumulator.acc.size() == 0) {
        accumulator.acc = partialReduce;
        accumulator.maskedAcc = partialReduce;
        continue;
      }
      Value mask = icmp_sge(warpId, i32_val(i + 1));
      accumulator.acc =
          accumulate(helper, rewriter, accumulator.acc, partialReduce, mask);
      for (unsigned j = 0; j < helper.getNumOperands(); ++j) {
        accumulator.maskedAcc[j] =
            select(mask, accumulator.acc[j], accumulator.maskedAcc[j]);
      }
    }

    Value pred = axisBlockId == 0 ? maskNotFirstWarp : Value{};
    auto temp = accumulate(helper, rewriter, accumulator.maskedAcc,
                           srcValues[srcIndex], pred);
    if (axisBlockId == 0) {
      // For the first warp and first chunk we don't have anything to
      // accumulate.
      auto val = srcValues[srcIndex];
      for (unsigned i = 0; i < helper.getNumOperands(); ++i) {
        temp[i] = select(maskNotFirstWarp, temp[i], val[i]);
      }
    }
    srcValues[srcIndex] = temp;
    // Update the rest of the contiguous elements.
    SmallVector<Value> lastElement(helper.getNumOperands());
    for (unsigned i = 0; i < helper.getNumOperands(); ++i) {
      auto elem = targetInfo.shuffleUp(rewriter, loc, temp[i], threadStride);
      lastElement[i] = select(maskNotFirstLane, elem, accumulator.maskedAcc[i]);
    }
    for (unsigned i = 1; i < scanElementsPerThreads; ++i) {
      pred = axisBlockId == 0 ? maskNotFirstThread : Value{};
      auto laneValue = srcValues[srcIndex - i * elementStride];
      laneValue = accumulate(helper, rewriter, lastElement, laneValue, pred);
      if (axisBlockId == 0) {
        // For the first warp and first chunk we don't have anything to
        // accumulate.
        for (unsigned j = 0; j < helper.getNumOperands(); ++j) {
          laneValue[j] = select(maskNotFirstThread, laneValue[j],
                                srcValues[srcIndex - i * elementStride][j]);
        }
      }
      srcValues[srcIndex - i * elementStride] = laneValue;
    }
    // For the next chunk start back from the value containing the
    // accumulated value of all the warps.
    accumulator.maskedAcc = accumulator.acc;
    chunkId++;
  }
}

static void AddPartialReduceOneWarp(SmallVector<SmallVector<Value>> &srcValues,
                                    ConversionPatternRewriter &rewriter,
                                    const TargetInfoBase &targetInfo,
                                    ScanLoweringHelper &helper, Value warpId,
                                    Value laneIdAxis, Value laneIdLast) {
  Location loc = helper.getLoc();
  unsigned scanElementsPerThreads = helper.getAxisNumElementsPerThread();
  unsigned parallelElementsPerThread = helper.getNonAxisNumElementsPerThread();
  unsigned elementStride = helper.getAxisElementStride();
  unsigned threadStride = helper.getAxisThreadStride();
  unsigned axisNumWarps = helper.getAxisNumWarpsWithUniqueData();
  unsigned numParallelLane = helper.getNonAxisNumThreadsPerCTA();
  unsigned scanDim = helper.getAxisNumThreadsPerWarpWithUniqueData();
  Value maskFirstWarp = icmp_eq(warpId, i32_val(0));
  Value maskFirstLane = icmp_eq(laneIdAxis, i32_val(0));
  Value maskFirstThread = and_(maskFirstWarp, maskFirstLane);
  unsigned numScanBlocks = helper.getAxisNumBlocks();
  unsigned numParallelBlocks = helper.getNonAxisNumBlocks();
  assert(numScanBlocks * numParallelBlocks * parallelElementsPerThread *
             scanElementsPerThreads ==
         srcValues.size());
  SmallVector<SmallVector<Value>> accumulators(numParallelBlocks *
                                               parallelElementsPerThread);
  unsigned chunkId = 0;
  unsigned blockStride = helper.getAxisBlockStride();
  for (unsigned srcIndex = 0; srcIndex < srcValues.size(); srcIndex++) {
    unsigned elementIdx = (srcIndex / elementStride) % scanElementsPerThreads;
    // Only consider the last element of each contiguous chunk of elements.
    if (elementIdx != scanElementsPerThreads - 1)
      continue;
    unsigned blockId = chunkId / parallelElementsPerThread;
    unsigned parallelBlockId =
        blockId % blockStride +
        ((blockId / blockStride) / numScanBlocks) * blockStride;
    unsigned accumulatorIndex = chunkId % parallelElementsPerThread +
                                parallelBlockId * parallelElementsPerThread;
    auto &accumulator = accumulators[accumulatorIndex];
    unsigned axisBlockId = (blockId / blockStride) % numScanBlocks;
    if (axisBlockId == 0) // First chunk and first block
      accumulator = srcValues[srcIndex];
    else
      srcValues[srcIndex] =
          accumulate(helper, rewriter, accumulator, srcValues[srcIndex]);
    // Update the rest of the contiguous elements.
    auto lastElement = srcValues[srcIndex];
    if (scanDim > 1) {
      for (unsigned i = 0; i < helper.getNumOperands(); ++i) {
        lastElement[i] = targetInfo.shuffleUp(
            rewriter, loc, srcValues[srcIndex][i], threadStride);
        lastElement[i] = select(maskFirstLane, accumulator[i], lastElement[i]);
        if (numScanBlocks > 1)
          // Update accumulator with the value from the last lane.
          accumulator[i] = targetInfo.shuffleIdx(
              rewriter, loc, srcValues[srcIndex][i], laneIdLast);
      }
    } else if (numScanBlocks > 1) {
      accumulator = srcValues[srcIndex];
    }
    for (unsigned i = 1; i < scanElementsPerThreads; ++i) {
      auto laneValue = srcValues[srcIndex - i * elementStride];
      laneValue = accumulate(helper, rewriter, lastElement, laneValue);
      if (axisBlockId == 0) {
        for (unsigned j = 0; j < helper.getNumOperands(); ++j) {
          // For the first warp and first chunk we don't have anything to
          // accumulate.
          laneValue[j] =
              select(maskFirstThread,
                     srcValues[srcIndex - i * elementStride][j], laneValue[j]);
        }
      }
      srcValues[srcIndex - i * elementStride] = laneValue;
    }
    // For the next chunk start back from the value containing the
    // accumulated value of all the warps.
    chunkId++;
  }
}

namespace {
struct ScanOpConversion
    : public ConvertTritonGPUReduceScanToLLVMPattern<triton::ScanOp> {
public:
  using ConvertTritonGPUReduceScanToLLVMPattern<
      triton::ScanOp>::ConvertTritonGPUReduceScanToLLVMPattern;
  explicit ScanOpConversion(LLVMTypeConverter &typeConverter,
                            const TargetInfoBase &targetInfo,
                            PatternBenefit benefit = 1)
      : ConvertTritonGPUReduceScanToLLVMPattern<triton::ScanOp>(typeConverter,
                                                                benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::ScanOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (succeeded(emitFastScan(op, adaptor, rewriter, targetInfo)))
      return success();
    return failure();
  }

private:
  const TargetInfoBase &targetInfo;
  SmallVector<Value> getMultiDimLaneId(ConversionPatternRewriter &rewriter,
                                       ScanLoweringHelper &helper,
                                       Value laneId) const;
  SmallVector<Value> getMultiDimWarpId(ConversionPatternRewriter &rewriter,
                                       ScanLoweringHelper &helper,
                                       Value warpId) const;
  std::tuple<Value, Value, Value>
  getDelinearizedIds(ConversionPatternRewriter &rewriter,
                     ScanLoweringHelper &helper, Value laneId,
                     Value warpId) const;
  LogicalResult emitFastScan(triton::ScanOp op, triton::ScanOpAdaptor adaptor,
                             ConversionPatternRewriter &rewriter,
                             const TargetInfoBase &targetInfo) const;
};

SmallVector<Value>
ScanOpConversion::getMultiDimLaneId(ConversionPatternRewriter &rewriter,
                                    ScanLoweringHelper &helper,
                                    Value laneId) const {
  auto loc = helper.getLoc();
  unsigned axis = helper.getAxis();
  auto srcEncoding = helper.getEncoding();

  auto threadsPerWarp = triton::gpu::getThreadsPerWarp(srcEncoding);
  auto warpsPerCTA = triton::gpu::getWarpsPerCTA(srcEncoding);
  auto order = triton::gpu::getOrder(srcEncoding);
  return delinearize(rewriter, loc, laneId, threadsPerWarp, order);
}

SmallVector<Value>
ScanOpConversion::getMultiDimWarpId(ConversionPatternRewriter &rewriter,
                                    ScanLoweringHelper &helper,
                                    Value warpId) const {
  auto loc = helper.getLoc();
  unsigned axis = helper.getAxis();
  auto srcEncoding = helper.getEncoding();

  auto threadsPerWarp = triton::gpu::getThreadsPerWarp(srcEncoding);
  auto warpsPerCTA = triton::gpu::getWarpsPerCTA(srcEncoding);
  auto warpOrder = triton::gpu::getWarpOrder(srcEncoding);
  return delinearize(rewriter, loc, warpId, warpsPerCTA, warpOrder);
}

// Break up the threadId into lane and warp id along the scan dimension and
// compute a flat id for the parallel dimensions.
std::tuple<Value, Value, Value>
ScanOpConversion::getDelinearizedIds(ConversionPatternRewriter &rewriter,
                                     ScanLoweringHelper &helper, Value laneId,
                                     Value warpId) const {
  auto loc = helper.getLoc();
  unsigned axis = helper.getAxis();
  auto srcEncoding = helper.getEncoding();

  auto threadsPerWarp = triton::gpu::getThreadsPerWarp(srcEncoding);
  auto warpsPerCTA = triton::gpu::getWarpsPerCTA(srcEncoding);
  auto order = triton::gpu::getOrder(srcEncoding);
  auto warpOrder = triton::gpu::getWarpOrder(srcEncoding);
  SmallVector<Value> multiDimLaneId =
      delinearize(rewriter, loc, laneId, threadsPerWarp, order);
  SmallVector<Value> multiDimWarpId =
      delinearize(rewriter, loc, warpId, warpsPerCTA, warpOrder);

  Value laneIdAxis = multiDimLaneId[axis];
  Value warpIdAxis = multiDimWarpId[axis];

  multiDimLaneId[axis] = i32_val(0);
  threadsPerWarp[axis] = 1;
  Value laneIdParallel =
      linearize(rewriter, loc, multiDimLaneId, threadsPerWarp, order);
  multiDimWarpId[axis] = i32_val(0);
  warpsPerCTA[axis] = 1;
  Value warpIdParallel =
      linearize(rewriter, loc, multiDimWarpId, warpsPerCTA, warpOrder);
  Value flatIdParallel =
      add(laneIdParallel,
          mul(warpIdParallel, i32_val(helper.getNonAxisNumThreadsPerWarp())));
  return std::make_tuple(laneIdAxis, warpIdAxis, flatIdParallel);
}

SmallVector<SmallVector<Value>>
unpackInputs(Location loc, triton::ScanOp op, triton::ScanOpAdaptor adaptor,
             ConversionPatternRewriter &rewriter,
             const LLVMTypeConverter &converter) {
  auto types = op.getInputTypes();
  auto operands = adaptor.getOperands();
  unsigned srcElems = getTotalElemsPerThread(types[0]);
  SmallVector<SmallVector<Value>> srcValues(srcElems);
  for (unsigned i = 0; i < op.getNumOperands(); ++i) {
    auto values = unpackLLElements(loc, operands[i], rewriter);

    assert(values.size() == srcValues.size());
    for (unsigned j = 0; j < srcValues.size(); ++j) {
      srcValues[j].push_back(values[j]);
    }
  }
  return srcValues;
}

// Flip the srcValues. Both reverses the chunks and reverses the lanes.
// Lane reversal is done with a butterfly shuffle flip (divide and flip).
SmallVector<SmallVector<Value>>
flipSrcValues(Location loc, triton::ScanOp op,
              ConversionPatternRewriter &rewriter,
              const TargetInfoBase &targetInfo,
              SmallVector<SmallVector<Value>> srcValues, int iWarpSize) {
  SmallVector<SmallVector<Value>> values(srcValues.size());
  for (int i = 0; i < srcValues.size(); ++i) {
    int revIndex = srcValues.size() - i - 1;
    for (unsigned j = 0; j < op.getNumOperands(); ++j) {
      for (unsigned k = iWarpSize / 2; k >= 1; k = k / 2) {
        srcValues[revIndex][j] =
            targetInfo.shuffleXor(rewriter, loc, srcValues[revIndex][j], k);
      }
      values[i].push_back(srcValues[revIndex][j]);
    }
  }
  return values;
}

// Lowering using warp shuffle operations to do warp level scan.
LogicalResult
ScanOpConversion::emitFastScan(triton::ScanOp op, triton::ScanOpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter,
                               const TargetInfoBase &targetInfo) const {
  ScanLoweringHelper helper(op);
  auto loc = helper.getLoc();
  if (!helper.isSupported())
    return failure();

  Value threadId = getThreadId(rewriter, loc);
  auto mod = op->getParentOfType<ModuleOp>();
  unsigned iWarpSize = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
  Value warpSize = i32_val(iWarpSize);
  Value warpId = udiv(threadId, warpSize);
  Value laneId = urem(threadId, warpSize);

  auto [laneIdAxis, warpIdAxis, flatIdParallel] =
      getDelinearizedIds(rewriter, helper, laneId, warpId);
  auto axisNumWarps = helper.getAxisNumWarpsWithUniqueData();
  warpIdAxis = urem(warpIdAxis, i32_val(axisNumWarps));
  auto srcValues =
      unpackInputs(loc, op, adaptor, rewriter, *getTypeConverter());

  // For the reverse option we apply flip(scan(flip()) in
  // order to avoid having a separate code path in the reverse direction.
  // We do this by 1) reversing chunks, 2) reversing lanes, 3) reversing
  // warp ids and then undoing this below.
  // (Note: Tried pretty hard to get shflDownSync to work but I ended up
  // having to add a lot of the complex cross warp code (if rev switch
  // first/last etc). Reverse first seems more maintainable.)
  if (op.getReverse()) {
    warpIdAxis = sub(i32_val(axisNumWarps - 1), warpIdAxis);
    srcValues =
        flipSrcValues(loc, op, rewriter, targetInfo, srcValues, iWarpSize);
  }

  // Scan contiguous elements in a thread and update `srcValues`.
  scanThreadContiguousElements(srcValues, rewriter, helper);
  // Apply warp level scan to the last element of each chunk of contiguous
  // elements.
  warpScan(srcValues, rewriter, targetInfo, helper, laneIdAxis);

  if (axisNumWarps > 1) {
    // Slow path for the case where there are multiple warps with unique data on
    // the axis.
    auto elems = helper.getScratchSizeInElems();
    SmallVector<Value> smemBases =
        getSmemBases(op, elems, rewriter, targetInfo);
    SmallVector<Type> smemTypes(op.getNumOperands());
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      smemTypes[i] = getElementType(op, i);
    }

    // Store the partial reducing for each warp into shared memory.
    storeWarpAccumulator(srcValues, rewriter, helper, laneIdAxis, warpIdAxis,
                         smemBases, smemTypes, flatIdParallel, targetInfo);
    barrier();
    // Read back the partial reduction of each warp and accumulate them based on
    // warpId. Then update each chunk of contiguous elements by adding the
    // accumulated value from the previous lane.
    AddPartialReduce(srcValues, rewriter, targetInfo, helper, smemBases,
                     smemTypes, warpIdAxis, laneIdAxis, flatIdParallel);
  } else if (srcValues.size() > 1) {
    // Fast path for the case where there is only one warp with unique data on
    // the axis.
    unsigned scanDim = helper.getAxisNumThreadsPerWarpWithUniqueData();
    auto multiDimLaneId = getMultiDimLaneId(rewriter, helper, laneId);
    multiDimLaneId[helper.getAxis()] = i32_val(scanDim - 1);
    auto threadsPerWarp = triton::gpu::getThreadsPerWarp(helper.getEncoding());
    auto laneIdLast = linearize(rewriter, loc, multiDimLaneId, threadsPerWarp,
                                triton::gpu::getOrder(helper.getEncoding()));
    AddPartialReduceOneWarp(srcValues, rewriter, targetInfo, helper, warpIdAxis,
                            laneIdAxis, laneIdLast);
  } // else axisNumWarps == 1 and srcValues.size() == 1, nothing to do.

  auto transpose = [](const SmallVector<SmallVector<Value>> &v) {
    assert(v.size() > 0 && v[0].size() > 0);
    auto ret = SmallVector<SmallVector<Value>>(v[0].size(),
                                               SmallVector<Value>(v.size()));
    for (int i = 0; i < v.size(); ++i) {
      for (int j = 0; j < v[0].size(); ++j) {
        ret[j][i] = v[i][j];
      }
    }
    return ret;
  };

  SmallVector<Value> results(op.getNumOperands());
  if (op.getReverse()) {
    srcValues =
        flipSrcValues(loc, op, rewriter, targetInfo, srcValues, iWarpSize);
  }

  auto valuesTransposed = transpose(srcValues);
  for (unsigned i = 0; i < op.getNumOperands(); ++i) {
    auto resultTy = dyn_cast<RankedTensorType>(op.getResult()[i].getType());
    results[i] = packLLElements(loc, getTypeConverter(), valuesTransposed[i],
                                rewriter, resultTy);
  }
  rewriter.replaceOp(op, results);
  return success();
}
} // namespace

void mlir::triton::populateScanOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfoBase &targetInfo, PatternBenefit benefit) {
  patterns.add<ScanOpConversion>(typeConverter, targetInfo, benefit);
}
