#include "ReduceScanCommon.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Tools/LayoutUtils.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::LLVM::delinearize;
using ::mlir::LLVM::linearize;

namespace blocked_scan {
// apply combine region to acc and cur and accumulate it into acc
static SmallVector<Value> accumulate(BlockedScanLoweringHelper &helper,
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
                             BlockedScanLoweringHelper &helper) {
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
                     BlockedScanLoweringHelper &helper, Value laneIdAxis) {
  Location loc = helper.getLoc();
  auto b = TritonLLVMOpBuilder(loc, rewriter);
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
      Value mask = b.icmp_sge(laneIdAxis, b.i32_val(i));
      SmallVector<Value> tempAcc =
          accumulate(helper, rewriter, shfl, acc, mask);
      for (unsigned j = 0; j < acc.size(); ++j) {
        acc[j] = b.select(mask, tempAcc[j], acc[j]);
      }
    }
    srcValues[srcIndex] = std::move(acc);
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
                                 BlockedScanLoweringHelper &helper,
                                 Value laneId, Value warpId,
                                 SmallVector<Value> smemBases,
                                 SmallVector<Type> smemTypes,
                                 Value parallelLaneId, Value isRepresentative,
                                 const TargetInfoBase &targetInfo) {
  Location loc = helper.getLoc();
  auto b = TritonLLVMOpBuilder(loc, rewriter);
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
    Value mask = b.icmp_eq(laneId, b.i32_val(scanDim - 1));
    mask = b.and_(mask, isRepresentative);
    Value index =
        b.add(parallelLaneId, b.mul(warpId, b.i32_val(numParallelLane)));
    index = b.add(index, b.i32_val(chunkId * numParallelLane * axisNumWarps));
    for (unsigned i = 0; i < lastElement.size(); ++i) {
      Value writePtr =
          b.gep(smemBases[i].getType(), smemTypes[i], smemBases[i], index);
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
                             BlockedScanLoweringHelper &helper,
                             ArrayRef<Value> smemBases,
                             ArrayRef<Type> smemTypes, Value warpId,
                             Value laneIdAxis, Value parallelLaneId) {
  Location loc = helper.getLoc();
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  unsigned numParallelLane = helper.getNonAxisNumThreadsPerCTA();
  unsigned scanElementsPerThreads = helper.getAxisNumElementsPerThread();
  unsigned parallelElementsPerThread = helper.getNonAxisNumElementsPerThread();
  unsigned elementStride = helper.getAxisElementStride();
  unsigned threadStride = helper.getAxisThreadStride();
  unsigned axisNumWarps = helper.getAxisNumWarpsWithUniqueData();
  Value maskNotFirstWarp = b.icmp_ne(warpId, b.i32_val(0));
  Value maskNotFirstLane = b.icmp_ne(laneIdAxis, b.i32_val(0));
  Value maskNotFirstThread = b.or_(maskNotFirstWarp, maskNotFirstLane);
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
      Value index =
          b.add(parallelLaneId,
                b.i32_val(numParallelLane * (i + chunkId * axisNumWarps)));
      SmallVector<Value> partialReduce(helper.getNumOperands());
      for (unsigned j = 0; j < helper.getNumOperands(); ++j) {
        auto elemTy = smemTypes[j];
        Value ptr = b.gep(smemBases[j].getType(), elemTy, smemBases[j], index);
        partialReduce[j] =
            targetInfo.loadShared(rewriter, loc, ptr, elemTy, b.true_val());
      }

      if (accumulator.acc.size() == 0) {
        accumulator.acc = partialReduce;
        accumulator.maskedAcc = partialReduce;
        continue;
      }
      Value mask = b.icmp_sge(warpId, b.i32_val(i + 1));
      accumulator.acc =
          accumulate(helper, rewriter, accumulator.acc, partialReduce);
      for (unsigned j = 0; j < helper.getNumOperands(); ++j) {
        accumulator.maskedAcc[j] =
            b.select(mask, accumulator.acc[j], accumulator.maskedAcc[j]);
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
        temp[i] = b.select(maskNotFirstWarp, temp[i], val[i]);
      }
    }
    srcValues[srcIndex] = temp;
    // Update the rest of the contiguous elements.
    SmallVector<Value> lastElement(helper.getNumOperands());
    for (unsigned i = 0; i < helper.getNumOperands(); ++i) {
      auto elem = targetInfo.shuffleUp(rewriter, loc, temp[i], threadStride);
      lastElement[i] =
          b.select(maskNotFirstLane, elem, accumulator.maskedAcc[i]);
    }
    for (unsigned i = 1; i < scanElementsPerThreads; ++i) {
      pred = axisBlockId == 0 ? maskNotFirstThread : Value{};
      auto laneValue = srcValues[srcIndex - i * elementStride];
      laneValue = accumulate(helper, rewriter, lastElement, laneValue, pred);
      if (axisBlockId == 0) {
        // For the first warp and first chunk we don't have anything to
        // accumulate.
        for (unsigned j = 0; j < helper.getNumOperands(); ++j) {
          laneValue[j] = b.select(maskNotFirstThread, laneValue[j],
                                  srcValues[srcIndex - i * elementStride][j]);
        }
      }
      srcValues[srcIndex - i * elementStride] = std::move(laneValue);
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
                                    BlockedScanLoweringHelper &helper,
                                    Value warpId, Value laneIdAxis,
                                    Value laneIdLast) {
  Location loc = helper.getLoc();
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  unsigned scanElementsPerThreads = helper.getAxisNumElementsPerThread();
  unsigned parallelElementsPerThread = helper.getNonAxisNumElementsPerThread();
  unsigned elementStride = helper.getAxisElementStride();
  unsigned threadStride = helper.getAxisThreadStride();
  unsigned scanDim = helper.getAxisNumThreadsPerWarpWithUniqueData();
  Value maskFirstWarp = b.icmp_eq(warpId, b.i32_val(0));
  Value maskFirstLane = b.icmp_eq(laneIdAxis, b.i32_val(0));
  Value maskFirstThread = b.and_(maskFirstWarp, maskFirstLane);
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
        lastElement[i] =
            b.select(maskFirstLane, accumulator[i], lastElement[i]);
        if (numScanBlocks > 1)
          // Update accumulator with the value from the last lane.
          accumulator[i] = targetInfo.shuffleIdx(
              rewriter, loc, srcValues[srcIndex][i], laneIdLast);
      }
    } else if (numScanBlocks > 1) {
      // The rest of the chunk needs the total of the previous blocks, not the
      // value that already includes this chunk.
      lastElement = accumulator;
      accumulator = srcValues[srcIndex];
    }
    for (unsigned i = 1; i < scanElementsPerThreads; ++i) {
      auto laneValue = srcValues[srcIndex - i * elementStride];
      // The first thread has no carry; do not execute a side-effectful
      // combiner on its undefined prefix.
      Value pred = axisBlockId == 0
                       ? Value(b.xor_(maskFirstThread, b.true_val()))
                       : Value{};
      laneValue = accumulate(helper, rewriter, lastElement, laneValue, pred);
      if (axisBlockId == 0) {
        for (unsigned j = 0; j < helper.getNumOperands(); ++j) {
          // For the first warp and first chunk we don't have anything to
          // accumulate.
          laneValue[j] = b.select(maskFirstThread,
                                  srcValues[srcIndex - i * elementStride][j],
                                  laneValue[j]);
        }
      }
      srcValues[srcIndex - i * elementStride] = std::move(laneValue);
    }
    // For the next chunk start back from the value containing the
    // accumulated value of all the warps.
    chunkId++;
  }
}

namespace {
struct ScanOpConversion : public ConvertOpToLLVMPattern<triton::ScanOp> {
public:
  using ConvertOpToLLVMPattern<triton::ScanOp>::ConvertOpToLLVMPattern;
  explicit ScanOpConversion(LLVMTypeConverter &typeConverter,
                            const TargetInfoBase &targetInfo,
                            PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<triton::ScanOp>(typeConverter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::ScanOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!useBlockedScan(op))
      return failure();
    if (succeeded(emitFastScan(op, adaptor, rewriter, targetInfo)))
      return success();
    return failure();
  }

  // Return the pointee type of the shared memory pointer for operand i.
  Type getElementType(triton::ScanOp op, int i) const {
    auto ty = op.getInputTypes()[i].getElementType();
    return getTypeConverter()->convertType(ty);
  }

  // Helper to compute the smem bases in both reductions and scans
  SmallVector<Value> getSmemBases(triton::ScanOp op, unsigned elems,
                                  ConversionPatternRewriter &rewriter,
                                  const TargetInfoBase &targetInfo) const {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    // indices will store the index of the op operands in descending order
    // of their bitwidths
    std::vector<unsigned> indices(op.getNumOperands());
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(), [&](unsigned i, unsigned j) {
      return op.getElementTypes()[i].getIntOrFloatBitWidth() >
             op.getElementTypes()[j].getIntOrFloatBitWidth();
    });
    // Assign base index to each operand in their order in indices
    std::map<unsigned, Value> indexToBase;
    auto basePtr =
        LLVM::getSharedMemoryBase(loc, rewriter, targetInfo, op.getOperation());
    indexToBase[indices[0]] = basePtr;
    for (unsigned i = 1; i < op.getNumOperands(); ++i) {
      indexToBase[indices[i]] =
          b.gep(basePtr.getType(), getElementType(op, indices[i - 1]),
                indexToBase[indices[i - 1]], b.i32_val(elems));
    }
    // smemBases[k] is the base pointer for the k-th operand
    SmallVector<Value> smemBases(op.getNumOperands());
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      smemBases[i] = indexToBase[i];
    }
    return smemBases;
  }

private:
  const TargetInfoBase &targetInfo;
  std::tuple<SmallVector<Value>, Value>
  getMultiDimLaneId(ConversionPatternRewriter &rewriter,
                    BlockedScanLoweringHelper &helper, Value laneId) const;
  std::tuple<SmallVector<Value>, Value>
  getMultiDimWarpId(ConversionPatternRewriter &rewriter,
                    BlockedScanLoweringHelper &helper, Value warpId) const;
  std::tuple<Value, Value, Value, Value>
  getDelinearizedIds(ConversionPatternRewriter &rewriter,
                     BlockedScanLoweringHelper &helper, Value laneId,
                     Value warpId) const;
  LogicalResult emitFastScan(triton::ScanOp op, triton::ScanOpAdaptor adaptor,
                             ConversionPatternRewriter &rewriter,
                             const TargetInfoBase &targetInfo) const;
};

std::tuple<SmallVector<Value>, Value>
ScanOpConversion::getMultiDimLaneId(ConversionPatternRewriter &rewriter,
                                    BlockedScanLoweringHelper &helper,
                                    Value laneId) const {
  auto loc = helper.getLoc();
  auto srcEncoding = helper.getEncoding();
  auto kWarp = rewriter.getStringAttr("lane");
  return delinearize(rewriter, loc, srcEncoding, helper.getShape(), kWarp,
                     laneId);
}

std::tuple<SmallVector<Value>, Value>
ScanOpConversion::getMultiDimWarpId(ConversionPatternRewriter &rewriter,
                                    BlockedScanLoweringHelper &helper,
                                    Value warpId) const {
  auto loc = helper.getLoc();
  auto srcEncoding = helper.getEncoding();
  auto kWarp = rewriter.getStringAttr("warp");
  return delinearize(rewriter, loc, srcEncoding, helper.getShape(), kWarp,
                     warpId);
}

// Break up the threadId into lane and warp id along the scan dimension and
// compute a flat id for the parallel dimensions.
std::tuple<Value, Value, Value, Value>
ScanOpConversion::getDelinearizedIds(ConversionPatternRewriter &rewriter,
                                     BlockedScanLoweringHelper &helper,
                                     Value laneId, Value warpId) const {
  auto loc = helper.getLoc();
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  unsigned axis = helper.getAxis();
  auto srcEncoding = helper.getEncoding();

  auto threadsPerWarp = srcEncoding.getThreadsPerWarp();
  auto warpsPerCTA = srcEncoding.getWarpsPerCTA();
  auto [multiDimLaneId, isRepresentativeLane] =
      getMultiDimLaneId(rewriter, helper, laneId);
  auto [multiDimWarpId, isRepresentativeWarp] =
      getMultiDimWarpId(rewriter, helper, warpId);

  Value laneIdAxis = multiDimLaneId[axis];
  Value warpIdAxis = multiDimWarpId[axis];

  multiDimLaneId[axis] = b.i32_val(0);
  threadsPerWarp[axis] = 1;
  Value laneIdParallel = linearize(rewriter, loc, multiDimLaneId,
                                   threadsPerWarp, helper.getOrder());
  multiDimWarpId[axis] = b.i32_val(0);
  warpsPerCTA[axis] = 1;
  Value warpIdParallel =
      linearize(rewriter, loc, multiDimWarpId, warpsPerCTA, helper.getOrder());
  Value flatIdParallel = b.add(
      laneIdParallel,
      b.mul(warpIdParallel, b.i32_val(helper.getNonAxisNumThreadsPerWarp())));
  auto isRepresentative = b.and_(isRepresentativeLane, isRepresentativeWarp);
  return std::make_tuple(laneIdAxis, warpIdAxis, flatIdParallel,
                         isRepresentative);
}

SmallVector<SmallVector<Value>>
unpackInputs(Location loc, triton::ScanOp op, triton::ScanOpAdaptor adaptor,
             ConversionPatternRewriter &rewriter, unsigned nElems) {
  auto operands = adaptor.getOperands();
  SmallVector<SmallVector<Value>> srcValues(nElems);
  for (unsigned i = 0; i < op.getNumOperands(); ++i) {
    auto values = unpackUniqueTensorElements(loc, operands[i], rewriter);

    assert(values.size() == srcValues.size());
    for (unsigned j = 0; j < srcValues.size(); ++j) {
      srcValues[j].push_back(values[j]);
    }
  }
  return srcValues;
}

// Flip the srcValues. Both reverses the chunks and reverses the lanes.
// Lane reversal is done with a single butterfly shuffle: for power-of-two
// warp sizes, `lane ^ (iWarpSize - 1)` equals `(iWarpSize - 1) - lane`.
SmallVector<SmallVector<Value>>
flipSrcValues(Location loc, triton::ScanOp op,
              ConversionPatternRewriter &rewriter,
              const TargetInfoBase &targetInfo,
              SmallVector<SmallVector<Value>> srcValues, int iWarpSize) {
  SmallVector<SmallVector<Value>> values(srcValues.size());
  for (int i = 0; i < srcValues.size(); ++i) {
    int revIndex = srcValues.size() - i - 1;
    for (unsigned j = 0; j < op.getNumOperands(); ++j) {
      srcValues[revIndex][j] = targetInfo.shuffleXor(
          rewriter, loc, srcValues[revIndex][j], iWarpSize - 1);
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
  BlockedScanLoweringHelper helper(op);
  auto loc = helper.getLoc();
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  if (!helper.isSupported())
    return op.emitError("TODO: unsupported scan layout");

  Value threadId = getThreadId(rewriter, loc);
  auto mod = op->getParentOfType<ModuleOp>();
  unsigned iWarpSize = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
  Value warpSize = b.i32_val(iWarpSize);
  Value warpId = b.udiv(threadId, warpSize);
  Value laneId = b.urem(threadId, warpSize);

  auto [laneIdAxis, warpIdAxis, flatIdParallel, isRepresentative] =
      getDelinearizedIds(rewriter, helper, laneId, warpId);
  auto axisNumWarps = helper.getAxisNumWarpsWithUniqueData();
  unsigned nElems = triton::gpu::getUniqueElemsPerThread(
      cast<RankedTensorType>(op.getOperands()[0].getType()));
  auto srcValues = unpackInputs(loc, op, adaptor, rewriter, nElems);

  // For the reverse option we apply flip(scan(flip()) in
  // order to avoid having a separate code path in the reverse direction.
  // We do this by 1) reversing chunks, 2) reversing lanes, 3) reversing
  // warp ids and then undoing this below.
  // (Note: Tried pretty hard to get shflDownSync to work but I ended up
  // having to add a lot of the complex cross warp code (if rev switch
  // first/last etc). Reverse first seems more maintainable.)
  if (op.getReverse()) {
    warpIdAxis = b.sub(b.i32_val(axisNumWarps - 1), warpIdAxis);
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
                         smemBases, smemTypes, flatIdParallel, isRepresentative,
                         targetInfo);
    b.barrier(triton::gpu::AddrSpace::Local);
    // Read back the partial reduction of each warp and accumulate them based on
    // warpId. Then update each chunk of contiguous elements by adding the
    // accumulated value from the previous lane.
    AddPartialReduce(srcValues, rewriter, targetInfo, helper, smemBases,
                     smemTypes, warpIdAxis, laneIdAxis, flatIdParallel);
  } else if (srcValues.size() > 1) {
    // Fast path for the case where there is only one warp with unique data on
    // the axis.
    unsigned scanDim = helper.getAxisNumThreadsPerWarpWithUniqueData();
    auto multiDimLaneId =
        std::get<0>(getMultiDimLaneId(rewriter, helper, laneId));
    multiDimLaneId[helper.getAxis()] = b.i32_val(scanDim - 1);
    auto linearEncoding = helper.getEncoding();
    auto kLane = StringAttr::get(rewriter.getContext(), "lane");
    Value laneIdLast =
        linearize(rewriter, loc, multiDimLaneId, linearEncoding, kLane);
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
    results[i] =
        packUniqueTensorElements(loc, getTypeConverter(), valuesTransposed[i],
                                 rewriter, op.getResult()[i].getType());
  }
  rewriter.replaceOp(op, results);
  return success();
}
} // namespace

} // namespace blocked_scan

namespace {
struct ScanOpConversion : public ConvertOpToLLVMPattern<triton::ScanOp> {
  // Values are indexed by register, then by combiner operand.
  using ScanValues = SmallVector<SmallVector<Value>>;

  struct ScanCarry {
    SmallVector<Value> values;
    Value pred;
  };

  ScanOpConversion(LLVMTypeConverter &typeConverter,
                   const TargetInfoBase &targetInfo, PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::ScanOp>(typeConverter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::ScanOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ScanLoweringHelper helper(op);
    if (!helper.isSupported())
      return op.emitError("scan axis distributed across CTAs is not supported");

    auto loc = op.getLoc();
    ScanValues values;
    for (Value operand : adaptor.getOperands()) {
      auto unpacked = unpackUniqueTensorElements(loc, operand, rewriter);
      values.resize(unpacked.size());
      for (unsigned r = 0; r < unpacked.size(); ++r)
        values[r].push_back(unpacked[r]);
    }

    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Value threadId = getThreadId(rewriter, loc);
    unsigned warpSize = triton::gpu::TritonGPUDialect::getThreadsPerWarp(
        op->getParentOfType<ModuleOp>());
    Value laneId = b.urem(threadId, b.i32_val(warpSize));
    Value warpId = b.udiv(threadId, b.i32_val(warpSize));

    // Keep native thread prefixes in place and communicate only segment totals.
    permuteRegisters(values, helper.getRegisterOrder());
    if (!scanWarpChunks(op, helper, values, laneId, warpId, rewriter)) {
      scanWithinThreads(op, values, helper.getThreadLocalSegmentSize(),
                        rewriter);
      ScanValues intraWarpTotals;
      SmallVector<ScanCarry> interWarpCarries;
      if (helper.getIntraWarpLayout())
        intraWarpTotals = scanWithinWarps(op, helper, values, laneId, rewriter);
      if (helper.getInterWarpLayout())
        interWarpCarries = scanAcrossWarps(
            op, helper, intraWarpTotals.empty() ? values : intraWarpTotals,
            laneId, warpId, rewriter);
      applyScanCarries(op, helper, values, intraWarpTotals, interWarpCarries,
                       laneId, warpId, rewriter);
    }
    permuteRegisters(values, helper.getRegisterOrder().inverse());

    SmallVector<Value> results;
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      SmallVector<Value> unpacked;
      for (const auto &row : values)
        unpacked.push_back(row[i]);
      results.push_back(packUniqueTensorElements(loc, getTypeConverter(),
                                                 unpacked, rewriter,
                                                 op.getResult()[i].getType()));
    }
    rewriter.replaceOp(op, results);
    return success();
  }

private:
  static unsigned getSegmentMask(const LinearLayout &layout, StringAttr dim,
                                 unsigned axis, unsigned segmentSize) {
    unsigned mask = 0;
    for (auto [bit, basis] : llvm::enumerate(layout.getBases().lookup(dim)))
      if (basis[axis] && basis[axis] < segmentSize)
        mask |= 1u << bit;
    return mask;
  }

  static Value getTerminalLane(TritonLLVMOpBuilder &b, Value laneId,
                               unsigned laneMask, bool reverse) {
    return reverse ? Value(b.and_(laneId, b.i32_val(~laneMask)))
                   : Value(b.or_(laneId, b.i32_val(laneMask)));
  }

  static ScanValues transposeValues(const ScanValues &values) {
    ScanValues result(values.front().size());
    for (const auto &row : values)
      for (auto [i, value] : llvm::enumerate(row))
        result[i].push_back(value);
    return result;
  }

  static ScanValues extractSegmentTotals(const ScanValues &values,
                                         unsigned numRegs, bool reverse) {
    ScanValues totals;
    for (unsigned base = 0; base < values.size(); base += numRegs)
      totals.push_back(values[base + (reverse ? 0 : numRegs - 1)]);
    return totals;
  }

  SmallVector<Value> shuffleValues(Location loc, SmallVector<Value> values,
                                   Value lane,
                                   ConversionPatternRewriter &rewriter) const {
    for (Value &value : values)
      value = targetInfo.shuffleIdx(rewriter, loc, value, lane);
    return values;
  }

  // Shift within a segment, preserving bits that select other segments.
  static Value shiftWithinSegment(TritonLLVMOpBuilder &b, Value index,
                                  int distance, unsigned segmentSize,
                                  unsigned axisSize) {
    unsigned mask = segmentSize - 1;
    Value shifted = b.and_(b.add(index, b.i32_val(distance)), b.i32_val(mask));
    if (segmentSize == axisSize)
      return shifted;
    return b.or_(b.and_(index, b.i32_val(~mask)), shifted);
  }

  // Reindex SSA values without emitting data movement.
  void permuteRegisters(ScanValues &values, const ColumnAction &order) const {
    if (order.isIdentity())
      return;
    auto operands = transposeValues(values);
    for (auto &operand : operands)
      operand = order.apply(operand);
    values = transposeValues(operands);
  }

  // Convert segment totals using warp-local shuffles.
  void convertScanValues(triton::ScanOp op, ScanValues &values,
                         const LinearLayout &src, const LinearLayout &dst,
                         ConversionPatternRewriter &rewriter) const {
    if (src == dst)
      return;
    auto operands = convertLayoutValues(
        op.getLoc(), rewriter, op, src, dst, transposeValues(values),
        getTypeConverter(), targetInfo, /*forceWarpShuffle=*/true);
    values = transposeValues(operands);
  }

  // Scan contiguous register groups in traversal order.
  void scanWithinThreads(triton::ScanOp op, ScanValues &values,
                         unsigned numRegs,
                         ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    bool reverse = op.getReverse();
    for (unsigned base = 0; base < values.size(); base += numRegs)
      for (unsigned i = 1; i < numRegs; ++i) {
        unsigned r = base + (reverse ? numRegs - 1 - i : i);
        unsigned prev = reverse ? r + 1 : r - 1;
        values[r] = applyCombineOp(loc, rewriter, op.getCombineOp(),
                                   values[prev], values[r]);
      }
  }

  // Scan lane totals, reusing each round's shuffle for the exclusive carries.
  void scanLaneTotals(triton::ScanOp op, ScanValues &values,
                      const LinearLayout &layout, unsigned numRegs,
                      unsigned segmentSize, Value laneId,
                      ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto kLane = StringAttr::get(op.getContext(), "lane");
    unsigned axis = op.getAxis();
    bool reverse = op.getReverse();
    unsigned numLanes = segmentSize / numRegs;
    if (numLanes == 1)
      return;
    auto dims = llvm::to_vector(layout.getOutDimNames());
    auto laneLayout = layout.sublayout({kLane}, dims);
    auto inverseLaneLayout = layout.pseudoinvert().sublayout(dims, {kLane});

    // Shift logical coordinates and invert the layout to find the source lane.
    // Preserve coordinates belonging to independent scans.
    auto coords =
        applyLinearLayout(loc, rewriter, laneLayout, {{kLane, laneId}});
    Value index = coords[axis].second;
    unsigned axisSize = layout.getOutDimSize(dims[axis]);
    Value segmentIndex = index;
    if (segmentSize != axisSize)
      segmentIndex = b.and_(index, b.i32_val(segmentSize - 1));
    // Source lanes and predicates are shared by every register group.
    SmallVector<std::pair<Value, Value>> rounds;
    for (unsigned offset = 1; offset < numLanes; offset *= 2) {
      int distance = offset * numRegs;
      auto source = coords;
      source[axis].second = shiftWithinSegment(
          b, index, reverse ? distance : -distance, segmentSize, axisSize);
      Value lane = applyLinearLayout(loc, rewriter, inverseLaneLayout, source)
                       .front()
                       .second;
      Value pred =
          reverse ? b.icmp_ult(segmentIndex, b.i32_val(segmentSize - distance))
                  : b.icmp_uge(segmentIndex, b.i32_val(distance));
      rounds.emplace_back(lane, pred);
    }
    for (unsigned base = 0; base < values.size(); base += numRegs) {
      unsigned last = base + (reverse ? 0 : numRegs - 1);
      auto acc = values[last];
      SmallVector<Value> prefix;
      for (auto [lane, pred] : rounds) {
        auto incoming = shuffleValues(loc, acc, lane, rewriter);
        if (numRegs > 1) {
          // Initialize the exclusive carry from the first shuffle, then prepend
          // earlier groups. The boundary lane never consumes its carry.
          if (prefix.empty())
            prefix = incoming;
          else
            prefix = combineWithPrefix(op, incoming, prefix, rewriter, pred);
        }
        acc = combineWithPrefix(op, incoming, acc, rewriter, pred);
      }
      values[last] = acc;
      if (numRegs == 1)
        continue;
      // Skip the boundary lane without assuming an identity value.
      Value pred = rounds.front().second;
      for (unsigned r = base; r < base + numRegs; ++r)
        if (r != last)
          values[r] = combineWithPrefix(op, prefix, values[r], rewriter, pred);
    }
  }

  // Finish each warp-local chunk before starting the next. Only its terminal
  // prefix is carried between chunks; independent scans keep separate carries.
  bool scanWarpChunks(triton::ScanOp op, const ScanLoweringHelper &helper,
                      ScanValues &values, Value laneId, Value warpId,
                      ConversionPatternRewriter &rewriter) const {
    if (!helper.canStreamWarpScan())
      return false;
    auto *ctx = op.getContext();
    auto kReg = StringAttr::get(ctx, "register");
    auto kLane = StringAttr::get(ctx, "lane");
    auto axis = StringAttr::get(ctx, "dim" + std::to_string(op.getAxis()));
    const auto &src = *helper.getIntraWarpLayout();
    const auto &dst = *helper.getIntraWarpScanLayout();
    unsigned numSegments = src.getOutDimSize(axis);
    unsigned numRegs = dst.sublayout({kReg}, {axis}).getNumConsecutiveInOut();
    unsigned laneMask = getSegmentMask(dst, kLane, op.getAxis(), numSegments);
    unsigned chunkSize = numRegs * (1u << llvm::popcount(laneMask));
    unsigned numChunks = numSegments / chunkSize;
    unsigned segmentRegs = helper.getThreadLocalSegmentSize();
    unsigned registersPerScan = numRegs * segmentRegs;
    auto chunkLayout = helper.getPermutedLayout()
                           .resizeOutDim(axis, chunkSize * segmentRegs)
                           .removeZeroBasesAlongDim(kReg);
    ScanLoweringHelper chunkHelper(
        chunkLayout, op.getAxis(), op.getElementTypes(),
        /*preserveLaneOrder=*/true, helper.getCombineCost());
    unsigned registersPerChunk = chunkLayout.getInDimSize(kReg);
    unsigned nativeLaneMask = getSegmentMask(chunkLayout, kLane, op.getAxis(),
                                             chunkSize * segmentRegs);
    auto b = TritonLLVMOpBuilder(op.getLoc(), rewriter);
    bool reverse = op.getReverse();
    Value terminalLane = getTerminalLane(b, laneId, nativeLaneMask, reverse);
    ScanValues carries;
    for (unsigned i = 0; i < numChunks; ++i) {
      unsigned chunk = reverse ? numChunks - 1 - i : i;
      auto nativeReg = [&](unsigned r) {
        return (r / registersPerScan * numChunks + chunk) * registersPerScan +
               r % registersPerScan;
      };
      ScanValues local;
      for (unsigned r = 0; r < registersPerChunk; ++r)
        local.push_back(values[nativeReg(r)]);
      scanWithinThreads(op, local, segmentRegs, rewriter);
      auto totals = scanWithinWarps(op, chunkHelper, local, laneId, rewriter);
      applyScanCarries(op, chunkHelper, local, totals, {}, laneId, warpId,
                       rewriter);
      if (!carries.empty())
        for (unsigned r = 0; r < registersPerChunk; ++r)
          local[r] = combineWithPrefix(op, carries[r / registersPerScan],
                                       local[r], rewriter, {});
      carries.clear();
      if (i + 1 < numChunks)
        for (unsigned base = 0; base < registersPerChunk;
             base += registersPerScan)
          carries.push_back(shuffleValues(
              op.getLoc(), local[base + (reverse ? 0 : registersPerScan - 1)],
              terminalLane, rewriter));
      for (unsigned r = 0; r < registersPerChunk; ++r)
        values[nativeReg(r)] = std::move(local[r]);
    }
    return true;
  }

  // Scan thread-segment totals, leaving native prefixes untouched.
  ScanValues scanWithinWarps(triton::ScanOp op,
                             const ScanLoweringHelper &helper,
                             const ScanValues &values, Value laneId,
                             ConversionPatternRewriter &rewriter) const {
    const auto &intraWarpLayout = *helper.getIntraWarpLayout();
    const auto &scanLayout = *helper.getIntraWarpScanLayout();
    unsigned segmentRegs = helper.getThreadLocalSegmentSize();
    unsigned numSegments = helper.getWarpLocalSegmentSize() / segmentRegs;
    bool reverse = op.getReverse();

    auto totals = extractSegmentTotals(values, segmentRegs, reverse);
    convertScanValues(op, totals, intraWarpLayout, scanLayout, rewriter);

    auto kReg = StringAttr::get(op.getContext(), "register");
    auto axis =
        StringAttr::get(op.getContext(), "dim" + std::to_string(op.getAxis()));
    unsigned numRegs =
        scanLayout.sublayout({kReg}, {axis}).getNumConsecutiveInOut();
    auto kLane = StringAttr::get(op.getContext(), "lane");
    unsigned laneMask =
        getSegmentMask(scanLayout, kLane, op.getAxis(), numSegments);
    unsigned chunkSize = numRegs * (1u << llvm::popcount(laneMask));
    scanWithinThreads(op, totals, numRegs, rewriter);
    scanLaneTotals(op, totals, scanLayout, numRegs, chunkSize, laneId,
                   rewriter);
    unsigned numChunks = numSegments / chunkSize;
    if (numChunks > 1)
      scanChunkTotals(op, totals, numRegs, numChunks, laneMask, laneId,
                      rewriter);
    // Keep this layout for extracting inter-warp totals.
    return totals;
  }

  // Broadcast chunk totals, scan them, and apply the preceding chunk's carry.
  void scanChunkTotals(triton::ScanOp op, ScanValues &totals, unsigned numRegs,
                       unsigned numChunks, unsigned laneMask, Value laneId,
                       ConversionPatternRewriter &rewriter) const {
    bool reverse = op.getReverse();
    auto b = TritonLLVMOpBuilder(op.getLoc(), rewriter);
    auto chunkTotals = extractSegmentTotals(totals, numRegs, reverse);
    Value terminal = getTerminalLane(b, laneId, laneMask, reverse);
    for (auto &total : chunkTotals)
      total = shuffleValues(op.getLoc(), total, terminal, rewriter);
    scanWithinThreads(op, chunkTotals, numChunks, rewriter);
    // The first chunk has no carry. Every other chunk uses its predecessor's
    // complete total, in traversal order, without requiring an identity.
    for (unsigned base = 0; base < chunkTotals.size(); base += numChunks)
      for (unsigned i = 1; i < numChunks; ++i) {
        unsigned chunk = base + (reverse ? numChunks - 1 - i : i);
        unsigned prev = reverse ? chunk + 1 : chunk - 1;
        for (unsigned r = chunk * numRegs; r < (chunk + 1) * numRegs; ++r)
          totals[r] =
              combineWithPrefix(op, chunkTotals[prev], totals[r], rewriter, {});
      }
  }

  SmallVector<ScanCarry>
  accumulateConsumerCarries(triton::ScanOp op, const ScanValues &totals,
                            const LinearLayout &layout, unsigned numWarps,
                            Value warpId,
                            ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto *ctx = op.getContext();
    auto kReg = StringAttr::get(ctx, "register");
    auto axis = StringAttr::get(ctx, "dim" + std::to_string(op.getAxis()));
    unsigned numRegs = layout.getOutDimSize(axis) / numWarps;
    bool reverse = op.getReverse();
    Value warp =
        applyLinearLayout(
            loc, rewriter, layout,
            {{kReg, b.i32_val(0)},
             {StringAttr::get(ctx, "lane"), b.i32_val(0)},
             {StringAttr::get(ctx, "warp"), warpId},
             {StringAttr::get(ctx, "block"), b.i32_val(0)}})[op.getAxis()]
            .second;
    SmallVector<Value> beforeWarp;
    for (unsigned w = 0; w < numWarps; ++w)
      beforeWarp.push_back(reverse ? b.icmp_ult(warp, b.i32_val(w))
                                   : b.icmp_ugt(warp, b.i32_val(w)));
    SmallVector<ScanCarry> carries(layout.getInDimSize(kReg));
    for (unsigned base = 0; base < carries.size(); base += numRegs) {
      SmallVector<Value> acc;
      for (unsigned i = 0; i < numRegs; ++i) {
        unsigned r = base + (reverse ? numRegs - 1 - i : i);
        auto carry = acc;
        for (unsigned j = 0; j < numWarps; ++j) {
          unsigned w = reverse ? numWarps - 1 - j : j;
          // The last total cannot contribute to any exclusive carry.
          if (i == numRegs - 1 && j == numWarps - 1)
            break;
          acc = applyCombineOp(loc, rewriter, op.getCombineOp(), acc,
                               totals[r * numWarps + w]);
          if (carry.empty())
            carry = acc;
          else if (j != numWarps - 1)
            for (unsigned k = 0; k < carry.size(); ++k)
              carry[k] = b.select(beforeWarp[w], acc[k], carry[k]);
        }
        Value pred =
            i ? Value(b.true_val()) : beforeWarp[reverse ? numWarps - 1 : 0];
        carries[r] = {std::move(carry), pred};
      }
    }
    return carries;
  }

  // Scan contiguous partitions, then exchange their terminal prefixes.
  SmallVector<ScanCarry>
  scanAcrossWarps(triton::ScanOp op, const ScanLoweringHelper &helper,
                  const ScanValues &values, Value laneId, Value warpId,
                  ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto *ctx = op.getContext();
    auto kLane = StringAttr::get(ctx, "lane");
    // Use converted thread totals when available, otherwise native prefixes.
    const auto &sourceLayout = helper.getIntraWarpScanLayout()
                                   ? *helper.getIntraWarpScanLayout()
                                   : helper.getPermutedLayout();
    unsigned segmentSize = helper.getWarpLocalSegmentSize();
    // Express the segment size in source-layout units.
    if (helper.getIntraWarpScanLayout())
      segmentSize /= helper.getThreadLocalSegmentSize();
    const auto &interWarpLayout = *helper.getInterWarpLayout();
    const auto &totalsLayout = *helper.getInterWarpScanLayout();
    bool reverse = op.getReverse();
    auto kReg = StringAttr::get(ctx, "register");
    unsigned segmentRegs =
        sourceLayout.getInDimSize(kReg) / interWarpLayout.getInDimSize(kReg);
    unsigned segmentLaneMask =
        getSegmentMask(sourceLayout, kLane, op.getAxis(), segmentSize);

    // Only terminal lanes store complete segment totals; conversion loads
    // distribute them to the destination layout.
    auto totals = extractSegmentTotals(values, segmentRegs, reverse);
    Value storePred;
    if (segmentLaneMask)
      storePred = b.icmp_eq(b.and_(laneId, b.i32_val(segmentLaneMask)),
                            b.i32_val(reverse ? 0 : segmentLaneMask));
    auto free = sourceLayout.getFreeVariableMasks();
    for (auto [dim, id] : SmallVector<std::pair<StringAttr, Value>>{
             {kLane, laneId}, {StringAttr::get(ctx, "warp"), warpId}})
      if (free[dim]) {
        Value representative =
            b.icmp_eq(b.and_(id, b.i32_val(free[dim])), b.i32_val(0));
        storePred =
            storePred ? b.and_(storePred, representative) : representative;
      }
    // Load raw totals into their consumers and keep scratch read-only after
    // publication. No return exchange or second synchronization is needed.
    auto axis = StringAttr::get(ctx, "dim" + std::to_string(op.getAxis()));
    if (const auto &consumerLayout = helper.getConsumerTotalsLayout()) {
      auto operands = convertScanTotals(
          loc, rewriter, op, interWarpLayout, *consumerLayout,
          transposeValues(totals), getTypeConverter(), targetInfo, storePred);
      unsigned numWarps = consumerLayout->getInDimSize(kReg) /
                          interWarpLayout.getInDimSize(kReg);
      return accumulateConsumerCarries(op, transposeValues(operands),
                                       interWarpLayout, numWarps, warpId,
                                       rewriter);
    }
    auto operands = convertScanTotals(
        loc, rewriter, op, interWarpLayout, totalsLayout,
        transposeValues(totals), getTypeConverter(), targetInfo, storePred);
    totals = transposeValues(operands);

    // Reuse the warp-local scan, including sequences spanning registers.
    ScanLoweringHelper totalsHelper(totalsLayout, op.getAxis(),
                                    op.getElementTypes(), false,
                                    helper.getCombineCost());
    permuteRegisters(totals, totalsHelper.getRegisterOrder());
    scanWithinThreads(op, totals, totalsHelper.getThreadLocalSegmentSize(),
                      rewriter);
    ScanValues intraWarpTotals;
    if (totalsHelper.getIntraWarpLayout())
      intraWarpTotals =
          scanWithinWarps(op, totalsHelper, totals, laneId, rewriter);
    SmallVector<ScanCarry> partitionCarries;
    if (totalsHelper.getInterWarpLayout()) {
      // All warps must finish loading before the next exchange reuses scratch.
      targetInfo.barrier(loc, rewriter, triton::gpu::AddrSpace::Local);
      partitionCarries = scanAcrossWarps(
          op, totalsHelper, intraWarpTotals.empty() ? totals : intraWarpTotals,
          laneId, warpId, rewriter);
    }
    applyScanCarries(op, totalsHelper, totals, intraWarpTotals,
                     partitionCarries, laneId, warpId, rewriter);
    permuteRegisters(totals, totalsHelper.getRegisterOrder().inverse());
    // When totals span registers and lanes, return carries through shared
    // memory instead of shuffling each candidate source register.
    if (!totalsHelper.getInterWarpLayout() &&
        (!totalsHelper.getIntraWarpLayout() ||
         totalsLayout.sublayoutIsZero({kReg}, {axis})))
      return getSegmentCarries(op, totals, interWarpLayout, totalsLayout,
                               interWarpLayout.getOutDimSize(axis), laneId,
                               warpId, rewriter);

    targetInfo.barrier(loc, rewriter, triton::gpu::AddrSpace::Local);
    auto returned = transposeValues(
        convertScanTotals(loc, rewriter, op, totalsLayout, interWarpLayout,
                          transposeValues(totals), getTypeConverter(),
                          targetInfo, {}, op.getAxis(), reverse ? 1 : -1));
    SmallVector<ScanCarry> carries;
    for (auto [r, carry] : llvm::enumerate(returned)) {
      auto coords =
          applyLinearLayout(loc, rewriter, interWarpLayout,
                            {{kReg, b.i32_val(r)},
                             {kLane, laneId},
                             {StringAttr::get(ctx, "warp"), warpId},
                             {StringAttr::get(ctx, "block"), b.i32_val(0)}});
      Value pred = b.icmp_ne(
          coords[op.getAxis()].second,
          b.i32_val(reverse ? interWarpLayout.getOutDimSize(axis) - 1 : 0));
      carries.push_back({std::move(carry), pred});
    }
    return carries;
  }

  // Include inter-warp carries in the thread totals before fetching the
  // preceding thread's total, so each local prefix receives one complete carry.
  void applyScanCarries(triton::ScanOp op, const ScanLoweringHelper &helper,
                        ScanValues &values, ScanValues &intraWarpTotals,
                        ArrayRef<ScanCarry> interWarpCarries, Value laneId,
                        Value warpId,
                        ConversionPatternRewriter &rewriter) const {
    auto b = TritonLLVMOpBuilder(op.getLoc(), rewriter);
    auto kReg = StringAttr::get(op.getContext(), "register");
    unsigned interWarpRegs = 0;
    if (helper.getInterWarpLayout()) {
      const auto &layout = *helper.getInterWarpLayout();
      interWarpRegs = helper.getPermutedLayout().getInDimSize(kReg) /
                      layout.getInDimSize(kReg);
    }
    if (!helper.getIntraWarpLayout()) {
      if (!interWarpCarries.empty())
        applySegmentCarries(op, values, interWarpCarries, interWarpRegs,
                            rewriter);
      return;
    }

    const auto &intraWarpLayout = *helper.getIntraWarpLayout();
    unsigned segmentRegs = helper.getThreadLocalSegmentSize();
    unsigned numSegments = helper.getWarpLocalSegmentSize() / segmentRegs;
    convertScanValues(op, intraWarpTotals, *helper.getIntraWarpScanLayout(),
                      intraWarpLayout, rewriter);
    if (!interWarpCarries.empty())
      applySegmentCarries(op, intraWarpTotals, interWarpCarries,
                          interWarpRegs / segmentRegs, rewriter);

    // Terminal prefixes are complete; only the other registers need carries.
    for (unsigned r = 0; r < intraWarpTotals.size(); ++r)
      values[r * segmentRegs + (op.getReverse() ? 0 : segmentRegs - 1)] =
          intraWarpTotals[r];
    if (segmentRegs == 1)
      return;

    auto carries =
        getSegmentCarries(op, intraWarpTotals, intraWarpLayout, intraWarpLayout,
                          numSegments, laneId, warpId, rewriter);
    if (!interWarpCarries.empty()) {
      for (auto [r, carry] : llvm::enumerate(carries)) {
        const auto &interWarpCarry =
            interWarpCarries[r * segmentRegs / interWarpRegs];
        // The first thread segment has no preceding local total. Use the
        // inter-warp carry there, and leave the scan boundary untouched.
        for (auto [value, interWarpValue] :
             llvm::zip(carry.values, interWarpCarry.values))
          value = b.select(carry.pred, value, interWarpValue);
        carry.pred = b.or_(carry.pred, interWarpCarry.pred);
      }
    }
    applySegmentCarries(op, values, carries, segmentRegs, rewriter,
                        /*skipTerminal=*/true);
  }

  // Enumerate the source registers that the lanes of this segment can request.
  static SmallVector<unsigned>
  getCarryRegisters(const LinearLayout &segmentLayout,
                    const LinearLayout &inverseTotalsLayout, unsigned reg,
                    unsigned axis, unsigned segmentsPerScan, bool reverse) {
    auto *ctx = segmentLayout.getInDimNames().begin()->getContext();
    auto kReg = StringAttr::get(ctx, "register");
    auto kLane = StringAttr::get(ctx, "lane");
    auto kWarp = StringAttr::get(ctx, "warp");
    auto kBlock = StringAttr::get(ctx, "block");
    unsigned mask = segmentsPerScan - 1;
    llvm::SmallSetVector<unsigned, 8> candidates;
    for (unsigned warp = 0; warp < segmentLayout.getInDimSize(kWarp); ++warp)
      for (unsigned lane = 0; lane < segmentLayout.getInDimSize(kLane);
           ++lane) {
        auto coords = segmentLayout.apply(
            {{kReg, reg}, {kLane, lane}, {kWarp, warp}, {kBlock, 0}});
        auto &index = coords[axis].second;
        index = (index & ~mask) | ((index + (reverse ? 1 : -1)) & mask);
        candidates.insert(inverseTotalsLayout.apply(coords).front().second);
      }
    return llvm::to_vector(candidates);
  }

  // Select after shuffling: each destination lane may request a different
  // register from its source lane.
  SmallVector<Value> shuffleCarry(Location loc, const ScanValues &totals,
                                  ArrayRef<unsigned> candidates, Value srcReg,
                                  Value srcLane,
                                  ConversionPatternRewriter &rewriter) const {
    assert(!candidates.empty() && "each segment has a warp-local carry");
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto carry =
        shuffleValues(loc, totals[candidates.front()], srcLane, rewriter);
    for (unsigned candidate : candidates.drop_front()) {
      auto incoming = shuffleValues(loc, totals[candidate], srcLane, rewriter);
      Value select = b.icmp_eq(srcReg, b.i32_val(candidate));
      for (auto [value, source] : llvm::zip(carry, incoming))
        value = b.select(select, source, value);
    }
    return carry;
  }

  // Map each native segment to its exclusive carry in the totals layout.
  SmallVector<ScanCarry>
  getSegmentCarries(triton::ScanOp op, const ScanValues &totals,
                    const LinearLayout &segmentLayout,
                    const LinearLayout &totalsLayout, unsigned segmentsPerScan,
                    Value laneId, Value warpId,
                    ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto *ctx = op.getContext();
    auto kReg = StringAttr::get(ctx, "register");
    auto kLane = StringAttr::get(ctx, "lane");
    auto kWarp = StringAttr::get(ctx, "warp");
    auto kBlock = StringAttr::get(ctx, "block");
    auto axis = StringAttr::get(ctx, "dim" + std::to_string(op.getAxis()));
    bool reverse = op.getReverse();
    // All required totals are present in this warp.
    auto inverse = totalsLayout.pseudoinvert().sublayout(
        llvm::to_vector(totalsLayout.getOutDimNames()), {kReg, kLane});
    unsigned numSegments = segmentLayout.getOutDimSize(axis);
    SmallVector<ScanCarry> carries;
    for (unsigned r = 0; r < segmentLayout.getInDimSize(kReg); ++r) {
      // Evaluate the logical segment before shifting. CTA ownership is
      // unchanged.
      auto coords = applyLinearLayout(loc, rewriter, segmentLayout,
                                      {{kReg, b.i32_val(r)},
                                       {kLane, laneId},
                                       {kWarp, warpId},
                                       {kBlock, b.i32_val(0)}});
      Value segment = coords[op.getAxis()].second;
      Value segmentInScan = segment;
      if (segmentsPerScan != numSegments)
        segmentInScan = b.and_(segment, b.i32_val(segmentsPerScan - 1));
      Value pred = b.icmp_ne(segmentInScan,
                             b.i32_val(reverse ? segmentsPerScan - 1 : 0));
      // Wrap within this scan; pred excludes the boundary without an identity.
      coords[op.getAxis()].second = shiftWithinSegment(
          b, segment, reverse ? 1 : -1, segmentsPerScan, numSegments);
      auto src = applyLinearLayout(loc, rewriter, inverse, coords);
      Value srcReg = src[0].second;
      Value srcLane = src[1].second;
      auto candidates = getCarryRegisters(
          segmentLayout, inverse, r, op.getAxis(), segmentsPerScan, reverse);
      auto carry =
          shuffleCarry(loc, totals, candidates, srcReg, srcLane, rewriter);
      carries.push_back({std::move(carry), pred});
    }
    return carries;
  }

  void applySegmentCarries(triton::ScanOp op, ScanValues &values,
                           ArrayRef<ScanCarry> carries, unsigned segmentRegs,
                           ConversionPatternRewriter &rewriter,
                           bool skipTerminal = false) const {
    assert(values.size() == carries.size() * segmentRegs);
    unsigned begin = skipTerminal && op.getReverse() ? 1 : 0;
    unsigned end = segmentRegs - (skipTerminal && !op.getReverse());
    for (auto [r, carry] : llvm::enumerate(carries))
      for (unsigned j = begin; j < end; ++j) {
        unsigned reg = r * segmentRegs + j;
        values[reg] = combineWithPrefix(op, carry.values, values[reg], rewriter,
                                        carry.pred);
      }
  }

  // Guard the combine (including side effects) and preserve boundary values.
  SmallVector<Value> combineWithPrefix(triton::ScanOp op, ValueRange prefix,
                                       ValueRange values,
                                       ConversionPatternRewriter &rewriter,
                                       Value pred) const {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto combined =
        applyCombineOp(loc, rewriter, op.getCombineOp(), prefix, values, pred);
    if (pred)
      for (unsigned i = 0; i < combined.size(); ++i)
        combined[i] = b.select(pred, combined[i], values[i]);
    return combined;
  }

  const TargetInfoBase &targetInfo;
};
} // namespace

void mlir::triton::populateScanOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfoBase &targetInfo, PatternBenefit benefit) {
  patterns.add<blocked_scan::ScanOpConversion>(typeConverter, targetInfo,
                                               benefit.getBenefit() + 1);
  patterns.add<ScanOpConversion>(typeConverter, targetInfo, benefit);
}
