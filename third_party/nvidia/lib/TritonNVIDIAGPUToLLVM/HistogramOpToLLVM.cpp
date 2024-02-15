#include "Utility.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "triton/Analysis/Utility.h"

using namespace mlir;
using namespace mlir::triton;

static int log2Int(int64_t num) { return (num > 1) ? 1 + log2Int(num / 2) : 0; }

// Compute a histogram within a warp. This uses an algorithm by @apgoucher
// that does the following:
// Create a ballot for each bit of the bin index (there
// are only log2(num_bins) of these) and then apply bitwise operations to get
// the indicator functions for the bins owned by this particular thread, and
// only popcount those.
static SmallVector<Value>
computeWarpLevelHistogram(Location loc, RankedTensorType srcType,
                          SmallVector<Value> &srcValues, int numBins,
                          int numThreadPerWarp, Value threadId,
                          ConversionPatternRewriter &rewriter) {
  assert(numBins % numThreadPerWarp == 0 &&
         "numBins must be divisible by numThreadPerWarp");
  Value zero = i32_val(0);
  int numBits = log2Int(numBins);
  int numBitsLaneId = log2Int(numThreadPerWarp);
  unsigned numElementsPerThreads = triton::gpu::getTotalElemsPerThread(srcType);
  unsigned numThreadWithUniqueData =
      triton::gpu::getThreadsPerWarpWithUniqueData(srcType.getEncoding(),
                                                   srcType.getShape())[0];
  // The histogram is distributed across threads, each thread owns `numBins /
  // numThreadPerWarp` bins.
  SmallVector<Value> warpLevelHistogram(numBins / numThreadPerWarp, zero);
  for (int i = 0; i < numElementsPerThreads; ++i) {
    Value value = srcValues[i];
    SmallVector<Value> ballotBits;
    for (int j = 0; j < numBits; ++j) {
      Value bitSet = and_(value, i32_val(1 << j));
      Value threadMask = i32_val(-1);
      Value bit = rewriter.create<NVVM::VoteBallotOp>(loc, i32_ty, threadMask,
                                                      icmp_ne(bitSet, zero));
      ballotBits.push_back(bit);
    }
    Value fullMask = i32_val(0xFFFFFFFF);
    Value mask = fullMask;
    // If not all threads have unique data, mask out the redundant ones.
    if (numThreadWithUniqueData < numThreadPerWarp)
      mask = i32_val((1 << numThreadWithUniqueData) - 1);
    for (int i = 0; i < numBitsLaneId; i++) {
      Value updateMask = select(icmp_ne(and_(threadId, i32_val(1 << i)), zero),
                                zero, fullMask);
      mask =
          and_(mask, xor_(ballotBits[i + numBits - numBitsLaneId], updateMask));
    }
    // at this point, 'mask' tells you which elements are in a bin owned by this
    // thread.
    for (int k = 0; k < warpLevelHistogram.size(); k++) {
      Value binMask = mask;
      for (int j = 0; j < numBits - numBitsLaneId; j++) {
        Value updateMask = i32_val(((k & (1 << j)) ? 0 : 0xffffffff));
        binMask = and_(binMask, xor_(ballotBits[j], updateMask));
      }
      // at this point, 'bin_mask' tells you which elements are in the kth bin
      // owned by this thread.
      Value bitCount = rewriter.create<LLVM::CtPopOp>(loc, i32_ty, binMask);
      warpLevelHistogram[k] = add(warpLevelHistogram[k], bitCount);
    }
  }
  return warpLevelHistogram;
}

static void atomicAdd(Value ptr, Value val, Location loc,
                      ConversionPatternRewriter &rewriter) {
  rewriter.create<LLVM::AtomicRMWOp>(loc, LLVM::AtomicBinOp::add, ptr, val,
                                     LLVM::AtomicOrdering::monotonic);
}

static SmallVector<Value> computeCrossWarpHistogram(
    Location loc, ConversionPatternRewriter &rewriter, RankedTensorType srcType,
    Value baseSharedMemPtr, const SmallVector<Value> &warpLevelHistogram,
    int numBins, int numThreadPerWarp, const SmallVector<Value> &indices,
    Value threadId, int numWarps) {
  SmallVector<Value> histogramValues;
  unsigned numWarpsWithUniqueData =
      mlir::triton::gpu::getWarpsPerCTAWithUniqueData(srcType.getEncoding(),
                                                      srcType.getShape())[0];
  Value laneId = and_(threadId, i32_val(numThreadPerWarp - 1));
  // Initialize the shared memory with zeros.
  int64_t numElementPerThread =
      ceil<int64_t>(numBins, numThreadPerWarp * numWarps);
  for (int i = 0; i < numElementPerThread; ++i) {
    Value offset = add(threadId, i32_val((i * numWarps * numThreadPerWarp)));
    offset = urem(offset, i32_val(numBins));
    Value sharedMemPtr =
        gep(baseSharedMemPtr.getType(), i32_ty, baseSharedMemPtr, offset);
    store(i32_val(0), sharedMemPtr);
  }
  barrier();
  Block *afterAtomics = nullptr;
  // If some warps have replicated data we need to skip those warps when
  // accumulating.
  if (numWarpsWithUniqueData < numWarps) {
    Block *currentBlock = rewriter.getInsertionBlock();
    afterAtomics =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    Block *atomicBlock = rewriter.createBlock(afterAtomics);
    rewriter.setInsertionPointToEnd(currentBlock);
    Value cond =
        icmp_ult(threadId, i32_val(numWarpsWithUniqueData * numThreadPerWarp));
    rewriter.create<LLVM::CondBrOp>(loc, cond, atomicBlock, afterAtomics);
    rewriter.setInsertionPointToStart(atomicBlock);
  }
  // Apply atomic add to update the histogram in shared memory.
  for (int i = 0; i < warpLevelHistogram.size(); ++i) {
    Value warpLevelHistogramValue = warpLevelHistogram[i];
    Value offset =
        add(mul(laneId, i32_val(warpLevelHistogram.size())), i32_val(i));
    Value sharedMemPtr =
        gep(baseSharedMemPtr.getType(), i32_ty, baseSharedMemPtr, offset);
    atomicAdd(sharedMemPtr, warpLevelHistogramValue, loc, rewriter);
  }
  if (afterAtomics) {
    rewriter.create<LLVM::BrOp>(loc, afterAtomics);
    rewriter.setInsertionPointToStart(afterAtomics);
  }
  barrier();
  // load the histogram to register with the right layout.
  for (Value index : indices) {
    Value sharedMemPtr =
        gep(baseSharedMemPtr.getType(), i32_ty, baseSharedMemPtr, index);
    Value val = load(i32_ty, sharedMemPtr);
    histogramValues.push_back(val);
  }
  return histogramValues;
}

namespace {
struct HistogramOpConversion
    : public ConvertOpToLLVMPattern<triton::HistogramOp> {
public:
  using ConvertOpToLLVMPattern<triton::HistogramOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::HistogramOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = adaptor.getSrc();
    auto typeConverter = getTypeConverter();
    SmallVector<Value> srcValues = unpackLLElements(loc, input, rewriter);
    int numBins = op.getType().getDimSize(0);
    int numThreadsPerWarp = 32;
    // Pad out the bins so that we have at least one bin per thread within a
    // warp.
    numBins = std::max(numBins, numThreadsPerWarp);
    Value threadId = getThreadId(rewriter, loc);
    auto srcType = op.getSrc().getType();
    // First compute a warp local histogram based on values owned by each warps.
    SmallVector<Value> warpLevelHistogram =
        computeWarpLevelHistogram(loc, srcType, srcValues, numBins,
                                  numThreadsPerWarp, threadId, rewriter);

    // Then use atomic to update the histogram in shared memory.
    // TODO: we could skip this for cases with num_warps=1 as long as we can
    // generate the right layout. Currently the warp level histogram generates
    // data in the default blocked layout.
    Value baseSharedMemPtr =
        LLVM::getSharedMemoryBase(loc, rewriter, op.getOperation());
    auto dstType = op.getType();
    auto mod = op->getParentOfType<ModuleOp>();
    int numWarps = triton::gpu::TritonGPUDialect::getNumWarps(mod);
    Attribute dstEncoding = dstType.getEncoding();
    auto indices =
        emitIndices(op.getLoc(), rewriter, dstEncoding, dstType, true);
    SmallVector<Value> innerDimIndices;
    for (int i = 0; i < indices.size(); ++i)
      innerDimIndices.push_back(indices[i][0]);
    SmallVector<Value> histogramValue = computeCrossWarpHistogram(
        loc, rewriter, srcType, baseSharedMemPtr, warpLevelHistogram, numBins,
        numThreadsPerWarp, innerDimIndices, threadId, numWarps);

    Value results = packLLElements(loc, typeConverter, histogramValue, rewriter,
                                   op.getType());
    rewriter.replaceOp(op, results);
    return success();
  }
};
} // namespace

void mlir::triton::populateHistogramOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<HistogramOpConversion>(typeConverter, benefit);
}
