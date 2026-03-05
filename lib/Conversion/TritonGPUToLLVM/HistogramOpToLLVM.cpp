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
computeHistogram(Location loc, ConversionPatternRewriter &rewriter,
                 Value baseSharedMemPtr, const SmallVector<Value> &srcValues,
                 const SmallVector<Value> &maskValues, int numBins,
                 int numThreadPerWarp, const SmallVector<Value> &indices,
                 Value threadId, int numWarps) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  SmallVector<Value> histogramValues;
  // Initialize the shared memory with zeros.
  int64_t numElementPerThread =
      ceil<int64_t>(numBins, numThreadPerWarp * numWarps);
  for (int i = 0; i < numElementPerThread; ++i) {
    Value offset =
        b.add(threadId, b.i32_val((i * numWarps * numThreadPerWarp)));
    offset = b.urem(offset, b.i32_val(numBins));
    Value sharedMemPtr =
        b.gep(baseSharedMemPtr.getType(), i32_ty, baseSharedMemPtr, offset);
    b.store(b.i32_val(0), sharedMemPtr);
  }
  b.barrier(triton::gpu::AddrSpace::Local);

  // Apply atomic add to update the histogram in shared memory.
  Value numBinsValue = b.i32_val(numBins);
  for (int i = 0; i < srcValues.size(); ++i) {
    Value updatePred = b.icmp_ult(srcValues[i], numBinsValue);
    if (!maskValues.empty())
      updatePred = b.and_(updatePred, maskValues[i]);

    auto [prevBlock, ifBlock, thenBlock] =
        createIfBlock(rewriter, loc, updatePred);
    (void)prevBlock;
    rewriter.setInsertionPointToStart(ifBlock);
    Value sharedMemPtr = b.gep(baseSharedMemPtr.getType(), i32_ty,
                               baseSharedMemPtr, srcValues[i]);
    atomicAddOne(sharedMemPtr, loc, rewriter);
    rewriter.setInsertionPointToStart(thenBlock);
  }

  b.barrier(triton::gpu::AddrSpace::Local);
  // load the histogram to register with the right layout.
  for (Value index : indices) {
    Value sharedMemPtr =
        b.gep(baseSharedMemPtr.getType(), i32_ty, baseSharedMemPtr, index);
    Value val = b.load(i32_ty, sharedMemPtr);
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
    Value input = adaptor.getSrc();
    auto typeConverter = getTypeConverter();
    SmallVector<Value> srcValues = unpackLLElements(loc, input, rewriter);

    Value llMask = adaptor.getMask();
    SmallVector<Value> maskValues;
    if (llMask)
      maskValues = unpackLLElements(loc, llMask, rewriter);

    int numBins = op.getType().getDimSize(0);
    auto mod = op->getParentOfType<ModuleOp>();
    int numThreadsPerWarp =
        triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
    assert(numThreadsPerWarp == 32 ||
           numThreadsPerWarp == 64 &&
               "Only supports 32 or 64 threads per warp");
    int numWarps = triton::gpu::lookupNumWarps(op);
    Value threadId = getThreadId(rewriter, loc);
    auto srcType = op.getSrc().getType();

    // Use atomic adds to update the histogram in shared memory.
    Value baseSharedMemPtr =
        LLVM::getSharedMemoryBase(loc, rewriter, targetInfo, op.getOperation());
    auto dstType = op.getType();
    Attribute dstEncoding = dstType.getEncoding();
    auto indices = emitIndices(op.getLoc(), rewriter, targetInfo, dstEncoding,
                               dstType, true);
    SmallVector<Value> innerDimIndices;
    for (int i = 0; i < indices.size(); ++i)
      innerDimIndices.push_back(indices[i][0]);
    SmallVector<Value> histogramValue = computeHistogram(
        loc, rewriter, baseSharedMemPtr, srcValues, maskValues, numBins,
        numThreadsPerWarp, innerDimIndices, threadId, numWarps);

    // Depending on the layout, some threads may have duplicate data. We can
    // account for this by calculating a "replication factor" and dividing the
    // results by it to avoid overcounting.
    auto replicationFactor = numWarps * numThreadsPerWarp;
    auto threadsPerWarp = getThreadsPerWarp(srcType);
    auto warpsPerCTA =
        getWarpsPerCTA(srcType.getEncoding(), srcType.getShape());
    replicationFactor /= std::accumulate(
        threadsPerWarp.begin(), threadsPerWarp.end(), 1, std::multiplies<>());
    replicationFactor /= std::accumulate(warpsPerCTA.begin(), warpsPerCTA.end(),
                                         1, std::multiplies<>());

    auto b = TritonLLVMOpBuilder(loc, rewriter);
    for (auto i = 0; i < histogramValue.size(); ++i) {
      histogramValue[i] =
          b.sdiv(histogramValue[i], b.i32_val(replicationFactor));
    }

    Value results = packLLElements(loc, typeConverter, histogramValue, rewriter,
                                   op.getType());
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
