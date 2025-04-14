#include "Analysis/ScopeIdAllocation.h"
#include "Conversion/ProtonToProtonGPU/Passes.h"
#include "Dialect/Proton/IR/Dialect.h"
#include "Dialect/ProtonGPU/IR/Dialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/Support/MathExtras.h"

namespace mlir {
namespace triton {
namespace proton {

#define GEN_PASS_DEF_CONVERTPROTONTOPROTONGPU
#include "Conversion/ProtonToProtonGPU/Passes.h.inc"

namespace {

void parseSelectIds(llvm::StringRef selectIds,
                    llvm::SmallVectorImpl<int32_t> &selectIdVec) {
  auto rest = selectIds;
  while (!rest.empty()) {
    llvm::StringRef id;
    std::tie(id, rest) = rest.split(',');
    if (id.trim().size() > 0) {
      selectIdVec.push_back(std::stoi(id.str()));
    }
    if (rest.trim().size() == 0)
      break;
  }
  llvm::sort(selectIdVec);
  selectIdVec.erase(llvm::unique(selectIdVec), selectIdVec.end());
}

class RecordOpCircularRewrite : public OpRewritePattern<proton::RecordOp> {
public:
  RecordOpCircularRewrite(MLIRContext *ctx, Value buffer, Value index,
                          Value segmentBase, MetricType metricType,
                          ModuleScopeIdAllocation &scopeInfo)
      : OpRewritePattern::OpRewritePattern(ctx), buffer(buffer), index(index),
        segmentBase(segmentBase), metricType(metricType), scopeInfo(scopeInfo) {
  }

  LogicalResult matchAndRewrite(proton::RecordOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    MLIRContext *context = op.getContext();

    rewriter.setInsertionPointAfter(op);

    Value counter = rewriter.create<proton::gpu::ReadCounterOp>(
        op.getLoc(), mlir::IntegerType::get(context, 32), metricType);

    int scopeId = scopeInfo.getOpScopeId(op);
    rewriter.create<proton::gpu::CircularStoreOp>(op.getLoc(), buffer, index,
                                                  counter, segmentBase,
                                                  op.getIsStart(), scopeId);

    rewriter.eraseOp(op);
    return success();
  }

private:
  Value buffer;
  Value index;
  Value segmentBase;
  MetricType metricType;
  ModuleScopeIdAllocation &scopeInfo;
};
} // namespace

class ConvertProtonToProtonGPUPass
    : public impl::ConvertProtonToProtonGPUBase<ConvertProtonToProtonGPUPass> {
public:
  ConvertProtonToProtonGPUPass(MetricType metricType,
                               SamplingStrategy samplingStrategy,
                               llvm::StringRef samplingOptions,
                               gpu::Granularity granularity,
                               gpu::BufferStrategy bufferStrategy,
                               gpu::BufferType bufferType, int32_t bufferSize,
                               int64_t profileScratchSize,
                               int32_t profileScratchAlignment)
      : ConvertProtonToProtonGPUBase<ConvertProtonToProtonGPUPass>() {
    this->metricType = metricType;
    this->samplingStrategy = samplingStrategy;
    this->granularity = granularity;
    this->samplingOptions = samplingOptions.str();
    this->bufferStrategy = bufferStrategy;
    this->bufferType = bufferType;
    this->bufferSize = bufferSize;
    this->profileScratchSize = profileScratchSize;
    this->profileScratchAlignment = profileScratchAlignment;
  }

  LogicalResult circularRecordStrategyLowering(FuncOp func) {
    MLIRContext *context = func.getContext();
    Location loc = func->getLoc();
    ModuleOp mod = llvm::cast<ModuleOp>(func->getParentOp());
    OpBuilder builder(context);
    builder.setInsertionPointToStart(&func.getBody().front());
    int numWarps = mlir::triton::gpu::lookupNumWarps(mod);

    llvm::SmallVector<int32_t, 8> selectIdVec;
    int segmentNum = numWarps;
    if (!samplingOptions.empty() &&
        samplingStrategy == SamplingStrategy::SELECTIVE) {
      parseSelectIds(samplingOptions, selectIdVec);
      segmentNum = selectIdVec.size();
      if (segmentNum && granularity != proton::gpu::Granularity::WARP) {
        mlir::emitError(
            loc, "only warp granularity supports selective ids for now.");
        return failure();
      }
    }

    int sharedMemUsed = 0;
    if (mod->hasAttr("ttg.shared"))
      sharedMemUsed =
          mod->getAttrOfType<mlir::IntegerAttr>("ttg.shared").getInt();

    const int bytesPerEntry = proton::gpu::getBytesPerClockEntry();
    const int wordsPerEntry = bytesPerEntry / 4; // 1 word = 4 bytes
    const int circularHeaderSize =
        proton::gpu::getCircularHeaderSize(); // byte size

    // We take any available shared memory left to allocate the circular
    // buffer. The buffer size per segment must be power of 2.
    // Let's conservatively use the maximum shared memory size as 64KB bytes
    // because using a larger size may cause the kernel to perform poorly due to
    // L1 cache inefficiencies. In the future, we should construct device
    // specific targetInfo here to estimate the shared memory size
    auto maxSharedMemSize = 64 * 1024;
    int segmentByteSize =
        llvm::NextPowerOf2(
            (maxSharedMemSize - llvm::alignTo(sharedMemUsed, bytesPerEntry)) /
            segmentNum) /
        2;
    int numSharedEntries = segmentByteSize * segmentNum / bytesPerEntry;
    int allocSharedMemSize = numSharedEntries * bytesPerEntry;
    if (bufferSize != 0)
      bufferSize = llvm::alignTo(bufferSize, bytesPerEntry);
    int allocBufferSize = bufferSize > 0 ? bufferSize : allocSharedMemSize;
    if (!allocBufferSize) {
      mlir::emitError(loc, "profiling buffer size can't be 0.");
      return failure();
    }

    // Circular strategy memory layout (total: allocprofileScratchSize bytes)
    //  +-----------------------------------------------+
    //  | header (circularHeaderSize bytes)             |
    //  +-----------------------------------------------+
    //  | number of events per warp (4 bytes x numWarps)|
    //  +-----------------------------------------------+
    //  | profiled data (allocBufferSize bytes)         |
    //  +-----------------------------------------------+
    int allocprofileScratchSize =
        llvm::alignTo(allocBufferSize + circularHeaderSize + numWarps * 4,
                      profileScratchAlignment);

    if (profileScratchSize < allocprofileScratchSize) {
      mlir::emitError(loc,
                      "Global scratch memory for proton profiling is not large "
                      "enough, should be at least " +
                          llvm::Twine(allocprofileScratchSize) + " bytes.");
      return failure();
    }

    Value buffer;
    auto ctaLayout =
        triton::gpu::CTALayoutAttr::get(context, /*CTAsPerCGA=*/{1},
                                        /*CTASplitNum=*/{1}, /*CTAOrder=*/{0});
    auto encoding = triton::gpu::SwizzledSharedEncodingAttr::get(
        context, 1, 1, 1, {0}, ctaLayout);

    if (bufferType == gpu::BufferType::SHARED) {
      Attribute sharedMemorySpace =
          triton::gpu::SharedMemorySpaceAttr::get(context);
      auto sharedBufferType = triton::gpu::MemDescType::get(
          {wordsPerEntry * numSharedEntries}, builder.getI32Type(), encoding,
          sharedMemorySpace, /*mutable_memory=*/true);
      buffer = builder.create<triton::gpu::LocalAllocOp>(loc, sharedBufferType);
    } else if (bufferType == gpu::BufferType::LOCAL) {
      Attribute stackMemorySpace =
          mlir::triton::proton::gpu::StackMemorySpaceAttr::get(context);
      auto stackBufferType = triton::gpu::MemDescType::get(
          {wordsPerEntry * numSharedEntries}, builder.getI32Type(), encoding,
          stackMemorySpace, /*mutable_memory=*/true);
      buffer = builder.create<proton::gpu::StackAllocOp>(loc, stackBufferType);
    } else if (bufferType == gpu::BufferType::GLOBAL) {
      mlir::emitError(loc, "not implemented yet");
      return failure();
    } else {
      mlir::emitError(loc, "buffer-type not supported");
      return failure();
    }

    Value profileMem = builder.create<proton::gpu::GlobalScratchAllocOp>(
        loc, triton::getPointerType(builder.getI32Type()),
        allocprofileScratchSize, profileScratchAlignment);

    // Profiler index is private to each thread, address space is 5. In
    // practice, it doesn't prevent us from register promotion.
    auto ptrTy =
        triton::PointerType::get(mlir::IntegerType::get(context, 32), 5);
    Value index = builder.create<proton::gpu::InitBufferIndexOp>(loc, ptrTy);

    Value segmentBase = builder.create<proton::gpu::SegmentBaseOp>(
        loc, proton::gpu::SegmentBaseType::get(context), buffer, granularity,
        builder.getDenseI32ArrayAttr(selectIdVec));

    mlir::RewritePatternSet patterns(context);
    ModuleScopeIdAllocation &scopeInfo = getAnalysis<ModuleScopeIdAllocation>();
    patterns.add<RecordOpCircularRewrite>(context, buffer, index, segmentBase,
                                          metricType, scopeInfo);
    if (applyPatternsGreedily(mod, std::move(patterns)).failed())
      return failure();

    func.walk([&](triton::ReturnOp ret) {
      builder.setInsertionPoint(ret);
      builder.create<mlir::gpu::BarrierOp>(loc);
      builder.create<proton::gpu::FinalizeOp>(loc, buffer, index, profileMem);
    });

    return success();
  }

  void runOnOperation() override {
    ModuleOp m = getOperation();
    Location loc = m->getLoc();

    assert(metricType == MetricType::CYCLE &&
           "only cycle metric is supported now");

    int numFuncs = llvm::range_size(m.getOps<triton::FuncOp>());
    if (numFuncs > 0)
      // We currently only support one function in the module, which means all
      // functions needs to be marked as inlined.
      assert(numFuncs == 1 && "we only support one function in the module now");
    else
      return;

    FuncOp func = *m.getOps<triton::FuncOp>().begin();
    // Return if there is no proton record in the function.
    bool hasProtonRecord = false;
    func.walk([&](proton::RecordOp op) { hasProtonRecord = true; });
    if (!hasProtonRecord) {
      return;
    }

    if (bufferSize > 0 && !llvm::isPowerOf2_32(bufferSize)) {
      mlir::emitError(loc, "buffer-size must be power of 2");
      signalPassFailure();
      return;
    }

    if (!llvm::isPowerOf2_32(profileScratchAlignment)) {
      mlir::emitError(loc, "profileScratchAlignment must be power of 2");
      signalPassFailure();
      return;
    }

    if (bufferStrategy == gpu::BufferStrategy::CIRCULAR) {
      if (failed(circularRecordStrategyLowering(func)))
        signalPassFailure();
    } else {
      mlir::emitError(loc, "buffer-strategy is not supported");
      signalPassFailure();
    }

    return;
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createConvertProtonToProtonGPUPass(
    MetricType metricType, SamplingStrategy samplingStrategy,
    llvm::StringRef samplingOptions, gpu::Granularity granularity,
    gpu::BufferStrategy bufferStrategy, gpu::BufferType bufferType,
    int32_t bufferSize, int64_t profileScratchSize,
    int32_t profileScratchAlignment) {
  return std::make_unique<ConvertProtonToProtonGPUPass>(
      metricType, samplingStrategy, samplingOptions, granularity,
      bufferStrategy, bufferType, bufferSize, profileScratchSize,
      profileScratchAlignment);
}

} // namespace proton
} // namespace triton
} // namespace mlir
