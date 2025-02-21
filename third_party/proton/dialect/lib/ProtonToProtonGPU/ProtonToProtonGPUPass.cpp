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
// FIXME(fywkevin): This is a placeholder for now. After Jeff's WS support in
// place, we should revisit and adjust this accordingly.
const int getWarpNumPerGroup() { return 4; }

class RecordOpCircularRewrite : public OpRewritePattern<proton::RecordOp> {
public:
  RecordOpCircularRewrite(MLIRContext *ctx, Value buffer, Value index,
                          StringRef metric, StringRef granularity,
                          ModuleScopeIdAllocation &scopeInfo)
      : OpRewritePattern::OpRewritePattern(ctx), buffer(buffer), index(index),
        metric(metric), granularity(granularity), scopeInfo(scopeInfo) {}

  LogicalResult matchAndRewrite(proton::RecordOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    MLIRContext *context = op.getContext();
    auto granularityEnum = granularity == "warpgroup"
                               ? proton::gpu::Granularity::WARPGROUP
                               : proton::gpu::Granularity::WARP;

    rewriter.setInsertionPointAfter(op);

    Value counter = rewriter.create<proton::gpu::ReadCounterOp>(
        op.getLoc(), mlir::IntegerType::get(context, 32),
        proton::Metric::CYCLE);

    int scopeId = scopeInfo.getOpScopeId(op);
    rewriter.create<proton::gpu::CircularStoreOp>(op.getLoc(), buffer, index,
                                                  counter, op.getIsStart(),
                                                  scopeId, granularityEnum);

    rewriter.eraseOp(op);
    return success();
  }

private:
  Value buffer;
  Value index;
  StringRef metric;
  StringRef granularity;
  ModuleScopeIdAllocation &scopeInfo;
};
} // namespace

class ConvertProtonToProtonGPUPass
    : public impl::ConvertProtonToProtonGPUBase<ConvertProtonToProtonGPUPass> {
public:
  ConvertProtonToProtonGPUPass(std::string metric, std::string granularity,
                               int32_t maxSharedMem, int32_t scratchMem,
                               int32_t alignment, std::string strategy,
                               std::string bufferType, int32_t bufferSize)
      : ConvertProtonToProtonGPUBase<ConvertProtonToProtonGPUPass>() {
    this->metric = metric;
    this->granularity = granularity;
    this->maxSharedMem = maxSharedMem;
    this->scratchMem = scratchMem;
    this->alignment = alignment;
    this->strategy = strategy;
    this->bufferType = bufferType;
    this->bufferSize = bufferSize;
  }

  LogicalResult circularRecordStrategyLowering(FuncOp func) {
    MLIRContext *context = func.getContext();
    Location loc = func->getLoc();
    ModuleOp mod = llvm::cast<ModuleOp>(func->getParentOp());
    OpBuilder builder(context);
    builder.setInsertionPointToStart(&func.getBody().front());

    int sharedMemUsed = 0;
    if (mod->hasAttr("ttg.shared"))
      sharedMemUsed =
          mod->getAttrOfType<mlir::IntegerAttr>("ttg.shared").getInt();

    const int bytesPerEntry = proton::gpu::getBytesPerClockEntry();
    const int wordsPerEntry = bytesPerEntry / 4; // 1 word = 4 bytes
    const int circularHeaderSize = 16;           // byte size
    // TODO(fywkevin): we should consider the WS support for shared memory
    // allocation.
    int sharedSlots =
        llvm::PowerOf2Ceil((maxSharedMem - sharedMemUsed) / bytesPerEntry);
    int allocSharedMemSize = sharedSlots * bytesPerEntry;
    int numWarps = triton::gpu::TritonGPUDialect::getNumWarps(mod);
    int allocBufferSize = bufferSize > 0 ? bufferSize : allocSharedMemSize;
    if (!allocBufferSize) {
      mlir::emitError(loc, "profiling buffer size can't be 0.");
      return failure();
    }

    // Circular strategy memory layout (total: allocScratchMemSize bytes)
    //  +-----------------------------------------------+
    //  | header (circularHeaderSize bytes)             |
    //  +-----------------------------------------------+
    //  | number of events per warp (4 bytes x numWarps)|
    //  +-----------------------------------------------+
    //  | profiled data (allocBufferSize bytes)         |
    //  +-----------------------------------------------+
    int allocScratchMemSize = llvm::alignTo(
        allocBufferSize + circularHeaderSize + numWarps * 4, alignment);

    if (scratchMem < allocScratchMemSize) {
      mlir::emitError(loc,
                      "Global scratch memory for proton profiling is not large "
                      "enough, should be at least " +
                          llvm::Twine(allocScratchMemSize) + " bytes.");
      return failure();
    }

    Value buffer;
    if (bufferType == "shared_mem") {
      Attribute sharedMemorySpace =
          triton::gpu::SharedMemorySpaceAttr::get(context);
      auto ctaLayout = triton::gpu::CTALayoutAttr::get(
          context, /*CTAsPerCGA=*/{1},
          /*CTASplitNum=*/{1}, /*CTAOrder=*/{0});

      // TODO(fywkevin): add another shared encoding attribute derived from the
      // SharedEncodingTrait, and use it here.
      auto encoding = triton::gpu::SwizzledSharedEncodingAttr::get(
          context, 1, 1, 1, {0}, ctaLayout);
      auto sharedBufferType = triton::gpu::MemDescType::get(
          {wordsPerEntry * sharedSlots}, builder.getI32Type(), encoding,
          sharedMemorySpace, /*mutable_memory=*/true);
      buffer = builder.create<triton::gpu::LocalAllocOp>(loc, sharedBufferType);
    } else {
      mlir::emitError(loc, "buffer-type not supported");
      return failure();
    }

    Value profileMem = builder.create<proton::gpu::GlobalScratchAllocOp>(
        loc, triton::getPointerType(builder.getI32Type()), allocScratchMemSize,
        alignment);

    // Profiler index is private to each thread, address space is 5. In
    // practice, it doesn't prevent us from register promotion.
    auto ptrTy =
        triton::PointerType::get(mlir::IntegerType::get(context, 32), 5);
    Value index = builder.create<proton::gpu::InitBufferIndexOp>(loc, ptrTy);

    mlir::RewritePatternSet patterns(context);
    ModuleScopeIdAllocation &scopeInfo = getAnalysis<ModuleScopeIdAllocation>();
    patterns.add<RecordOpCircularRewrite>(context, buffer, index, metric,
                                          granularity, scopeInfo);
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

    assert(metric == "cycle" && "only cycle metric is supported now");

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

    if (bufferType != "shared_mem" && bufferType != "stack_mem") {
      mlir::emitError(loc, "buffer-type must be shared_mem or stack_mem");
      signalPassFailure();
      return;
    }

    if (bufferType != "shared_mem" && bufferSize == 0) {
      mlir::emitError(loc, "buffer-size must be greater than 0");
      signalPassFailure();
      return;
    }

    if (bufferSize > 0 && !llvm::isPowerOf2_32(bufferSize)) {
      mlir::emitError(loc, "buffer-size must be power of 2");
      signalPassFailure();
      return;
    }

    if (!llvm::isPowerOf2_32(alignment)) {
      mlir::emitError(loc, "alignment must be power of 2");
      signalPassFailure();
      return;
    }

    if (granularity != "warp" && granularity != "warpgroup") {
      mlir::emitError(loc, "granularity must be warp or warpgroup");
      signalPassFailure();
      return;
    }

    if (strategy == "circular") {
      if (failed(circularRecordStrategyLowering(func)))
        signalPassFailure();
    } else {
      mlir::emitError(loc, "strategy is not supported");
      signalPassFailure();
    }

    return;
  }
};

std::unique_ptr<OperationPass<ModuleOp>>
createConvertProtonToProtonGPUPass(std::string metric, std::string granularity,
                                   int32_t maxSharedMem, int32_t scratchMem,
                                   int32_t alignment, std::string strategy,
                                   std::string bufferType, int32_t bufferSize) {
  return std::make_unique<ConvertProtonToProtonGPUPass>(
      metric, granularity, maxSharedMem, scratchMem, alignment, strategy,
      bufferType, bufferSize);
}

} // namespace proton
} // namespace triton
} // namespace mlir
