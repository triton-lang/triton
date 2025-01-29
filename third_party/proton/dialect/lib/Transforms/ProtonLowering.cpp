#include "Dialect/Proton/IR/Dialect.h"
#include "Transforms/Passes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace triton {
namespace proton {

#define GEN_PASS_DEF_PROTONLOWERING
#include "Transforms/Passes.h.inc"

namespace {
int maxPowerof2(unsigned int n) {
  if (n < 1)
    return 0;
  int res = 1;
  for (int i = 0; i < 8 * sizeof(unsigned int); i++) {
    int curr = 1 << i;
    if (curr > n)
      break;
    res = curr;
  }
  return res;
}
} // namespace

class RecordOpLowering : public OpRewritePattern<proton::RecordOp> {
public:
  RecordOpLowering(MLIRContext *ctx, Value buf, Value ptr)
      : OpRewritePattern::OpRewritePattern(ctx), buffer(buf), index(ptr) {}

  LogicalResult matchAndRewrite(proton::RecordOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getStrategy() == Strategy::CIRCULAR) {
      rewriter.replaceOpWithNewOp<proton::CircularRecordOp>(
          op, buffer, index, op.getIsStart(), op.getRegionId(), op.getMetric(),
          op.getGranularity());
      return success();
    }

    rewriter.eraseOp(op);
    return failure();
  }

private:
  Value buffer = nullptr;
  Value index = nullptr;
};

class ProtonLoweringPass : public impl::ProtonLoweringBase<ProtonLoweringPass> {
public:
  ProtonLoweringPass(int32_t maxSharedMem, int32_t scratchMem,
                     int32_t alignment)
      : ProtonLoweringBase<ProtonLoweringPass>() {
    this->maxSharedMem = maxSharedMem;
    this->scratchMem = scratchMem;
    this->alignment = alignment;
  }

  void runOnOperation() override {
    ModuleOp m = getOperation();
    MLIRContext *context = m.getContext();

    assert(llvm::range_size(m.getOps<triton::FuncOp>()) == 1 &&
           "we only support one function in the module now");

    FuncOp func = *m.getOps<triton::FuncOp>().begin();

    Location loc = func.getLoc();

    bool hasProtonRecord = false;
    func.walk([&](proton::RecordOp op) { hasProtonRecord = true; });
    if (!hasProtonRecord) {
      return;
    }

    int sharedMemUsed = 0;
    if (m->hasAttr("ttg.shared"))
      sharedMemUsed =
          m->getAttrOfType<mlir::IntegerAttr>("ttg.shared").getInt();

    int scratchMemUsed = 0;
    int scratchMemAlignment = alignment;
    if (m->hasAttr("ttg.global_scratch_memory_size")) {
      scratchMemUsed =
          m->getAttrOfType<mlir::IntegerAttr>("ttg.global_scratch_memory_size")
              .getInt();
      scratchMemAlignment = m->getAttrOfType<mlir::IntegerAttr>(
                                 "ttg.global_scratch_memory_alignment")
                                .getInt();
    }

    OpBuilder builder(context);
    builder.setInsertionPointToStart(&func.getBody().front());

    const int bytesPerEntry = getBytesPerEntry();
    const int wordsPerEntry = bytesPerEntry / 4;
    const int headerSize = getHeaderSize();
    int slots = maxPowerof2((maxSharedMem - sharedMemUsed) / bytesPerEntry);
    int allocSharedMemSize = slots * bytesPerEntry;
    int numWarps = triton::gpu::TritonGPUDialect::getNumWarps(m);
    int minAllocScratchMemSize = allocSharedMemSize + headerSize + numWarps * 4;

    if (scratchMem < minAllocScratchMemSize) {
      mlir::emitError(loc, "Global scratch memory for proton is not large "
                           "enough, should be at least " +
                               llvm::Twine(minAllocScratchMemSize) + " bytes.");
      signalPassFailure();
      return;
    }

    // Memory resource allocation (e.g., global scratch memory, shared memory).
    Attribute sharedMemorySpace =
        triton::gpu::SharedMemorySpaceAttr::get(context);
    auto ctaLayout =
        triton::gpu::CTALayoutAttr::get(context, /*CTAsPerCGA=*/{1},
                                        /*CTASplitNum=*/{1}, /*CTAOrder=*/{0});
    auto encoding =
        triton::gpu::SharedEncodingAttr::get(context, 1, 1, 1, {0}, ctaLayout);
    auto bufferType = triton::gpu::MemDescType::get(
        {wordsPerEntry * slots}, builder.getI32Type(), encoding,
        sharedMemorySpace, /*mutable_memory=*/true);
    Value buffer = builder.create<triton::gpu::LocalAllocOp>(loc, bufferType);
    buffer.getDefiningOp()->setAttr(
        "allocation.offset",
        IntegerAttr::get(IntegerType::get(context, 32), sharedMemUsed));

    int scratchOffset = scratchMemUsed + scratchMemAlignment - 1;
    scratchOffset = scratchOffset - (scratchOffset % scratchMemAlignment);
    Value profileMem = builder.create<triton::gpu::GlobalScratchAllocOp>(
        loc, triton::getPointerType(builder.getI32Type()), scratchMem,
        scratchMemAlignment);
    profileMem.getDefiningOp()->setAttr(
        "ttg.global_scratch_memory_offset",
        IntegerAttr::get(IntegerType::get(context, 32), scratchOffset));

    // Lower Proton RecordOp and add InitOp.
    builder.setInsertionPointToStart(&func.getBody().front());
    auto ptrTy =
        triton::PointerType::get(mlir::IntegerType::get(context, 32), 1);
    Value index = builder.create<InitOp>(loc, ptrTy);

    mlir::RewritePatternSet patterns(context);
    patterns.add<RecordOpLowering>(context, buffer, index);
    if (applyPatternsGreedily(m, std::move(patterns)).failed())
      signalPassFailure();

    // Finalize the profiler, adding FinalizedOp before return.
    func.walk([&](triton::ReturnOp ret) {
      builder.setInsertionPoint(ret);
      builder.create<FinalizeOp>(
          loc, buffer, index, profileMem,
          mlir::IntegerAttr::get(mlir::IntegerType::get(context, 32),
                                 scratchMem));
    });

    // Update module-level attributes.
    m->setAttr("ttg.shared",
               mlir::IntegerAttr::get(mlir::IntegerType::get(context, 32),
                                      maxSharedMem));
    m->setAttr("ttg.global_scratch_memory_size",
               mlir::IntegerAttr::get(mlir::IntegerType::get(context, 32),
                                      scratchMem + scratchOffset));
    m->setAttr("ttg.global_scratch_memory_alignment",
               mlir::IntegerAttr::get(mlir::IntegerType::get(context, 32),
                                      scratchMemAlignment));

    return;
  }
};

std::unique_ptr<OperationPass<ModuleOp>>
createProtonLowering(int32_t maxSharedMem, int32_t scratchMem,
                     int32_t alignment) {
  return std::make_unique<ProtonLoweringPass>(maxSharedMem, scratchMem,
                                              alignment);
}

} // namespace proton
} // namespace triton
} // namespace mlir
