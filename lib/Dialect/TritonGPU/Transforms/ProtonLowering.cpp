#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPUPROTONLOWERING
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

class ProtonRecordOpLowering : public OpRewritePattern<ProtonRecordOp> {
public:
  ProtonRecordOpLowering(MLIRContext *ctx, Value buf, Value ptr)
      : OpRewritePattern::OpRewritePattern(ctx), buffer(buf), index(ptr) {}

  LogicalResult matchAndRewrite(ProtonRecordOp op,
                                PatternRewriter &rewriter) const override {
    MLIRContext *ctx = op.getContext();
    rewriter.replaceOpWithNewOp<LocalRecordOp>(
        op, buffer, index, op.getIsStart(), op.getRegionId(), op.getMetric(),
        op.getGranularity());
    return success();
  }

private:
  Value buffer = nullptr;
  Value index = nullptr;
};

class TritonGPUProtonLoweringPass
    : public impl::TritonGPUProtonLoweringBase<TritonGPUProtonLoweringPass> {
public:
  TritonGPUProtonLoweringPass() = default;

  void runOnOperation() override {
    ModuleOp m = getOperation();
    MLIRContext *context = m.getContext();

    assert(llvm::range_size(m.getOps<triton::FuncOp>()) == 1 &&
           "expect only one function in the module");
    FuncOp func = *m.getOps<triton::FuncOp>().begin();

    Location loc = func.getLoc();

    // TODO (fywkevin): Remove the proton_record op when its metric attr is
    // "invalid".
    bool hasProtonRecordOp = false;
    func.walk([&](triton::ProtonRecordOp op) { hasProtonRecordOp = true; });
    if (!hasProtonRecordOp) {
      return;
    }

    //===--------------------------------------------------------------------===//
    // Allocate shared memory resources.
    //===--------------------------------------------------------------------===//

    OpBuilder builder(context);
    builder.setInsertionPointToStart(&func.getBody().front());

    const int wordsPerEntry = triton::gpu::getWordsPerProtonEntry();
    int slots =
        cast<IntegerAttr>(m->getAttr("triton_gpu.proton-slots")).getInt();

    // Alloc the shared memory for buffer (uninitialized).
    Attribute sharedMemorySpace =
        triton::gpu::SharedMemorySpaceAttr::get(context);
    auto ctaLayout =
        triton::gpu::CTALayoutAttr::get(context, /*CTAsPerCGA=*/{1},
                                        /*CTASplitNum=*/{1}, /*CTAOrder=*/{0});
    auto encoding =
        triton::gpu::SharedEncodingAttr::get(context, 1, 1, 1, {0}, ctaLayout);
    auto bufferType =
        MemDescType::get({wordsPerEntry * slots}, builder.getI32Type(),
                         encoding, sharedMemorySpace, /*mutable_memory=*/true);
    Value buffer = builder.create<triton::gpu::LocalAllocOp>(loc, bufferType);

    //===--------------------------------------------------------------------===//
    // Insert and lower Proton operators.
    //===--------------------------------------------------------------------===//

    builder.setInsertionPointToStart(&func.getBody().front());
    auto ptrTy =
        triton::PointerType::get(mlir::IntegerType::get(context, 32), 1);
    Value index = builder.create<ProtonInitOp>(loc, ptrTy);

    mlir::RewritePatternSet patterns(context);
    patterns.add<ProtonRecordOpLowering>(context, buffer, index);
    if (applyPatternsAndFoldGreedily(m, std::move(patterns)).failed())
      signalPassFailure();

    Operation *ret = &func.getBody().front().back();
    builder.setInsertionPoint(ret);
    Value profileMem = func.getArguments().back();
    builder.create<ProtonFinalizeOp>(loc, buffer, index, profileMem);
    return;
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
