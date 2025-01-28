#include "TritonNVIDIAGPUToLLVM/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/Patterns.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_DECOMPOSEUNSUPPORTEDNVIDIACONVERSIONS
#include "TritonNVIDIAGPUToLLVM/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

using namespace mlir;
using namespace triton;
using namespace triton::gpu;

// Loading from Hopper shared memory layout to dot operand is not supported. We
// need to break it down and use a different shared layout. This would mostly
// happen when TMAs are used with MMAV2 and will cause poor performance.
class DecomposeLocalLoadToDotOperand
    : public OpRewritePattern<triton::gpu::LocalLoadOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::gpu::LocalLoadOp op,
                                PatternRewriter &rewriter) const override {

    auto dstDotOp = dyn_cast<triton::gpu::DotOperandEncodingAttr>(
        op.getType().getEncoding());
    MemDescType srcType = op.getSrc().getType();
    auto sharedEncoding = dyn_cast<SharedEncodingAttr>(srcType.getEncoding());
    if (!dstDotOp || !sharedEncoding || !sharedEncoding.getHasLeadingOffset())
      return failure();
    RankedTensorType type = op.getType();
    auto parentEnc = dstDotOp.getParent();
    int numWarps = triton::gpu::getNumWarpsPerCTA(parentEnc);
    int threadsPerWarp = triton::gpu::TritonGPUDialect::getThreadsPerWarp(
        op->getParentOfType<ModuleOp>());
    int numCTAs = triton::gpu::getNumCTAs(parentEnc);
    auto blockEncoding = getDefaultBlockedEncoding(
        op.getContext(), type.getShape(), numWarps, threadsPerWarp, numCTAs);
    auto tmpType = RankedTensorType::get(type.getShape(), type.getElementType(),
                                         blockEncoding);
    Value load =
        rewriter.create<LocalLoadOp>(op.getLoc(), tmpType, op.getSrc());
    auto newSharedDescTy = MemDescType::get(
        type.getShape(), type.getElementType(),
        triton::gpu::SharedEncodingAttr::get(
            op.getContext(), dstDotOp, type.getShape(),
            triton::gpu::getOrder(parentEnc),
            triton::gpu::getCTALayout(parentEnc), type.getElementType()),
        srcType.getMemorySpace());
    auto tmp = rewriter.create<triton::gpu::LocalAllocOp>(
        op.getLoc(), newSharedDescTy, load);
    auto newConvert =
        rewriter.create<triton::gpu::LocalLoadOp>(op.getLoc(), type, tmp);
    rewriter.replaceOp(op, newConvert);
    return success();
  }
};

struct DecomposeUnsupportedConversions
    : public mlir::triton::impl::DecomposeUnsupportedNVIDIAConversionsBase<
          DecomposeUnsupportedConversions> {
  void runOnOperation() override {
    // FIXME [Dot LL]
    // Remove the decomposeTensorCoreToDotLayoutConversion class entirely after
    // we have enabled the new layout conversion for all the cases.
    auto nvidiaShortCutFn = [&](RankedTensorType srcTy,
                                RankedTensorType dstTy) { return true; };
    ModuleOp mod = getOperation();
    triton::gpu::decomposeSplatOpToSharedLayoutConversion(mod);
    triton::gpu::decomposeTensorCoreToDotLayoutConversion(mod,
                                                          nvidiaShortCutFn);
    triton::gpu::decomposeBlockedToDotLayoutConversion(mod);

    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<DecomposeLocalLoadToDotOperand>(&getContext());
    if (mlir::applyPatternsGreedily(mod, std::move(patterns)).failed()) {
      signalPassFailure();
    }
  }
};
} // namespace

namespace mlir::triton::NVIDIA {

std::unique_ptr<OperationPass<ModuleOp>>
createDecomposeUnsupportedConversionsPass() {
  return std::make_unique<DecomposeUnsupportedConversions>();
}

} // namespace mlir::triton::NVIDIA
