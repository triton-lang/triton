#include "TritonAMDGPUToLLVM/TargetUtils.h"
#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include <memory>

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace {
bool supportedGlobalAtomicPacked(StringRef arch) {
  if (arch.contains("gfx12"))
    return true;

  auto isaFamily = tt::AMD::deduceISAFamily(arch);
  return isaFamily == tt::AMD::ISAFamily::CDNA1 ||
         isaFamily == tt::AMD::ISAFamily::CDNA2 ||
         isaFamily == tt::AMD::ISAFamily::CDNA3;
}

// This rewriter is required to recombine datalayouts for atomicRMW fadd
// operation to be able to pack 2 16 bit elements and process them by a single
// instruction.
class AtomicRmwPackF16 : public OpRewritePattern<tt::AtomicRMWOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  AtomicRmwPackF16(MLIRContext *context)
      : OpRewritePattern<tt::AtomicRMWOp>(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(tt::AtomicRMWOp atomOp,
                                PatternRewriter &rewriter) const override {
    auto atomicRmwBinOp = atomOp.getAtomicRmwOp();
    if (atomicRmwBinOp != RMWOp::FADD)
      return failure();

    auto ctx = atomOp->getContext();

    Value ptr = atomOp.getPtr();
    Value val = atomOp.getVal();
    Value mask = atomOp.getMask();

    Value opResult = atomOp.getResult();
    auto tensorTy = dyn_cast<RankedTensorType>(opResult.getType());
    if (!tensorTy)
      return failure();

    auto oldEncoding = tensorTy.getEncoding();
    if (!oldEncoding || !isa<ttg::BlockedEncodingAttr>(oldEncoding))
      return failure();

    Type resElemTy = tensorTy.getElementType();
    const size_t resElemNbits = resElemTy.getIntOrFloatBitWidth();
    if (!isa<FloatType>(resElemTy) || resElemNbits != 16)
      return failure();

    auto order = ttg::getOrder(oldEncoding);
    auto oldTotalElemsPerThread = ttg::getElemsPerThread(tensorTy);
    // No need to do anything. Elements already could be collected to pairs
    if (oldTotalElemsPerThread[order[0]] % 2 == 0)
      return failure();

    // Just multiply elements per thread by 2. Layout will be built considering
    // this. There are cases where some threads will be disabled. It is also
    // better for perf because in the not paired case we have to provide unique
    // access to each element, this leads to overhead.
    auto packedElemsPerThread = oldTotalElemsPerThread;
    packedElemsPerThread[order[0]] *= 2;

    auto mod = atomOp->getParentOfType<ModuleOp>();
    int numWarps = ttg::TritonGPUDialect::getNumWarps(mod);
    int numThreads = ttg::TritonGPUDialect::getThreadsPerWarp(mod);
    int numCTAs = ttg::TritonGPUDialect::getNumCTAs(mod);
    auto shape = tensorTy.getShape();
    auto newPackedEncoding = ttg::BlockedEncodingAttr::get(
        ctx, shape, packedElemsPerThread, order, numWarps, numThreads, numCTAs);

    // Convert val, ptr, mask to packed layout
    auto newResTensorType =
        RankedTensorType::get(shape, resElemTy, newPackedEncoding);
    Value convertedVal = rewriter.create<ttg::ConvertLayoutOp>(
        val.getLoc(), newResTensorType, val);

    Type ptrElemTy = cast<RankedTensorType>(ptr.getType()).getElementType();
    auto newPtrTensorType =
        RankedTensorType::get(shape, ptrElemTy, newPackedEncoding);
    Value convertedPtr = rewriter.create<ttg::ConvertLayoutOp>(
        ptr.getLoc(), newPtrTensorType, ptr);

    Type maskElemTy = cast<RankedTensorType>(mask.getType()).getElementType();
    auto newMaskTensorType =
        RankedTensorType::get(shape, maskElemTy, newPackedEncoding);
    Value convertedMask = rewriter.create<ttg::ConvertLayoutOp>(
        ptr.getLoc(), newMaskTensorType, mask);

    // Create new operation with required operand types
    auto newAtomicRmw = rewriter.create<tt::AtomicRMWOp>(
        atomOp.getLoc(), newResTensorType, atomicRmwBinOp, convertedPtr,
        convertedVal, convertedMask, atomOp.getSem(), atomOp.getScope());

    // Convert result back to the old layout
    Value convertedRes = rewriter.create<ttg::ConvertLayoutOp>(
        ptr.getLoc(), tensorTy, newAtomicRmw);

    rewriter.replaceOp(atomOp, convertedRes);
    return success();
  }
};
} // namespace

#define GEN_PASS_CLASSES
#include "TritonAMDGPUTransforms/Passes.h.inc"

class TritonAMDGPUOptimizeAtomicLayoutsPass
    : public TritonAMDGPUOptimizeAtomicLayoutsBase<
          TritonAMDGPUOptimizeAtomicLayoutsPass> {
public:
  TritonAMDGPUOptimizeAtomicLayoutsPass() = default;
  TritonAMDGPUOptimizeAtomicLayoutsPass(StringRef archGen) {
    this->archGenerationName = archGen.data();
  }
  void runOnOperation() override {

    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    RewritePatternSet patterns(context);
    if (supportedGlobalAtomicPacked(archGenerationName)) {
      patterns.add<::AtomicRmwPackF16>(context);
    }
    if (applyPatternsAndFoldGreedily(m, std::move(patterns)).failed()) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<Pass>
mlir::createTritonAMDGPUOptimizeAtomicLayoutsPass(std::string archGen) {
  return std::make_unique<TritonAMDGPUOptimizeAtomicLayoutsPass>(archGen);
}
