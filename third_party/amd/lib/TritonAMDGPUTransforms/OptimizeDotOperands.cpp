#include "TritonAMDGPUToLLVM/TargetUtils.h"
#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LayoutUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/LinearLayout.h"
#include <memory>

#define DEBUG_TYPE "tritonamdgpu-optimize-dot-operands"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace mlir::triton::amdgpu {

namespace {

// Detect a pair of tt.dot ops that both use the same tt.load result, one
// directly and one via tt.trans and use the same shared memory buffer in this
// case. Given:
//   load -> cvt -> .. -> dot1
//        -> cvt -> .. -> trans -> cvt -> .. -> dot2
// Rewrite to:
//  load -> local_alloc -> local_load            -> dot1
//                      -> local_load_transposed -> dot2
class ReuseShmemForDirectAndTransposedUse : public OpRewritePattern<LoadOp> {
public:
  ReuseShmemForDirectAndTransposedUse(MLIRContext *context,
                                      triton::AMD::ISAFamily isaFamily)
      : OpRewritePattern(context), isaFamily(isaFamily) {}

  LogicalResult matchAndRewrite(tt::LoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    auto numUsers = llvm::range_size(loadOp->getUsers());
    if (numUsers < 2) {
      return rewriter.notifyMatchFailure(loadOp,
                                         "load op must have at least 2 users");
    }

    auto srcTy = dyn_cast<RankedTensorType>(loadOp.getType());
    if (!srcTy) {
      return rewriter.notifyMatchFailure(loadOp, "src type must be a tensor");
    }

    LDBG("ReuseShmemForDirectAndTransposedUse for load Op: " << *loadOp);

    tt::DotOpInterface directDot = nullptr;
    tt::DotOpInterface transDot = nullptr;
    ttg::ConvertLayoutOp cvtOp = nullptr;
    unsigned directOpIdx = 0;
    unsigned transOpIdx = 0;

    auto followConvertLayoutChain =
        [](mlir::Value &usedValue, mlir::Operation *op) -> mlir::Operation * {
      while (isa<ttg::ConvertLayoutOp>(op)) {
        // Ensure we have exactly one user
        if (!(op->hasOneUse()))
          return nullptr;
        usedValue = op->getResult(0);
        op = *(op->getUsers().begin());
      }

      return op;
    };

    mlir::Value usedValue;
    for (mlir::Operation *user : loadOp->getUsers()) {
      auto op = user;

      op = followConvertLayoutChain(usedValue, op);

      if (auto transOp = dyn_cast_or_null<tt::TransOp>(op)) {
        LDBG("Found tranpose op: " << *transOp);
        cvtOp = transOp.getSrc().getDefiningOp<ttg::ConvertLayoutOp>();
        LDBG("Found parent cvt op of transpose: " << *cvtOp);
        usedValue = transOp->getResult(0);
        op =
            followConvertLayoutChain(usedValue, *(transOp->getUsers().begin()));
        if (auto dotOp = dyn_cast<tt::DotOpInterface>(op)) {
          transDot = dotOp;
          transOpIdx = (usedValue == dotOp.getA()) ? 0 : 1;
        }
      } else if (auto dotOp = dyn_cast_or_null<tt::DotOpInterface>(op)) {
        directDot = dotOp;
        directOpIdx = (usedValue == dotOp.getA()) ? 0 : 1;
      }

      if (directDot && transDot)
        break;
    }

    if (!directDot)
      return rewriter.notifyMatchFailure(loadOp,
                                         "expected a direct tt.dot user");
    if (!transDot)
      return rewriter.notifyMatchFailure(
          loadOp, "expected a tt.trans feeding a tt.dot user");
    if (directOpIdx != transOpIdx) {
      return rewriter.notifyMatchFailure(loadOp, [&](mlir::Diagnostic &d) {
        d << "operand indices of direct and transposed tt.dot users must be "
             "the same. Got indices: direct: "
          << directOpIdx << " and transposed: " << transOpIdx;
      });
    }

    LDBG("load is shared between transposed and non-transposed users");
    LDBG("Non-transposed access tt.dot: " << *directDot);
    LDBG("Transposed access tt.dot: " << *transDot);

    unsigned opIdx = directOpIdx;

    auto directOperandType =
        cast<RankedTensorType>(directDot->getOperand(opIdx).getType());
    auto transOperandType =
        cast<RankedTensorType>(transDot->getOperand(opIdx).getType());
    auto directDotEnc =
        dyn_cast<ttg::DotOperandEncodingAttr>(directOperandType.getEncoding());
    auto transDotEnc =
        dyn_cast<ttg::DotOperandEncodingAttr>(transOperandType.getEncoding());

    if (!directDotEnc || !transDotEnc) {
      return rewriter.notifyMatchFailure(loadOp,
                                         "wrong encodings for tt.dot users");
    }

    if (directDotEnc.getKWidth() != transDotEnc.getKWidth()) {
      return rewriter.notifyMatchFailure(loadOp, [&](mlir::Diagnostic &d) {
        d << "kWidths are mismatching. direct: " << directDotEnc.getKWidth()
          << " and transposed: " << transDotEnc.getKWidth();
      });
    }

    // We need to ensure that the parents of direct and transposed dot encodings
    // are matching in order to get the same shared memory encoding. Note that
    // they can have different instrShape(s) (mfma instructions) but still map
    // to the same shared memory encoding.
    auto directCTALayout = ttg::getCTALayout(directDotEnc);
    auto transCTALayout = ttg::getCTALayout(transDotEnc);

    if (directCTALayout != transCTALayout) {
      return rewriter.notifyMatchFailure(
          loadOp,
          "CTA layouts of direct and transposed tt.dot users are mismatching");
    }

    auto ctx = getContext();
    auto sharedOrder = ttg::getOrderForMemory(srcTy);
    auto sharedEnc = ttg::SwizzledSharedEncodingAttr::get(
        ctx, directDotEnc, directOperandType.getShape(), sharedOrder,
        directCTALayout, directOperandType.getElementType(),
        /*needTrans=*/false);

    LDBG("Created shared encoding: " << sharedEnc);
    rewriter.setInsertionPointAfter(loadOp);
    auto sharedMemorySpace = ttg::SharedMemorySpaceAttr::get(ctx);
    Location loc = loadOp.getLoc();
    auto alloc = rewriter.create<ttg::LocalAllocOp>(
        loc,
        ttg::MemDescType::get(srcTy.getShape(), srcTy.getElementType(),
                              sharedEnc, sharedMemorySpace),
        loadOp.getResult());
    LDBG("Created local alloc op: " << *alloc);
    auto localLoad =
        rewriter.create<ttg::LocalLoadOp>(loc, directOperandType, alloc);
    LDBG("Created local load op:" << *localLoad);
    rewriter.modifyOpInPlace(
        directDot, [&]() { directDot->setOperand(opIdx, localLoad); });
    LDBG("Updated Direct dot: " << *directDot);
    if (!canUseLocalLoadTransposed(opIdx, sharedOrder)) {
      rewriter.modifyOpInPlace(cvtOp, [&]() {
        cvtOp.getSrcMutable().assign(localLoad.getResult());
      });
      LDBG("Updated cvt op: " << *cvtOp);
    } else {
      return rewriter.notifyMatchFailure(loadOp, "currently not supported");
    }

    LDBG("Updated Trans dot: " << *transDot);

    return success();
  }

private:
  bool canUseLocalLoadTransposed(unsigned opIdx,
                                 ArrayRef<unsigned> sharedOrder) const {
    // TODO(PMylon): Comment out for now, until lowering from
    // local_load_transposed to ds_read_tr is supported.
    // unsigned kDimIdx = (opIdx == 0) ? 1 : 0;
    // bool isCDNA4 = (isaFamily == triton::AMD::ISAFamily::CDNA4);
    // bool isKContig = (sharedOrder[0] == kDimIdx);
    return false;
  }

  triton::AMD::ISAFamily isaFamily;
};

} // namespace

#define GEN_PASS_DEF_TRITONAMDGPUOPTIMIZEDOTOPERANDS
#include "TritonAMDGPUTransforms/Passes.h.inc"

class TritonAMDGPUOptimizeDotOperands
    : public impl::TritonAMDGPUOptimizeDotOperandsBase<
          TritonAMDGPUOptimizeDotOperands> {
public:
  using Base::Base;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    mlir::RewritePatternSet patterns(context);
    auto isaFamily = triton::AMD::deduceISAFamily(archGenerationName);
    patterns.add<ReuseShmemForDirectAndTransposedUse>(context, isaFamily);
    ttg::ConvertLayoutOp::getCanonicalizationPatterns(patterns, context);
    if (failed(applyPatternsGreedily(m, std::move(patterns))))
      signalPassFailure();
  }
};

void registerTritonAMDGPUOptimizeDotOperands() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createTritonAMDGPUOptimizeDotOperands();
  });
}

} // namespace mlir::triton::amdgpu
