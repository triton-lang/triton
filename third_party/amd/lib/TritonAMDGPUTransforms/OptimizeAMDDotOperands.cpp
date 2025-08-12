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
//  load -> local_alloc -> local_load           -> dot1
//                      -> local_load_transposed -> dot2
class ReuseShmemForDirectAndTransposedUse : public OpRewritePattern<LoadOp> {
public:
  using OpRewritePattern::OpRewritePattern;

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

    tt::DotOp directDot = nullptr;
    tt::DotOp transDot = nullptr;
    unsigned directOpIdx = 0;
    unsigned transOpIdx = 0;
    ArrayRef<int> transOrder;

    auto followConvertLayoutChain =
        [](mlir::Value &usedValue, mlir::Operation *op) -> mlir::Operation * {
      while (isa<ttg::ConvertLayoutOp>(op)) {
        if (op->getUsers().empty())
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

      if (auto transOp = dyn_cast<tt::TransOp>(op)) {
        LDBG("Found tranpose op: " << *transOp);
        transOrder = transOp.getOrder();
        usedValue = transOp->getResult(0);
        op =
            followConvertLayoutChain(usedValue, *(transOp->getUsers().begin()));
        if (auto dotOp = dyn_cast<tt::DotOp>(op)) {
          transDot = dotOp;
          transOpIdx = (usedValue == dotOp.getA()) ? 0 : 1;
        }
      } else if (auto dotOp = dyn_cast<tt::DotOp>(op)) {
        directDot = dotOp;
        directOpIdx = (usedValue == dotOp.getA()) ? 0 : 1;
      }
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
        cast<RankedTensorType>(directDot.getOperand(opIdx).getType());
    auto transOperandType =
        cast<RankedTensorType>(transDot.getOperand(opIdx).getType());
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

    auto ctx = getContext();
    auto sharedEnc = ttg::SwizzledSharedEncodingAttr::get(
        ctx, directDotEnc, directOperandType.getShape(),
        /*order=*/ttg::getOrderForMemory(srcTy),
        ttg::getCTALayout(directDotEnc), directOperandType.getElementType(),
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
    directDot.setOperand(opIdx, localLoad);
    LDBG("Updated Direct dot: " << *directDot);
    auto transposedLocalLoad =
        rewriter.create<triton::amdgpu::LocalLoadTransposedOp>(
            loc, transOperandType, alloc);
    LDBG("Created transposed local load op:" << *transposedLocalLoad);
    transDot.setOperand(opIdx, transposedLocalLoad);
    LDBG("Updated Trans dot: " << *transDot);

    return success();
  }
};

} // namespace

#define GEN_PASS_DEF_TRITONAMDGPUOPTIMIZEDOTOPERANDS
#include "TritonAMDGPUTransforms/Passes.h.inc"

class TritonAMDGPUOptimizeDotOperands
    : public impl::TritonAMDGPUOptimizeDotOperandsBase<
          TritonAMDGPUOptimizeDotOperands> {
public:
  using impl::TritonAMDGPUOptimizeDotOperandsBase<
      TritonAMDGPUOptimizeDotOperands>::TritonAMDGPUOptimizeDotOperandsBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    OpPassManager pm;
    pm.addPass(mlir::createCanonicalizerPass());
    if (failed(runPipeline(pm, m)))
      return signalPassFailure();

    mlir::RewritePatternSet patterns(context);
    patterns.add<ReuseShmemForDirectAndTransposedUse>(context);
    ttg::ConvertLayoutOp::getCanonicalizationPatterns(patterns, context);
    if (failed(applyPatternsGreedily(m, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace mlir::triton::amdgpu
