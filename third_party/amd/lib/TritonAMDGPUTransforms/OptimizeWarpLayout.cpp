#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "TritonAMDGPUToLLVM/TargetUtils.h"
#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tritonamdgpu-warp-layout"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttag = mlir::triton::amdgpu;

namespace mlir {

#define GEN_PASS_DEF_TRITONAMDGPUOPTIMIZEWARPLAYOUT
#include "TritonAMDGPUTransforms/Passes.h.inc"

namespace {

ttg::BlockedEncodingAttr getNewBlockedEnc(RankedTensorType loadType) {
  // get the current blocked encoding
  auto loadEnc = loadType.getEncoding();
  auto shape = loadType.getShape();
  auto blockedEnc = cast<ttg::BlockedEncodingAttr>(loadEnc);
  auto sizePerThread = blockedEnc.getSizePerThread();
  auto order = blockedEnc.getOrder();
  auto numWarps = product(blockedEnc.getWarpsPerCTA());
  auto threadsPerWarp = blockedEnc.getThreadsPerWarp();
  auto CTALayout = blockedEnc.getCTALayout();

  unsigned rank = sizePerThread.size();
  SmallVector<unsigned, 4> warpsPerCTA(rank);
  SmallVector<int64_t> shapePerCTA =
      triton::gpu::getShapePerCTA(CTALayout.getCTASplitNum(), shape);
  unsigned remainingWarps = numWarps;

  // starting from the row dimension
  unsigned rowDim = 0;
  int64_t rowValue = shapePerCTA[rowDim];
  int64_t rowNumWarps = rowValue;
  // e.g. rowValue: 3 remainingWarps: 8, at least one warp is wasted.
  if ((rowValue <= remainingWarps) && (remainingWarps % rowValue != 0)) {
    return blockedEnc;
  }

  // e.g. rowValue:10, remainingWarps:8, the better row warps is 2.
  if (rowValue > remainingWarps) {
    rowNumWarps = rowValue % remainingWarps;
    // e.g. rowValue:16, remainingWarps:8, the better row warps is 8.
    rowNumWarps = rowNumWarps == 0 ? remainingWarps : rowNumWarps;

    // The new calculated value is less than the original value.
    if (rowNumWarps < blockedEnc.getWarpsPerCTA()[rowDim]) {
      return blockedEnc;
    }
  }

  warpsPerCTA[rowDim] =
      std::clamp<unsigned>(rowNumWarps, 1, std::max<unsigned>(1, rowValue));
  remainingWarps /= warpsPerCTA[rowDim];

  unsigned colDim = 1;
  warpsPerCTA[colDim] = remainingWarps;

  return ttg::BlockedEncodingAttr::get(blockedEnc.getContext(), sizePerThread,
                                       threadsPerWarp, warpsPerCTA, order,
                                       CTALayout);
}

static Type replaceEncoding(Type type, Attribute encoding) {
  RankedTensorType tensorType = cast<RankedTensorType>(type);
  return RankedTensorType::get(tensorType.getShape(),
                               tensorType.getElementType(), encoding);
}

class Reduce2DWarpLayoutPattern : public OpRewritePattern<tt::ReduceOp> {
public:
  Reduce2DWarpLayoutPattern(MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit) {}

  LogicalResult matchAndRewrite(tt::ReduceOp reduceOp,
                                PatternRewriter &rewriter) const override {
    LDBG("Consider " << reduceOp);
    auto refTensorType =
        cast<RankedTensorType>(reduceOp->getOperands()[0].getType());

    if (!refTensorType || refTensorType.getRank() != 2) {
      LDBG("Only optimize the scenario where the input of the reduce operation "
           "is 2D.");
      return failure();
    }

    auto blockedEnc =
        dyn_cast<ttg::BlockedEncodingAttr>(refTensorType.getEncoding());
    if (!blockedEnc) {
      return failure();
    }

    if (blockedEnc.getOrder() != ArrayRef<unsigned>({1, 0})) {
      return failure();
    }

    auto currentEnc = refTensorType.getEncoding();
    auto newBlockedEnc = getNewBlockedEnc(refTensorType);
    if (currentEnc && currentEnc == newBlockedEnc) {
      return failure();
    }

    auto loc = reduceOp->getLoc();
    rewriter.setInsertionPoint(reduceOp);
    // Convert operands
    SmallVector<Value, 4> newArgs;
    for (auto operand : reduceOp->getOperands()) {
      auto tensorType = dyn_cast<RankedTensorType>(operand.getType());
      if (tensorType) {
        Type newType = replaceEncoding(tensorType, newBlockedEnc);
        newArgs.push_back(
            rewriter.create<ttg::ConvertLayoutOp>(loc, newType, operand));
      } else {
        newArgs.push_back(operand);
      }
    }

    // Construct new reduce with the new encoding
    auto attrs = reduceOp->getAttrs();
    auto newReduce = rewriter.create<tt::ReduceOp>(loc, newArgs, attrs);
    rewriter.inlineRegionBefore(reduceOp.getRegion(), newReduce.getRegion(),
                                newReduce.getRegion().begin());

    // Convert output to original encoding
    unsigned numResults = reduceOp->getNumResults();
    SmallVector<Value> finalOutputs;

    for (unsigned i = 0; i < numResults; ++i) {
      auto originalOutputType = reduceOp->getResult(i).getType();
      auto newOutput = newReduce->getResult(i);

      if (newOutput.getType() != originalOutputType) {
        auto convertOp = rewriter.create<ttg::ConvertLayoutOp>(
            loc, originalOutputType, newOutput);
        finalOutputs.push_back(convertOp->getResult(0));
      } else {
        finalOutputs.push_back(newOutput);
      }
    }

    rewriter.replaceOp(reduceOp, finalOutputs);
    return success();
  }
};

} // anonymous namespace

class TritonAMDGPUOptimizeWarpLayoutPass
    : public impl::TritonAMDGPUOptimizeWarpLayoutBase<
          TritonAMDGPUOptimizeWarpLayoutPass> {

public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    mlir::RewritePatternSet patterns(context);
    patterns.add<Reduce2DWarpLayoutPattern>(context);
    ttg::ConvertLayoutOp::getCanonicalizationPatterns(patterns, context);
    if (failed(applyPatternsGreedily(m, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace mlir
