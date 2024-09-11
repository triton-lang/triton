#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

// This pass hoists auxiliar operations over dot outside the K loop
// Example of optimized loop:
//
//   void kernel() {
//     x = tensor<8, 1, 64>
//     y = tensor<8, 64, 32>
//     acc = tensor<1, 32>
//     for (iter = 0; iter < 3; ++iter) {
//       acc = reshape(acc, <1, 1, 32>)
//       acc = broadcast(acc, <8, 1, 32>)
//       acc = dot(x, y, acc)
//       acc = reduce_sum(acc, axis = 0)
//     }
//     store(acc)
//   }
//
// Transforms to:
//
//   void kernel() {
//     x = tensor<8, 1, 64>
//     y = tensor<8, 64, 32>
//     acc = tensor<1, 32>
//     acc = reshape(acc, <1, 1, 32>)
//     acc = broadcast(acc, <8, 1, 32>)
//     for (iter = 0; iter < 3; ++iter) {
//       acc = dot(x, y, acc)
//     }
//     acc = reduce_sum(acc, axis = 0)
//     store(acc)
//   }

using namespace mlir;

namespace {

bool isPermutableOp(const mlir::Operation &op) {
  return isa<mlir::arith::AddIOp>(op) || isa<mlir::arith::AddFOp>(op);
}

bool isApproprateReduction(triton::ReduceOp op) {
  if (op.getAxis() != 0)
    return false;
  auto redBody = op.getBody();
  if (redBody->getNumSuccessors() != 0)
    return false;
  auto &redBodyOps = redBody->getOperations();
  if (redBodyOps.size() != 2)
    return false;
  auto &redBaseOp = redBodyOps.front();
  auto returnOp = cast<triton::ReduceReturnOp>(redBodyOps.back());
  if (redBaseOp.getNumResults() != 1)
    return false;
  if (!isPermutableOp(redBaseOp) &&
      returnOp.getOperand(0) != redBaseOp.getResult(0))
    return false;
  return true;
}

struct dotReductionChainOperations {
  BlockArgument loopBlockArg;
  triton::ReshapeOp reshape;
  triton::BroadcastOp broadcast;
  triton::gpu::ConvertLayoutOp preDotConvert;
  arith::ConstantOp accZeroInitializer;
  triton::DotOp dot;
  triton::ReduceOp reduction;
  triton::gpu::ConvertLayoutOp postDotConversion;
  Operation *accumulatorAdd;
  scf::YieldOp yield;
  int yieldOpNo;
};

OpOperand *getSingleUse(Operation *op) {
  if (!op->getResult(0).hasOneUse())
    return nullptr;
  return &(*op->getUses().begin());
}

std::optional<dotReductionChainOperations>
matchDotReductionInLoopPattern(triton::ReduceOp redOp) {
  dotReductionChainOperations chain;
  chain.reduction = redOp;
  auto postDotConversionOperand = getSingleUse(redOp);
  if (!postDotConversionOperand)
    return std::nullopt;
  chain.postDotConversion = dyn_cast<triton::gpu::ConvertLayoutOp>(
      postDotConversionOperand->getOwner());
  if (!chain.postDotConversion)
    return std::nullopt;
  auto dotResultUse = getSingleUse(chain.postDotConversion);
  if (!dotResultUse)
    return std::nullopt;
  if (isa<arith::AddFOp, arith::AddIOp>(dotResultUse->getOwner())) {
    chain.accumulatorAdd = dotResultUse->getOwner();
    int inductionOpNo = dotResultUse->getOperandNumber() == 0 ? 1 : 0;
    auto loopArgument = dyn_cast<BlockArgument>(
        chain.accumulatorAdd->getOperand(inductionOpNo));
    chain.loopBlockArg = loopArgument;
    dotResultUse = getSingleUse(chain.accumulatorAdd);
    if (!dotResultUse)
      return std::nullopt;
  }
  chain.yieldOpNo = dotResultUse->getOperandNumber();
  chain.yield = dyn_cast<scf::YieldOp>(dotResultUse->getOwner());
  if (!chain.yield)
    return std::nullopt;

  chain.dot = dyn_cast<triton::DotOp>(redOp->getOperand(0).getDefiningOp());
  if (!chain.dot)
    return std::nullopt;
  if (!chain.accumulatorAdd) {
    chain.preDotConvert = dyn_cast<triton::gpu::ConvertLayoutOp>(
        chain.dot.getC().getDefiningOp());
    if (!chain.preDotConvert)
      return std::nullopt;
    chain.broadcast = dyn_cast<triton::BroadcastOp>(
        chain.preDotConvert.getSrc().getDefiningOp());
    if (!chain.broadcast)
      return std::nullopt;
    chain.reshape =
        dyn_cast<triton::ReshapeOp>(chain.broadcast.getSrc().getDefiningOp());
    if (!chain.reshape)
      return std::nullopt;
    auto loopArgument = dyn_cast<BlockArgument>(chain.reshape.getSrc());
    if (!loopArgument || loopArgument.getArgNumber() != chain.yieldOpNo + 1)
      return std::nullopt;
    chain.loopBlockArg = loopArgument;
  } else {
    chain.accZeroInitializer =
        dyn_cast<mlir::arith::ConstantOp>(chain.dot.getC().getDefiningOp());
    if (!chain.accZeroInitializer)
      return std::nullopt;
  }
  if (chain.loopBlockArg.getArgNumber() != chain.yieldOpNo + 1)
    return std::nullopt;
  return chain;
}

void hoistReductionOps(mlir::PatternRewriter &rewriter, scf::ForOp loopOp,
                       dotReductionChainOperations dfChain) {
  auto accInitializer = loopOp.getInitArgs()[dfChain.yieldOpNo];

  if (!dfChain.accumulatorAdd) {
    // hoist operations outside the loop
    rewriter.moveOpBefore(dfChain.reshape, loopOp);
    rewriter.moveOpBefore(dfChain.broadcast, loopOp);
    rewriter.moveOpBefore(dfChain.preDotConvert, loopOp);

    rewriter.moveOpAfter(dfChain.reduction, loopOp);
    rewriter.moveOpAfter(dfChain.postDotConversion, dfChain.reduction);

    // adjust operations DF
    dfChain.reshape.setOperand(accInitializer);
    loopOp.setOperand(dfChain.yieldOpNo + 3, dfChain.preDotConvert);
    dfChain.dot.setOperand(2, dfChain.loopBlockArg);
    dfChain.yield.setOperand(dfChain.yieldOpNo, dfChain.dot);

    rewriter.replaceAllUsesWith(loopOp.getResult(dfChain.yieldOpNo),
                                dfChain.postDotConversion);
    dfChain.reduction.setOperand(0, loopOp.getResult(dfChain.yieldOpNo));

    // adjust loop types
    auto newAccTy = dfChain.preDotConvert.getType();
    // loopOp.getOperand(dfChain.yieldOpNo + 3).setType(newAccTy);
    loopOp.getResult(dfChain.yieldOpNo).setType(newAccTy);
    dfChain.loopBlockArg.setType(newAccTy);
  } else {
    // hoist operations outside the loop
    rewriter.moveOpBefore(dfChain.accZeroInitializer, loopOp);

    rewriter.moveOpAfter(dfChain.reduction, loopOp);
    rewriter.moveOpAfter(dfChain.postDotConversion, dfChain.reduction);
    rewriter.moveOpAfter(dfChain.accumulatorAdd, dfChain.postDotConversion);

    // adjust operations DF
    dfChain.accumulatorAdd->setOperand(0, dfChain.postDotConversion);
    dfChain.accumulatorAdd->setOperand(1, accInitializer);
    loopOp.setOperand(dfChain.yieldOpNo + 3, dfChain.accZeroInitializer);
    dfChain.dot.setOperand(2, dfChain.loopBlockArg);
    dfChain.yield.setOperand(dfChain.yieldOpNo, dfChain.dot);

    rewriter.replaceAllUsesWith(loopOp.getResult(dfChain.yieldOpNo),
                                dfChain.accumulatorAdd->getResult(0));
    dfChain.reduction.setOperand(0, loopOp.getResult(dfChain.yieldOpNo));

    // adjust loop types
    auto newAccTy = dfChain.accZeroInitializer.getType();
    // loopOp.getOperand(dfChain.yieldOpNo + 3).setType(newAccTy);
    loopOp.getResult(dfChain.yieldOpNo).setType(newAccTy);
    dfChain.loopBlockArg.setType(newAccTy);
  }
}

class HoistReduction : public mlir::RewritePattern {

public:
  explicit HoistReduction(mlir::MLIRContext *context)
      : mlir::RewritePattern(triton::ReduceOp::getOperationName(), 1, context) {
  }

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto redOp = llvm::cast<triton::ReduceOp>(op);
    if (!isApproprateReduction(redOp))
      return failure();

    auto loopOp = dyn_cast<scf::ForOp>(redOp->getParentOp());
    if (!loopOp)
      return failure();

    auto matchedDotReduction = matchDotReductionInLoopPattern(redOp);
    if (!matchedDotReduction.has_value())
      return failure();

    hoistReductionOps(rewriter, loopOp, matchedDotReduction.value());

    return mlir::success();
  }
};

} // namespace

#define GEN_PASS_CLASSES
#include "TritonAMDGPUTransforms/Passes.h.inc"

class TritonAMDGPUHoistReductionPass
    : public TritonAMDGPUHoistReductionBase<TritonAMDGPUHoistReductionPass> {

public:
  TritonAMDGPUHoistReductionPass() = default;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    mlir::RewritePatternSet patterns(context);

    patterns.add<HoistReduction>(context);

    if (applyPatternsAndFoldGreedily(m, std::move(patterns)).failed()) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<Pass> mlir::createTritonAMDGPUHoistReductionPass() {
  return std::make_unique<TritonAMDGPUHoistReductionPass>();
}
