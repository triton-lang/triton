#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

#include <memory>

using namespace mlir;

static bool isSharedLayout(Value v) {
  if (auto tensorType = v.getType().dyn_cast<RankedTensorType>()) {
    Attribute encoding = tensorType.getEncoding();
    return encoding.isa<triton::gpu::SharedEncodingAttr>();
  }
  return false;
}

namespace {
#include "TritonGPUCombine.inc"

// -----------------------------------------------------------------------------
//
// -----------------------------------------------------------------------------

// Layout conversions can't deduce their return type automatically.
// IIUC they are therefore not handled by DRR right now
class SimplifyConversion : public mlir::RewritePattern {
public:
  SimplifyConversion(mlir::MLIRContext *context)
      : mlir::RewritePattern(triton::gpu::ConvertLayoutOp::getOperationName(),
                             2, context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!llvm::isa<triton::gpu::ConvertLayoutOp>(op))
      return mlir::failure();
    // convert to the same layout -- we can delete
    if (op->getResultTypes() == op->getOperandTypes()) {
      rewriter.replaceOp(op, op->getOperands());
      return mlir::success();
    }
    Operation *arg = op->getOperand(0).getDefiningOp();
    // block argument
    if (!arg)
      return mlir::failure();
    // cvt(type2, cvt(type1, x)) -> cvt(type2, x)
    if (llvm::isa<triton::gpu::ConvertLayoutOp>(arg)) {
      rewriter.replaceOpWithNewOp<triton::gpu::ConvertLayoutOp>(
          op, op->getResultTypes().front(), arg->getOperand(0));
      return mlir::success();
    }
    // cvt(type1, splat(type2, x)) -> splat(type1, x)
    if (auto splat = llvm::dyn_cast<triton::SplatOp>(arg)) {
      rewriter.replaceOpWithNewOp<triton::SplatOp>(op, op->getResultTypes(),
                                                   splat.src());
      return mlir::success();
    }
    // cvt(type1, make_range(type2, x)) -> make_range(type1, x)
    if (auto range = llvm::dyn_cast<triton::MakeRangeOp>(arg)) {
      rewriter.replaceOpWithNewOp<triton::MakeRangeOp>(
          op, op->getResultTypes(), range.start(), range.end());
      return mlir::success();
    }
    // cvt(type, constant) -> constant
    if (auto cst = llvm::dyn_cast<arith::ConstantOp>(arg))
      if (auto ret = cst.getValue().dyn_cast<SplatElementsAttr>()) {
        auto newRet = SplatElementsAttr::get(op->getResultTypes().front(),
                                             ret.getSplatValue<Attribute>());
        rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, newRet);
        return mlir::success();
      }
    return mlir::failure();
  }
};

// -----------------------------------------------------------------------------
//
// -----------------------------------------------------------------------------

// Layout conversions are expensive. They require going through
// shared memory, which is orders of magnitude slower than
// other non-i/o operations in the dialect.
// It therefore makes sense to remove them whenever possible,
// even if it means rematerializing all values whose definitions
// are reachable from it without passing through any memory operation.
class PullConversionToSource : public mlir::RewritePattern {
public:
  PullConversionToSource(mlir::MLIRContext *context)
      : mlir::RewritePattern(triton::gpu::ConvertLayoutOp::getOperationName(),
                             2, context) {}

  Attribute invertEncoding(Type targetType, Operation *op) const {
    RankedTensorType targetTensorType = targetType.cast<RankedTensorType>();
    if (auto expand_dims = dyn_cast<triton::ExpandDimsOp>(op)) {
      return targetTensorType.getEncoding()
          .cast<triton::gpu::BlockedEncodingAttr>()
          .squeeze(expand_dims.axis());
    }
    return targetTensorType.getEncoding();
  }

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *cvt,
                  mlir::PatternRewriter &rewriter) const override {
    if (!llvm::isa<triton::gpu::ConvertLayoutOp>(cvt))
      return mlir::failure();
    // constants/splat are handled separately
    Operation *op = cvt->getOperand(0).getDefiningOp();
    if (!op)
      return mlir::failure();

    auto blacklist = [](Operation *op) {
      if (isa<triton::gpu::ExtractSliceOp, triton::gpu::AllocTensorOp,
              triton::gpu::InsertSliceAsyncOp, triton::LoadOp, triton::StoreOp>(
              op))
        return true;
      if (isa<scf::YieldOp, scf::ForOp>(op))
        return true;
      return false;
    };

    // find all the ops that the conversion depends on
    SetVector<Operation *> depIntoOps;
    mlir::getBackwardSlice(cvt, &depIntoOps, [&](Operation *op) {
      return !blacklist(op) && op->getResult(0) &&
             (op->getResult(0).getParentBlock() ==
              cvt->getResult(0).getParentBlock());
    });
    // find all the ops that depend on something expensive
    // to rematerialize and break the dependency
    // chain there
    SetVector<Operation *> blacklistedOps;
    mlir::getBackwardSlice(cvt, &blacklistedOps, blacklist);
    for (Operation *op : blacklistedOps) {
      SetVector<Operation *> toRemove;
      mlir::getBackwardSlice(op, &toRemove);
      depIntoOps.set_subtract(toRemove);
    }
    // if there is nothing that can benefit from moving conversions
    // in the remaining op, we don't do anything
    auto it = llvm::find_if(depIntoOps, [&](Operation *op) {
      if (isa<triton::gpu::ConvertLayoutOp>(op)) {
        // conversions in for loops interfere with the
        // push-to-sink pass. Need a better cost model if how many conversions
        // we can actually remove by moving them to the beginning of the block
        auto forOp = dyn_cast<scf::ForOp>(cvt->getParentOp());
        if (!forOp &&
            (cvt->getResult(0).getType() == op->getOperand(0).getType()))
          return true;
      }
      if (isa<arith::ConstantOp, triton::MakeRangeOp, triton::SplatOp>(op))
        return true;
      return false;
    });
    if (it == depIntoOps.end()) {
      return mlir::failure();
    }

    // We convert cvt(op(arg_0, arg_1, ..., arg_n))
    // into op(cvt_0(arg_0), cvt_1(arg_1), ..., cvt_n(arg_n))
    BlockAndValueMapping mapping;
    for (Value argI : op->getOperands()) {
      // Compute new argument types
      auto oldArgType = argI.getType().dyn_cast<RankedTensorType>();
      if (!oldArgType)
        continue;
      auto newEncoding = invertEncoding(cvt->getResultTypes()[0], op);
      auto newArgType = RankedTensorType::get(
          oldArgType.getShape(), oldArgType.getElementType(), newEncoding);
      // Create new argument
      auto cvtI = rewriter.create<triton::gpu::ConvertLayoutOp>(
          op->getLoc(), newArgType, argI);
      cvtI->moveBefore(op);
      mapping.map(argI, cvtI);
    }
    Operation *newOp = rewriter.clone(*op, mapping);
    newOp->getResult(0).setType(cvt->getResult(0).getType());
    rewriter.replaceOp(cvt, newOp->getResults());

    return mlir::success();
  }
};

// -----------------------------------------------------------------------------
//
// -----------------------------------------------------------------------------

// This modifies the loop in-place
bool tryLegalizeOp(Operation *op, DenseSet<Value> toPreserve,
                   mlir::PatternRewriter &rewriter) {
  auto targetType = toPreserve.begin()->getType().cast<RankedTensorType>();
  auto newType = [&](RankedTensorType origType) {
    return RankedTensorType::get(origType.getShape(), origType.getElementType(),
                                 targetType.getEncoding());
  };
  bool hasSameTypes = op->getDialect()->getNamespace() == "arith" ||
                      isa<triton::SplatOp, triton::AddPtrOp>(op);
  if (hasSameTypes) {
    // replace argument types
    for (auto arg : llvm::enumerate(op->getOperands())) {
      auto argType = arg.value().getType().dyn_cast<RankedTensorType>();
      if (toPreserve.count(arg.value()) || !argType)
        continue;
      auto newArg = rewriter.create<triton::gpu::ConvertLayoutOp>(
          rewriter.getUnknownLoc(), newType(argType), arg.value());
      newArg->moveBefore(op);
      op->setOperand(arg.index(), newArg);
    }
    // replace result types
    if (!isa<triton::SplatOp>(op))
      op->getResult(0).setType(op->getOperand(0).getType());
    return true;
  }
  return false;
}

std::pair<SmallVector<Value, 4>, scf::ForOp>
tryConvertIterArg(scf::ForOp &forOp, mlir::PatternRewriter &rewriter, size_t i,
                  Type newType) {
  auto newEncoding = newType.cast<RankedTensorType>().getEncoding();
  auto ctx = forOp.getContext();
  auto isInLoop = [&](Operation *op) { return op->getParentOp() == forOp; };
  // Rewrite init argument
  Type origType = forOp.getInitArgs()[i].getType();
  SmallVector<Value, 4> newInitArgs = forOp.getInitArgs();
  newInitArgs[i] = rewriter.create<triton::gpu::ConvertLayoutOp>(
      newInitArgs[i].getLoc(), newType, newInitArgs[i]);
  // Clone for loop
  scf::ForOp newForOp = rewriter.create<scf::ForOp>(
      forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
      forOp.getStep(), newInitArgs);
  newForOp->moveBefore(forOp);
  rewriter.setInsertionPointToStart(newForOp.getBody());
  BlockAndValueMapping mapping;
  for (const auto &arg : llvm::enumerate(forOp.getRegionIterArgs()))
    mapping.map(arg.value(), newForOp.getRegionIterArgs()[arg.index()]);
  // traverse all ops in the loop
  for (Operation &op : forOp.getBody()->without_terminator()) {
    // we clone the op
    Operation *newOp = rewriter.clone(op, mapping);
    // if any argument of this op has changed type, then the
    // new operation is not legal and we should try to
    // legalize it.
    DenseSet<Value> modifiedTypes;
    for (Value arg : op.getOperands()) {
      if (mapping.contains(arg) &&
          mapping.lookup(arg).getType() != arg.getType())
        modifiedTypes.insert(mapping.lookup(arg));
    }

    bool shouldTryLegalize = !modifiedTypes.empty();
    if (shouldTryLegalize)
      tryLegalizeOp(newOp, modifiedTypes, rewriter);
  }
  // create yield, inserting conversions if necessary
  auto yieldOp = forOp.getBody()->getTerminator();
  SmallVector<Value, 4> newYieldArgs;
  for (Value arg : yieldOp->getOperands())
    newYieldArgs.push_back(mapping.lookup(arg));
  newYieldArgs[i] = rewriter.create<triton::gpu::ConvertLayoutOp>(
      yieldOp->getLoc(), newType, newYieldArgs[i]);
  rewriter.create<scf::YieldOp>(forOp.getLoc(), newYieldArgs);

  // replace
  SmallVector<Value, 4> newResults = newForOp->getResults();
  newResults[i] = rewriter.create<triton::gpu::ConvertLayoutOp>(
      rewriter.getUnknownLoc(), origType, newForOp->getResult(i));
  newResults[i].getDefiningOp()->moveAfter(newForOp);
  return {newResults, newForOp};
}

class MoveArgConvertOutOfLoop : public mlir::RewritePattern {
public:
  MoveArgConvertOutOfLoop(mlir::MLIRContext *context)
      : mlir::RewritePattern(scf::ForOp::getOperationName(), 1, context) {}

  mlir::LogicalResult matchAndRewrite(mlir::Operation *op,
                                      mlir::PatternRewriter &rewriter) const {

    auto forOp = cast<scf::ForOp>(op);
    auto isInLoop = [&](Operation *op) { return op->getParentOp() == forOp; };
    auto iterArgs = forOp.getRegionIterArgs();
    for (auto iterArg : llvm::enumerate(iterArgs)) {
      for (auto op : iterArg.value().getUsers()) {
        auto currOps = mlir::getSlice(op, isInLoop);
        auto pred = [&](Operation *op) {
          return isa<triton::LoadOp, triton::StoreOp>(op);
        };
        auto isCvt = [&](Operation *op) {
          return isa<triton::gpu::ConvertLayoutOp>(op);
        };
        auto isYield = [&](Operation *op) { return isa<scf::YieldOp>(op); };
        auto opIt = std::find(currOps.begin(), currOps.end(), op);
        auto yieldIt = std::find_if(currOps.begin(), currOps.end(), isYield);
        auto fwdEndIt = std::find_if(opIt, currOps.end(), pred);
        auto bwdBeginIt = std::find_if(currOps.begin(), opIt, pred);
        auto fwdCvtIt = std::find_if(opIt, fwdEndIt, isCvt);
        auto bwdCvtIt = std::find_if(bwdBeginIt, opIt, isCvt);

        if (fwdCvtIt != fwdEndIt) {
          auto newFor = tryConvertIterArg(forOp, rewriter, iterArg.index(),
                                          (*fwdCvtIt)->getResult(0).getType());
          rewriter.replaceOp(forOp, newFor.first);
          return success();
        }
      }
    }
    return failure();
  }
};

// -----------------------------------------------------------------------------
//
// -----------------------------------------------------------------------------

class PushConversionToSink : public mlir::RewritePattern {
public:
  PushConversionToSink(mlir::MLIRContext *context)
      : mlir::RewritePattern(triton::gpu::ConvertLayoutOp::getOperationName(),
                             2, context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *_cvtOp,
                  mlir::PatternRewriter &rewriter) const override {
    auto cvt = cast<triton::gpu::ConvertLayoutOp>(_cvtOp);
    auto forOp = dyn_cast<scf::ForOp>(cvt->getParentOp());
    if (!forOp)
      return mlir::failure();
    auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    auto isInLoop = [&](Operation *op) { return op->getParentOp() == forOp; };

    SetVector<Operation *> cvtSlices;
    auto filter = [&](Operation *op) {
      return isInLoop(op) && !isa<triton::LoadOp>(op) &&
             !isa<triton::DotOp>(op) && !isa<scf::YieldOp>(op) &&
             !isa<triton::gpu::ConvertLayoutOp>(op);
    };
    mlir::getForwardSlice(cvt.getResult(), &cvtSlices, filter);
    if (cvtSlices.empty())
      return failure();
    // if other operands are in the loop
    // then we don't touch anything
    Operation *op = cvtSlices.front();
    for (Value _arg : op->getOperands()) {
      Operation *arg = _arg.getDefiningOp();
      if (arg && isInLoop(arg) && (arg != cvt))
        return failure();
    }
    // otherwise, we push the conversion forward
    // since we'll be able to move it out of
    // the loop once it reaches the yield op
    // op(cvt(arg_0), arg_1, ..., arg_n)
    // -> cvt(op(arg_0, cvt(arg_1), ..., cvt(arg_n)))
    BlockAndValueMapping mapping;
    for (Value arg : op->getOperands()) {
      if (arg.getDefiningOp() == cvt)
        mapping.map(arg, cvt.getOperand());
      else {
        auto cvtI = rewriter.create<triton::gpu::ConvertLayoutOp>(
            arg.getLoc(), cvt.getOperand().getType(), arg);
        mapping.map(arg, cvtI);
      }
    }
    Operation *newOp = rewriter.clone(*op, mapping);
    newOp->getResult(0).setType(cvt.getOperand().getType());
    auto newCvt = rewriter.create<triton::gpu::ConvertLayoutOp>(
        newOp->getLoc(), cvt.getResult().getType(), newOp->getResult(0));
    rewriter.replaceOp(op, newCvt->getResults());
    return success();
  }
};

// -----------------------------------------------------------------------------
//
// -----------------------------------------------------------------------------

class BlockedToMMA : public mlir::RewritePattern {
public:
  BlockedToMMA(mlir::MLIRContext *context)
      : mlir::RewritePattern(triton::DotOp::getOperationName(), 2, context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto dotOp = cast<triton::DotOp>(op);
    // TODO: Check data-types and SM compatibility
    auto oldRetType = dotOp.getResult().getType().cast<RankedTensorType>();
    if (oldRetType.getEncoding().isa<triton::gpu::MmaEncodingAttr>())
      return failure();
    // TODO: compute warpsPerCTA
    auto newRetType = RankedTensorType::get(
        oldRetType.getShape(), oldRetType.getElementType(),
        triton::gpu::MmaEncodingAttr::get(oldRetType.getContext(), 2, {2, 2}));
    auto oldAcc = dotOp.getOperand(2);
    auto newAcc = rewriter.create<triton::gpu::ConvertLayoutOp>(
        oldAcc.getLoc(), newRetType, oldAcc);
    auto newDot = rewriter.create<triton::DotOp>(
        dotOp.getLoc(), newRetType, dotOp.getOperand(0), dotOp.getOperand(1),
        newAcc, dotOp.allowTF32());

    rewriter.replaceOpWithNewOp<triton::gpu::ConvertLayoutOp>(
        op, oldRetType, newDot.getResult());
    return success();
  }
};

} // namespace

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

class TritonGPUCombineOpsPass
    : public TritonGPUCombineOpsBase<TritonGPUCombineOpsPass> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    mlir::RewritePatternSet patterns(context);

    patterns.add<SimplifyConversion>(context);
    patterns.add<PullConversionToSource>(context);
    patterns.add<PushConversionToSink>(context);
    patterns.add<MoveArgConvertOutOfLoop>(context);
    patterns.add<BlockedToMMA>(context);

    if (applyPatternsAndFoldGreedily(m, std::move(patterns)).failed())
      signalPassFailure();
  }
};

std::unique_ptr<Pass> mlir::createTritonGPUCombineOpsPass() {
  return std::make_unique<TritonGPUCombineOpsPass>();
}
