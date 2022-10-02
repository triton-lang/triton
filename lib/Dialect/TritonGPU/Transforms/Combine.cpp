#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
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
                             4, context) {}

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
      return triton::gpu::SliceEncodingAttr::get(
          getContext(), expand_dims.axis(), targetTensorType.getEncoding());
    }
    return targetTensorType.getEncoding();
  }

  void rematerializeCvt(mlir::Operation *cvt, mlir::PatternRewriter &rewriter,
                        SetVector<Operation *> &toRemat,
                        SetVector<Operation *> &seen,
                        BlockAndValueMapping &mapping) const {
    if (seen.contains(cvt))
      return;
    seen.insert(cvt);
    rewriter.startRootUpdate(cvt);
    Operation *op = cvt->getOperand(0).getDefiningOp();
    SmallVector<triton::gpu::ConvertLayoutOp, 4> cvts;
    for (Value argI : op->getOperands()) {
      // Compute new argument types
      auto oldArgType = argI.getType().dyn_cast<RankedTensorType>();
      if (!oldArgType)
        continue;
      if (mapping.contains(argI))
        continue;
      auto newEncoding = invertEncoding(cvt->getResultTypes()[0], op);
      auto newArgType = RankedTensorType::get(
          oldArgType.getShape(), oldArgType.getElementType(), newEncoding);
      // Create new argument
      auto cvtI = rewriter.create<triton::gpu::ConvertLayoutOp>(
          op->getLoc(), newArgType, argI);
      cvtI->moveAfter(argI.getDefiningOp());
      if (toRemat.contains(argI.getDefiningOp()))
        cvts.push_back(cvtI);
      mapping.map(argI, cvtI);
    }
    Operation *newOp = rewriter.clone(*op, mapping);
    newOp->moveBefore(op);
    newOp->getResult(0).setType(cvt->getResult(0).getType());
    cvt->replaceAllUsesWith(newOp->getResults());
    for (auto cvtI : cvts)
      rematerializeCvt(cvtI, rewriter, toRemat, seen, mapping);
    rewriter.finalizeRootUpdate(cvt);
    rewriter.eraseOp(cvt);
  }

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *cvt,
                  mlir::PatternRewriter &rewriter) const override {
    if (!llvm::isa<triton::gpu::ConvertLayoutOp>(cvt))
      return mlir::failure();
    // we don't touch block arguments
    Operation *op = cvt->getOperand(0).getDefiningOp();
    if (!op)
      return mlir::failure();
    // we don't want to rematerialize any conversion to/from shared
    if (isSharedLayout(cvt->getResults()[0]) ||
        isSharedLayout(cvt->getOperand(0)))
      return mlir::failure();
    // determine whether layout conversions can be folded into a given operation
    // llvm::outs() << "trying to convert " << *cvt << " and " << *op << "\n";
    auto can_fold_conversion = [&](Operation *op) {
      if (!op)
        return false;
      // if `cvt` is in a loop and `op` doesn't depend on any other variables in
      // that loop
      auto forParent = cvt->getParentOfType<mlir::scf::ForOp>();
      if (forParent) {
        auto isInLoop = [&](Value v) {
          return v.getDefiningOp() &&
                 v.getDefiningOp()->getParentOfType<mlir::scf::ForOp>() ==
                     forParent;
        };
        if (std::find_if(op->operand_begin(), op->operand_end(), isInLoop) ==
            op->operand_end())
          return true;
      }
      if (isa<triton::gpu::ConvertLayoutOp>(op))
        return op != cvt;
      // return cvt->getResult(0).getType() == op->getOperand(0).getType();
      if (isa<arith::ConstantOp, triton::MakeRangeOp, triton::SplatOp>(op))
        return true;
      return false;
    };
    // determine whether an operation is expensive to rematerialize
    auto expensive_to_remat = [](Operation *op) {
      if (!op)
        return true;
      if (isa<triton::gpu::ExtractSliceOp, triton::gpu::AllocTensorOp,
              triton::gpu::InsertSliceAsyncOp, triton::LoadOp, triton::StoreOp,
              triton::DotOp>(op))
        return true;
      if (isa<scf::YieldOp, scf::ForOp>(op))
        return true;
      return false;
    };
    // we first get the backward dependencies into the conversion
    // stopping whenever the conversion can be folded
    // this will be our potential list of operations to rematerialize
    SetVector<Operation *> depIntoOps;
    mlir::getBackwardSlice(cvt, &depIntoOps, [&](Operation *op) {
      return !can_fold_conversion(op) && op->getResult(0) &&
             (op->getResult(0).getParentBlock() ==
              cvt->getResult(0).getParentBlock());
    });
    // if there is anything expensive to rematerialize, we don't do anything
    if (depIntoOps.empty())
      return mlir::failure();
    auto it = llvm::find_if(depIntoOps, expensive_to_remat);
    if (it != depIntoOps.end()) {
      // if (isa<arith::TruncFOp>(op))
      //   for (Operation *op : depIntoOps)
      //     llvm::outs() << *op << "\n";
      // llvm::outs() << "contains something expensive to rematerialize at "
      //              << **it << "\n";
      return mlir::failure();
    }
    // we only rematerialize if all the conversions that we create get folded
    for (Operation *op : depIntoOps) {
      for (Value arg : op->getOperands()) {
        bool is_backpropagated = depIntoOps.contains(arg.getDefiningOp());
        bool is_foldable = can_fold_conversion(arg.getDefiningOp());
        if (!is_backpropagated && !is_foldable)
          return mlir::failure();
      }
    }
    // we can now rematerialize all operations into depIntoOps recursively
    // for each operation `op`, we convert cvt(op(arg_0, arg_1, ..., arg_n))
    // into op(cvt_0(arg_0), cvt_1(arg_1), ..., cvt_n(arg_n))
    FuncOp parentFunc = cvt->getParentOfType<FuncOp>();
    SetVector<Operation *> seen;
    BlockAndValueMapping mapping;
    rematerializeCvt(cvt, rewriter, depIntoOps, seen, mapping);
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
  forOp.getInductionVar();
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
  mapping.map(forOp.getInductionVar(), newForOp.getInductionVar());
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
    // we don't want to rematerialize any conversion to/from shared
    if (isSharedLayout(cvt->getResults()[0]) ||
        isSharedLayout(cvt->getOperand(0)))
      return mlir::failure();
    //
    auto targetEncoding =
        cvt.getOperand().getType().cast<RankedTensorType>().getEncoding();

    auto forOp = dyn_cast<scf::ForOp>(cvt->getParentOp());
    if (!forOp)
      return mlir::failure();
    auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    auto isInLoop = [&](Operation *op) { return op->getParentOp() == forOp; };

    // determine whether layout conversions can be folded into a given operation
    auto can_fold_conversion = [&](Operation *op) {
      if (isa<triton::gpu::ConvertLayoutOp>(op))
        return op != cvt;
      // return cvt->getResult(0).getType() == op->getOperand(0).getType();
      if (isa<scf::YieldOp>(op))
        return true;
      return false;
    };
    // determine whether an operation is expensive to rematerialize
    auto expensive_to_remat = [](Operation *op) {
      if (isa<triton::gpu::ExtractSliceOp, triton::gpu::AllocTensorOp,
              triton::gpu::InsertSliceAsyncOp, triton::LoadOp, triton::StoreOp,
              triton::DotOp>(op))
        return true;
      if (isa<scf::YieldOp, scf::ForOp>(op))
        return true;
      return false;
    };

    SetVector<Operation *> cvtSlices;
    mlir::getForwardSlice(cvt.getResult(), &cvtSlices, [&](Operation *op) {
      return !can_fold_conversion(op) && op->getResult(0) &&
             (op->getResult(0).getParentBlock() ==
              cvt->getResult(0).getParentBlock());
    });
    if (cvtSlices.empty())
      return failure();
    // if there is anything expensive to rematerialize, we don't do anything
    auto it = llvm::find_if(cvtSlices, expensive_to_remat);
    if (cvtSlices.empty())
      return mlir::failure();
    if (it != cvtSlices.end()) {
      return mlir::failure();
    }

    // we try push the conversion forward
    // since we'll be able to move it out of
    // the loop once it reaches the yield op
    // op(cvt(arg_0), arg_1, ..., arg_n)
    // -> cvt(op(arg_0, cvt(arg_1), ..., cvt(arg_n)))
    BlockAndValueMapping mapping;
    Operation *op = cvtSlices.front();
    for (Value arg : op->getOperands()) {
      auto argTensorType = arg.getType().cast<RankedTensorType>();
      Operation *argOp = arg.getDefiningOp();
      if (argOp == cvt)
        mapping.map(arg, cvt.getOperand());
      else if (argTensorType.getEncoding() != targetEncoding) {
        // adds a conversion
        auto newArgType = RankedTensorType::get(argTensorType.getShape(),
                                                argTensorType.getElementType(),
                                                targetEncoding);
        if (argOp)
          rewriter.setInsertionPointAfter(argOp);
        auto cvtI = rewriter.create<triton::gpu::ConvertLayoutOp>(
            arg.getLoc(), newArgType, arg);
        // we've added an unremovable conversion inside of the loop
        // then we cancel everything and we're done
        if (argOp && isInLoop(argOp) && !can_fold_conversion(argOp))
          return failure();
        mapping.map(arg, cvtI);
      }
    }
    rewriter.setInsertionPoint(op);
    Operation *newOp = rewriter.clone(*op, mapping);
    auto origOpType = op->getResult(0).getType().cast<RankedTensorType>();
    auto newRetType = RankedTensorType::get(
        origOpType.getShape(), origOpType.getElementType(), targetEncoding);
    newOp->getResult(0).setType(newRetType);
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
        newAcc, dotOp.allowTF32(), dotOp.transA(), dotOp.transB());

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
