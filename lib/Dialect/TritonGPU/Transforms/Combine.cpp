#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
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

static LogicalResult invertEncoding(Attribute targetEncoding, Operation *op,
                                    Attribute &ret) {
  ret = targetEncoding;
  if (auto expand_dims = dyn_cast<triton::ExpandDimsOp>(op)) {
    ret = triton::gpu::SliceEncodingAttr::get(
        op->getContext(), expand_dims.axis(), targetEncoding);
  }
  if (auto reduce = dyn_cast<triton::ReduceOp>(op)) {
    auto sliceEncoding =
        targetEncoding.dyn_cast<triton::gpu::SliceEncodingAttr>();
    if (!sliceEncoding)
      return failure();
    ret = sliceEncoding.getParent();
  }
  return success();
}

inline bool expensive_to_remat(Operation *op) {
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

Operation *cloneWithInferType(mlir::PatternRewriter &rewriter, Operation *op,
                              BlockAndValueMapping &mapping) {
  Operation *newOp = rewriter.clone(*op, mapping);
  auto origType = op->getResult(0).getType().cast<RankedTensorType>();
  auto newType = RankedTensorType::get(
      origType.getShape(), origType.getElementType(),
      newOp->getOperand(0).getType().cast<RankedTensorType>().getEncoding());
  newOp->getResult(0).setType(newType);
  auto typeInfer = dyn_cast<InferTypeOpInterface>(newOp);
  if (typeInfer) {
    SmallVector<Type, 1> newType;
    auto sucess = typeInfer.inferReturnTypes(
        newOp->getContext(), newOp->getLoc(), newOp->getOperands(),
        newOp->getAttrDictionary(), newOp->getRegions(), newType);
    if (success)
      newOp->getResult(0).setType(newType.front());
  }
  return newOp;
}

// Layout conversions are expensive. They require going through
// shared memory, which is orders of magnitude slower than
// other non-i/o operations in the dialect.
// It therefore makes sense to remove them whenever possible,
// even if it means rematerializing all values whose definitions
// are reachable from it without passing through any memory operation.
class RematerializeBackward : public mlir::RewritePattern {
public:
  RematerializeBackward(mlir::MLIRContext *context)
      : mlir::RewritePattern(triton::gpu::ConvertLayoutOp::getOperationName(),
                             2, context) {}

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
    auto targetType = cvt->getResultTypes()[0].cast<RankedTensorType>();
    // DFS
    SetVector<Operation *> processed;
    SetVector<Attribute> layout;
    std::vector<std::pair<Operation *, Attribute>> queue;
    std::vector<std::pair<Value, Attribute>> toConvert;
    queue.push_back({cvt, targetType.getEncoding()});
    int numCvts = 1;
    while (!queue.empty()) {
      Operation *currOp;
      Attribute currLayout;
      std::tie(currOp, currLayout) = queue.back();
      queue.pop_back();
      // If the current operation is expensive to rematerialize,
      // we stop everything
      if (expensive_to_remat(currOp))
        break;
      // a conversion will be removed here (i.e. transfered to operands)
      numCvts -= 1;
      // done processing
      processed.insert(currOp);
      layout.insert(currLayout);
      // add all operands to the queue
      for (Value argI : currOp->getOperands()) {
        Attribute newEncoding;
        if (failed(invertEncoding(currLayout, currOp, newEncoding)))
          return mlir::failure();
        toConvert.push_back({argI, newEncoding});
        Operation *opArgI = argI.getDefiningOp();
        if (!opArgI)
          continue;
        if (!opArgI || processed.contains(opArgI) ||
            (opArgI->getBlock() != cvt->getBlock()))
          continue;
        // if the conversion can be folded into opArgI then
        // we actually haven't added anny conversion
        if (isa<triton::gpu::ConvertLayoutOp, arith::ConstantOp,
                triton::MakeRangeOp, triton::SplatOp>(*opArgI))
          continue;
        // we add one expensive conversion for the current operand
        numCvts += 1;
        queue.push_back({opArgI, newEncoding});
      }
    }
    // if rematerialization would add more conversions than it removes
    // then we don't do it
    if (numCvts > 0)
      return mlir::failure();

    FuncOp parentFunc = cvt->getParentOfType<FuncOp>();
    bool test = cvt->getResult(0)
                    .getType()
                    .cast<RankedTensorType>()
                    .getEncoding()
                    .isa<triton::gpu::MmaEncodingAttr>();
    // if (test)
    //   llvm::outs() << "--------\nConverting " << *cvt << "\n---------\n";

    BlockAndValueMapping mapping;
    for (int i = toConvert.size() - 1; i >= 0; i--) {
      // unpack information
      Value currOperand;
      Attribute targetLayout;
      std::tie(currOperand, targetLayout) = toConvert[i];
      // if (test)
      //   llvm::outs() << "current " << currOperand << "\n";
      // rematerialize the operand if necessary
      Operation *currOperation = currOperand.getDefiningOp();
      if (processed.contains(currOperation)) {
        currOperation = cloneWithInferType(rewriter, currOperation, mapping);
        currOperand = currOperation->getResult(0);
      }
      if (i == 0)
        break;
      // compute target type for the layout cast
      auto currType = currOperand.getType().cast<RankedTensorType>();
      auto newType = RankedTensorType::get(
          currType.getShape(), currType.getElementType(), targetLayout);
      auto newOperand = rewriter.create<triton::gpu::ConvertLayoutOp>(
          currOperand.getLoc(), newType, currOperand);
      if (currOperation)
        newOperand->moveAfter(currOperation);
      mapping.map(currOperand, newOperand);
    }
    rewriter.replaceOp(cvt, mapping.lookup(cvt->getOperand(0)));
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

class MoveConvertOutOfLoop : public mlir::RewritePattern {
public:
  MoveConvertOutOfLoop(mlir::MLIRContext *context)
      : mlir::RewritePattern(scf::ForOp::getOperationName(), 1, context) {}

  mlir::LogicalResult matchAndRewrite(mlir::Operation *op,
                                      mlir::PatternRewriter &rewriter) const {

    auto forOp = cast<scf::ForOp>(op);
    auto isInLoop = [&](Operation *op) { return op->getParentOp() == forOp; };
    auto iterArgs = forOp.getRegionIterArgs();
    for (auto iterArg : llvm::enumerate(iterArgs)) {
      // skip non-tensor types
      if (!iterArg.value().getType().isa<RankedTensorType>())
        continue;
      // check
      for (auto op : iterArg.value().getUsers()) {
        if (isa<triton::gpu::ConvertLayoutOp>(op)) {
          auto newFor = tryConvertIterArg(forOp, rewriter, iterArg.index(),
                                          op->getResult(0).getType());
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

class RematerializeForward : public mlir::RewritePattern {
public:
  RematerializeForward(mlir::MLIRContext *context)
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
    patterns.add<RematerializeBackward>(context);
    patterns.add<RematerializeForward>(context);
    patterns.add<MoveConvertOutOfLoop>(context);
    patterns.add<BlockedToMMA>(context);

    if (applyPatternsAndFoldGreedily(m, std::move(patterns)).failed()) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<Pass> mlir::createTritonGPUCombineOpsPass() {
  return std::make_unique<TritonGPUCombineOpsPass>();
}