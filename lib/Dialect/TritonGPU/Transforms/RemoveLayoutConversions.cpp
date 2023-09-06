#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
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
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

#include <memory>

using namespace mlir;
namespace {
using triton::DotOp;
using triton::gpu::ConvertLayoutOp;
using triton::gpu::DotOperandEncodingAttr;
using triton::gpu::MmaEncodingAttr;
using triton::gpu::SliceEncodingAttr;

struct PatternSharedInfo {
  // If a conversion cannot be eliminated with a high-benefit pattern (e.g.,
  // SimplifyConversion, RematerializeBackward), it will be pushed forward in
  // the hope that this will enable the elimination of these conversions later.
  // However, pushing a conversion forward can introduce more conversions
  // (op(cvt(arg_0), arg_1, ..., arg_n) -> cvt(op(arg_0, cvt(arg_1), ...,
  // cvt(arg_n))). This is why the RematerializeForward pattern performs an
  // analysis to determine whether these added conversions can be eliminated
  // later. The RematerializeBackward pattern, applied after pushing this
  // conversion forward, will eliminate these newly added conversions by
  // reversing the process achieved with RematerializeForward. This can create
  // an infinite loop between these two optimizations. To avoid this, we keep
  // track of the conversions that were pushed forward and skip them in the
  // RematerializeBackward pattern. A similar kind of loop can occur with the
  // RematerializeForward and MoveConvertOutOfLoop patterns.
  llvm::DenseMap<Operation *, Operation *> cvtsPushedForwardMap;
};

// -----------------------------------------------------------------------------
//
// -----------------------------------------------------------------------------

// convert(blocked, dot_operand) ->
// convert(blocked, mma) + convert(mma,  dot_operand)
// if this value is itself the result of a dot operation
// this is a heuristic to accommodate some pattern seen in fused attention
// kernels.
// TODO: replace this by something more generic, i.e. layout-aware CSE
class DecomposeDotOperand : public mlir::RewritePattern {

public:
  explicit DecomposeDotOperand(mlir::MLIRContext *context)
      : mlir::RewritePattern(triton::gpu::ConvertLayoutOp::getOperationName(),
                             1, context) {}

  template <typename encTy>
  mlir::LogicalResult processEncoding(encTy encoding,
                                      triton::gpu::ConvertLayoutOp convert,
                                      RankedTensorType &dstType,
                                      mlir::PatternRewriter &rewriter) const {
    SetVector<Operation *> bwdSlices;
    mlir::getBackwardSlice(convert.getResult(), &bwdSlices);
    if (llvm::find_if(bwdSlices, [](Operation *op) {
          return isa<triton::DotOp>(op);
        }) == bwdSlices.end())
      return mlir::failure();

    auto tmpType = RankedTensorType::get(dstType.getShape(),
                                         dstType.getElementType(), encoding);
    auto tmp = rewriter.create<triton::gpu::ConvertLayoutOp>(
        convert.getLoc(), tmpType, convert.getOperand());
    rewriter.replaceOpWithNewOp<triton::gpu::ConvertLayoutOp>(convert, dstType,
                                                                tmp);
    return mlir::success();
  }

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!llvm::isa<triton::gpu::ConvertLayoutOp>(op))
      return mlir::failure();
    auto convert = llvm::cast<triton::gpu::ConvertLayoutOp>(op);
    auto srcType = convert.getOperand().getType().cast<RankedTensorType>();
    auto dstType = convert.getType().cast<RankedTensorType>();
    if (srcType.getEncoding().isa<triton::gpu::BlockedEncodingAttr>() &&
        dstType.getEncoding().isa<triton::gpu::DotOperandEncodingAttr>()) {
      auto dstDotOperand =
          dstType.getEncoding().cast<triton::gpu::DotOperandEncodingAttr>();
      auto dstParent = dstDotOperand.getParent();
      if (dstDotOperand.getOpIdx() == 1 ||
          (!dstParent.isa<triton::gpu::MmaEncodingAttr>() &&
           !dstParent.isa<triton::gpu::MfmaEncodingAttr>()))
        return mlir::failure();

      if (dstParent.isa<triton::gpu::MmaEncodingAttr>()) {
        auto dstParentMma = dstParent.cast<triton::gpu::MmaEncodingAttr>();
        if (dstParentMma.isVolta() || dstParentMma.getWarpsPerCTA()[1] > 1)
          return mlir::failure();
        return processEncoding(dstParentMma, convert, dstType, rewriter);
      }

      if (dstParent.isa<triton::gpu::MfmaEncodingAttr>()) {
        auto dstParentMfma = dstParent.cast<triton::gpu::MfmaEncodingAttr>();
        if (dstParentMfma.getWarpsPerCTA()[1] > 1)
          return mlir::failure();
        return processEncoding(dstParentMfma, convert, dstType, rewriter);
      }
    }
    return mlir::failure();
  }
};

// It's beneficial to move the conversion
// to after the reduce if necessary since it will be
// done on a rank-reduced tensor hence cheaper
class SimplifyReduceCvt : public mlir::RewritePattern {
public:
  explicit SimplifyReduceCvt(mlir::MLIRContext *context)
      : mlir::RewritePattern(triton::gpu::ConvertLayoutOp::getOperationName(),
                             2, context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!llvm::isa<triton::gpu::ConvertLayoutOp>(op))
      return mlir::failure();
    auto convert = llvm::cast<triton::gpu::ConvertLayoutOp>(op);
    triton::ReduceOp reduce;
    for (auto &use : convert.getResult().getUses()) {
      auto owner = llvm::dyn_cast<triton::ReduceOp>(use.getOwner());
      if (!owner) {
        continue;
      }

      // TODO: This only moves conversions from the first argument which is
      // fine for argmin/argmax but may not be optimal generally
      if (convert.getResult() != owner.getOperands()[0]) {
        continue;
      }
      reduce = owner;
      break;
    }
    if (!reduce)
      return mlir::failure();

    SmallVector<Value> newOperands = reduce.getOperands();

    newOperands[0] = convert.getOperand();
    auto newEncoding =
        newOperands[0].getType().cast<RankedTensorType>().getEncoding();

    // this may generate unsupported conversions in the LLVM codegen
    if (newEncoding.isa<triton::gpu::MmaEncodingAttr>() ||
        newEncoding.isa<triton::gpu::MfmaEncodingAttr>()) {
      return failure();
    }

    for (unsigned i = 1; i < newOperands.size(); ++i) {
      auto oldTy = newOperands[i].getType().cast<RankedTensorType>();
      RankedTensorType newTy =
          RankedTensorType::Builder(oldTy).setEncoding(newEncoding);

      newOperands[i] = rewriter.create<triton::gpu::ConvertLayoutOp>(
          op->getLoc(), newTy, newOperands[i]);
    }

    rewriter.setInsertionPoint(reduce);
    auto newReduce = rewriter.create<triton::ReduceOp>(
        op->getLoc(), newOperands, reduce.getAxis());
    auto &newCombineOp = newReduce.getCombineOp();
    rewriter.cloneRegionBefore(reduce.getCombineOp(), newCombineOp,
                               newCombineOp.end());

    SmallVector<Value> newRet = newReduce.getResult();
    auto oldTypes = reduce.getResult().getType();
    for (unsigned i = 0; i < reduce.getNumOperands(); ++i) {
      // it's still beneficial to move the conversion
      // to after the reduce if necessary since it will be
      // done on a rank-reduced tensor hence cheaper
      if (newRet[i].getType() != oldTypes[i])
        newRet[i] = rewriter.create<triton::gpu::ConvertLayoutOp>(
            op->getLoc(), oldTypes[i], newRet[i]);
    }
    rewriter.replaceAllUsesWith(reduce.getResult(), newRet);

    return success();
  }
};

// Layout conversions can't deduce their return type automatically.
// IIUC they are therefore not handled by DRR right now
class SimplifyConversion : public mlir::RewritePattern {
public:
  explicit SimplifyConversion(mlir::MLIRContext *context)
      : mlir::RewritePattern(triton::gpu::ConvertLayoutOp::getOperationName(),
                             4, context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!llvm::isa<triton::gpu::ConvertLayoutOp>(op))
      return mlir::failure();
    auto convert = llvm::cast<triton::gpu::ConvertLayoutOp>(op);
    return ConvertLayoutOp::canonicalize(convert, rewriter);
  }
};

// -----------------------------------------------------------------------------
//
// -----------------------------------------------------------------------------

// op(cvt(arg_0), arg_1, ..., arg_n)
// -> cvt(op(arg_0, cvt(arg_1), ..., cvt(arg_n)))
void pushConversionForward(triton::gpu::ConvertLayoutOp cvt,
                           SetVector<Operation *> &cvtSlices,
                           PatternSharedInfo &sharedInfo,
                           mlir::PatternRewriter &rewriter) {
  auto srcEncoding =
      cvt.getOperand().getType().cast<RankedTensorType>().getEncoding();
  auto dstEncoding =
      cvt.getResult().getType().cast<RankedTensorType>().getEncoding();
  IRMapping mapping;
  auto op = cvtSlices.front();
  for (Value arg : op->getOperands()) {
    if (arg.getDefiningOp() == cvt)
      mapping.map(arg, cvt.getOperand());
    else {
      auto oldType = arg.getType().cast<RankedTensorType>();
      auto newType = RankedTensorType::get(
          oldType.getShape(), oldType.getElementType(), srcEncoding);
      auto cvtI = rewriter.create<triton::gpu::ConvertLayoutOp>(arg.getLoc(),
                                                                newType, arg);
      if (Operation *argOp = arg.getDefiningOp())
        cvtI->moveAfter(argOp);
      mapping.map(arg, cvtI);
    }
  }
  rewriter.setInsertionPoint(op);
  if (op->getNumResults() == 0) {
    Operation *newOp = rewriter.clone(*op, mapping);
    rewriter.eraseOp(op);
    return;
  }
  auto *newOp = cloneWithInferType(rewriter, op, mapping);
  auto newType = newOp->getResult(0).getType().cast<RankedTensorType>();
  auto newCvtType = RankedTensorType::get(
      newType.getShape(), newType.getElementType(), dstEncoding);
  auto newCvt = rewriter.create<triton::gpu::ConvertLayoutOp>(
      newOp->getLoc(), newCvtType, newOp->getResult(0));
  sharedInfo.cvtsPushedForwardMap[newCvt] = newCvt->getOperand(0).getDefiningOp();
  rewriter.replaceOp(op, newCvt->getResults());
}

//
class MoveConvertOutOfIf : public mlir::RewritePattern {
public:
  explicit MoveConvertOutOfIf(mlir::MLIRContext *context)
      : mlir::RewritePattern(scf::IfOp::getOperationName(), 2, context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ifOp = cast<scf::IfOp>(*op);
    // If “scf.if” defines no values, “scf.yield” will be inserted implicitly.
    // However, "scf.else" is not required to be present, so we need to check
    // if it exists.
    auto thenYield = ifOp.thenYield();
    int numOps = thenYield.getNumOperands();
    SmallVector<Value> newThenYieldOps = thenYield.getOperands();
    SetVector<Operation *> thenCvts;
    SmallVector<Type> newRetTypes;

    bool hasElse = !ifOp.getElseRegion().empty();

    scf::YieldOp elseYield;
    SmallVector<Value> newElseYieldOps;
    SetVector<Operation *> elseCvts;
    if (hasElse) {
      elseYield = ifOp.elseYield();
      newElseYieldOps = elseYield.getOperands();
    }

    IRMapping mapping;
    for (size_t i = 0; i < numOps; i++) {
      auto thenCvt = dyn_cast_or_null<triton::gpu::ConvertLayoutOp>(
          thenYield.getOperand(i).getDefiningOp());
      if (hasElse) {
        auto elseYield = ifOp.elseYield();
        auto elseCvt = dyn_cast_or_null<triton::gpu::ConvertLayoutOp>(
            elseYield.getOperand(i).getDefiningOp());
        if (thenCvt && elseCvt &&
            std::distance(elseCvt->user_begin(), elseCvt->user_end()) == 1 &&
            std::distance(thenCvt->user_begin(), thenCvt->user_end()) == 1 &&
            thenCvt.getOperand().getType() == elseCvt.getOperand().getType()) {
          // If thenCvt and elseCvt's type are the same, it means a single
          // conversion is enough to replace both of them. We can move the
          // conversion out of scf.if and replace both thenCvt and elseCvt with
          // the new conversion.
          mapping.map(thenCvt.getResult(), thenCvt.getOperand());
          thenCvts.insert((Operation *)thenCvt);
          newRetTypes.push_back(thenCvt.getOperand().getType());
          mapping.map(elseCvt.getResult(), elseCvt.getOperand());
          elseCvts.insert((Operation *)elseCvt);
        } else
          // Cannot move out of scf.if because thenCvt != elseCvt
          // Moving it out of scf.if will introduce a new conversion
          newRetTypes.push_back(thenYield.getOperand(i).getType());
      } else {
        if (thenCvt &&
            std::distance(thenCvt->user_begin(), thenCvt->user_end()) == 1) {
          // If there's only a single use of the conversion then we can move it
          mapping.map(thenCvt.getResult(), thenCvt.getOperand());
          thenCvts.insert((Operation *)thenCvt);
          newRetTypes.push_back(thenCvt.getOperand().getType());
        } else
          // Cannot move out of scf.if because either there's another use of
          // the conversion or there's no conversion at all
          newRetTypes.push_back(thenYield.getOperand(i).getType());
      }
    }
    if (mapping.getValueMap().empty())
      return mlir::failure();

    auto newIfOp = rewriter.create<scf::IfOp>(ifOp.getLoc(), newRetTypes,
                                              ifOp.getCondition(), hasElse);
    auto rematerialize = [&](Block *block, SetVector<Operation *> &cvts) {
      for (Operation &op : block->getOperations()) {
        if (cvts.contains(&op)) {
          if (mapping.contains(op.getOperand(0)))
            mapping.map(op.getResult(0), mapping.lookup(op.getOperand(0)));
          continue;
        }
        rewriter.clone(op, mapping);
      }
    };
    rewriter.setInsertionPointToEnd(newIfOp.thenBlock());
    rematerialize(ifOp.thenBlock(), thenCvts);
    if (hasElse) {
      rewriter.setInsertionPointToEnd(newIfOp.elseBlock());
      rematerialize(ifOp.elseBlock(), elseCvts);
    }

    rewriter.setInsertionPointAfter(newIfOp);
    SmallVector<Value> newRetValues = newIfOp.getResults();
    for (size_t i = 0; i < numOps; i++) {
      if (newIfOp.getResult(i).getType() != ifOp.getResult(i).getType()) {
        newRetValues[i] = rewriter.create<triton::gpu::ConvertLayoutOp>(
            newIfOp.getLoc(), ifOp.getResult(i).getType(),
            newIfOp.getResult(i));
      }
    }

    rewriter.replaceOp(op, newRetValues);
    return mlir::success();
  }
};

//
class RematerializeForward : public mlir::RewritePattern {
  PatternSharedInfo &sharedInfo;

public:
  explicit RematerializeForward(mlir::MLIRContext *context, PatternSharedInfo &sharedInfo)
      : mlir::RewritePattern(triton::gpu::ConvertLayoutOp::getOperationName(),
                             1, context), sharedInfo(sharedInfo) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *cvtOp,
                  mlir::PatternRewriter &rewriter) const override {
    auto cvt = dyn_cast<triton::gpu::ConvertLayoutOp>(*cvtOp);
    auto srcEncoding =
        cvt.getOperand().getType().cast<RankedTensorType>().getEncoding();
    auto dstEncoding =
        cvt.getResult().getType().cast<RankedTensorType>().getEncoding();
    if (srcEncoding.isa<triton::gpu::SharedEncodingAttr>() ||
        dstEncoding.isa<triton::gpu::SharedEncodingAttr>())
      return failure();
    // heuristics for flash attention
    if (srcEncoding.isa<triton::gpu::SliceEncodingAttr>())
      return failure();
    // For cases like:
    // %0 = convert_layout %arg0
    // We should try to move %0 out of scf.for first, if it couldn't be moved
    // out additional conversions will be added to the loop body.
    if (!cvt.getOperand().getDefiningOp() &&
        isa<scf::ForOp>(cvt->getParentOp()))
      return failure();

    SetVector<Operation *> cvtSlices;
    auto filter = [&](Operation *op) {
      return op->getBlock() == cvt->getBlock() &&
             !isa<triton::gpu::ConvertLayoutOp, scf::YieldOp>(op) &&
             !(isa<triton::ReduceOp>(op) &&
               !op->getResult(0).getType().isa<RankedTensorType>());
    };
    mlir::getForwardSlice(cvt.getResult(), &cvtSlices, {filter});
    if (cvtSlices.empty())
      return failure();

    for (Operation *op : cvtSlices) {
      // don't rematerialize anything expensive
      if (isExpensiveToRemat(op, srcEncoding))
        return failure();
      // don't rematerialize non-element-wise
      if (!op->hasTrait<mlir::OpTrait::SameOperandsAndResultEncoding>() &&
          !op->hasTrait<mlir::OpTrait::Elementwise>() &&
          !isa<triton::StoreOp, triton::AssertOp, triton::PrintOp,
               triton::ReduceOp>(op))
        return failure();
      // don't rematerialize if it adds an extra conversion that can't
      // be removed
      for (Value arg : op->getOperands()) {
        Operation *argOp = arg.getDefiningOp();
        SetVector<Operation *> processed;
        SetVector<Attribute> layout;
        llvm::MapVector<Value, Attribute> toConvert;
        int numAddedConvs = simulateBackwardRematerialization(
            argOp, processed, layout, toConvert, srcEncoding);
        if (argOp && !isa<triton::gpu::ConvertLayoutOp>(argOp) &&
            cvtSlices.count(argOp) == 0 && numAddedConvs > 0)
          return failure();
      }
    }

    // Call SimplifyReduceCvt instead of the general push conversion forward
    if (isa<triton::ReduceOp>(cvtSlices.front()))
      return failure();

    pushConversionForward(cvt, cvtSlices, sharedInfo, rewriter);
    return success();
  }
};

// Layout conversions are expensive. They require going through
// shared memory, which is orders of magnitude slower than
// other non-i/o operations in the dialect.
// It therefore makes sense to remove them whenever possible,
// even if it means rematerializing all values whose definitions
// are reachable from it without passing through any memory operation.
class RematerializeBackward : public mlir::RewritePattern {
  PatternSharedInfo &sharedInfo;

public:
  explicit RematerializeBackward(mlir::MLIRContext *context, PatternSharedInfo &sharedInfo)
      : mlir::RewritePattern(triton::gpu::ConvertLayoutOp::getOperationName(),
                             3, context), sharedInfo(sharedInfo) {}


  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *cvt,
                  mlir::PatternRewriter &rewriter) const override {
    if (!llvm::isa<triton::gpu::ConvertLayoutOp>(cvt))
      return mlir::failure();

    auto it = sharedInfo.cvtsPushedForwardMap.find(cvt);
    if (it != sharedInfo.cvtsPushedForwardMap.end() &&
        it->second == cvt->getOperand(0).getDefiningOp())
      return mlir::failure();

    // we don't touch block arguments
    Operation *op = cvt->getOperand(0).getDefiningOp();
    if (!op)
      return mlir::failure();
    // we don't want to rematerialize any conversion to/from shared
    if (triton::gpu::isSharedEncoding(cvt->getResults()[0]) ||
        triton::gpu::isSharedEncoding(cvt->getOperand(0)))
      return mlir::failure();
    // we don't handle conversions to DotOperandEncodingAttr
    // this is a heuristics to accommodate fused attention
    auto targetType = cvt->getResultTypes()[0].cast<RankedTensorType>();
    if (targetType.getEncoding().isa<triton::gpu::DotOperandEncodingAttr>())
      return mlir::failure();
    // DFS
    SetVector<Operation *> processed;
    SetVector<Attribute> layout;
    llvm::MapVector<Value, Attribute> toConvert;
    if (simulateBackwardRematerialization(cvt, processed, layout, toConvert,
                                          targetType.getEncoding()) > 0)
      return mlir::failure();

    IRMapping mapping;
    rematerializeConversionChain(toConvert, rewriter, processed, mapping);
    rewriter.replaceOp(cvt, mapping.lookup(cvt->getOperand(0)));

    return mlir::success();
  }
};

// -----------------------------------------------------------------------------
//
// -----------------------------------------------------------------------------

class MoveConvertOutOfLoop : public mlir::RewritePattern {
  PatternSharedInfo &sharedInfo;

public:
  explicit MoveConvertOutOfLoop(mlir::MLIRContext *context,
                                PatternSharedInfo &sharedInfo)
      : mlir::RewritePattern(scf::ForOp::getOperationName(), 1, context),
        sharedInfo(sharedInfo) {}

  SmallVector<Value, 4>
  rematerializeForLoop(mlir::PatternRewriter &rewriter, scf::ForOp &forOp,
                       size_t i, RankedTensorType newType,
                       triton::gpu::ConvertLayoutOp origConversion) const {
    // Rewrite init argument
    Type origType = forOp.getInitArgs()[i].getType();
    SmallVector<Value, 4> newInitArgs = forOp.getInitArgs();
    newInitArgs[i] = rewriter.create<triton::gpu::ConvertLayoutOp>(
        newInitArgs[i].getLoc(), newType, newInitArgs[i]);
    // Clone for loop
    auto newForOp = rewriter.create<scf::ForOp>(
        forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
        forOp.getStep(), newInitArgs);
    newForOp->moveBefore(forOp);
    rewriter.setInsertionPointToStart(newForOp.getBody());
    IRMapping mapping;
    for (const auto &arg : llvm::enumerate(forOp.getRegionIterArgs()))
      mapping.map(arg.value(), newForOp.getRegionIterArgs()[arg.index()]);
    mapping.map(origConversion.getResult(), newForOp.getRegionIterArgs()[i]);

    mapping.map(forOp.getInductionVar(), newForOp.getInductionVar());
    for (Operation &op : forOp.getBody()->without_terminator()) {
      if (&op == (Operation *)(&origConversion))
        continue;
      Operation *newOp = rewriter.clone(op, mapping);
    }
    // create yield, inserting conversions if necessary
    auto yieldOp = forOp.getBody()->getTerminator();
    SmallVector<Value, 4> newYieldArgs;
    for (Value arg : yieldOp->getOperands())
      newYieldArgs.push_back(mapping.lookup(arg));
    if (newYieldArgs[i].getType() != newType)
      newYieldArgs[i] = rewriter.create<triton::gpu::ConvertLayoutOp>(
          yieldOp->getLoc(), newType, newYieldArgs[i]);
    rewriter.create<scf::YieldOp>(forOp.getLoc(), newYieldArgs);

    // replace
    SmallVector<Value, 4> newResults = newForOp->getResults();
    newResults[i] = rewriter.create<triton::gpu::ConvertLayoutOp>(
        newForOp.getLoc(), origType, newForOp->getResult(i));
    newResults[i].getDefiningOp()->moveAfter(newForOp);
    return newResults;
  }

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto forOp = cast<scf::ForOp>(op);
    auto iterArgs = forOp.getRegionIterArgs();
    for (const auto &iterArg : llvm::enumerate(iterArgs)) {
      // skip non-tensor types
      if (!iterArg.value().getType().isa<RankedTensorType>())
        continue;
      SmallVector<Operation *> cvts;
      if (canMoveOutOfLoop(iterArg.value(), cvts).failed())
        continue;
      // check
      for (auto *op : cvts) {
        auto cvt = dyn_cast<triton::gpu::ConvertLayoutOp>(op);
        auto it = sharedInfo.cvtsPushedForwardMap.find(cvt);
        if (it != sharedInfo.cvtsPushedForwardMap.end())
          return mlir::failure();
        auto targetType = op->getResultTypes()[0].cast<RankedTensorType>();
        auto newFor = rematerializeForLoop(rewriter, forOp, iterArg.index(),
                                           targetType, cvt);
        rewriter.replaceOp(forOp, newFor);
        return success();
      }
    }
    return failure();
  }
};

//
class ConvertDotConvert : public mlir::RewritePattern {
public:
  ConvertDotConvert(mlir::MLIRContext *context)
      : mlir::RewritePattern(triton::gpu::ConvertLayoutOp::getOperationName(),
                             1, context) {}

  LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto dstOp = cast<triton::gpu::ConvertLayoutOp>(op);
    auto dotOp =
        dyn_cast_or_null<triton::DotOp>(dstOp.getSrc().getDefiningOp());
    if (!dotOp)
      return mlir::failure();
    if (std::distance(dstOp->user_begin(), dstOp->user_end()) != 1 ||
        std::distance(dotOp->user_begin(), dotOp->user_end()) != 1)
      return mlir::failure();
    auto cvtOp = dyn_cast_or_null<triton::gpu::ConvertLayoutOp>(
        dotOp.getOperand(2).getDefiningOp());
    if (!cvtOp)
      return mlir::failure();
    auto loadOp =
        dyn_cast_or_null<triton::LoadOp>(cvtOp.getSrc().getDefiningOp());
    if (!loadOp)
      return mlir::failure();
    auto dstTy = dstOp.getResult().getType().cast<RankedTensorType>();
    auto srcTy = cvtOp.getOperand().getType().cast<RankedTensorType>();
    if (dstTy != srcTy)
      return mlir::failure();

    // TODO: int tensor cores
    auto out_dtype = dstTy.getElementType().cast<FloatType>();
    APFloat value(0.0f);
    if (out_dtype.isBF16())
      value = APFloat(APFloat::IEEEhalf(), APInt(16, 0));
    else if (out_dtype.isF16())
      value = APFloat(APFloat::IEEEhalf(), APInt(16, 0));
    else if (out_dtype.isF32())
      value = APFloat(0.0f);
    else
      llvm_unreachable("unsupported data type");

    auto _0f =
        rewriter.create<arith::ConstantFloatOp>(op->getLoc(), value, out_dtype);
    auto _0 = rewriter.create<triton::SplatOp>(
        op->getLoc(), dotOp.getResult().getType(), _0f);
    auto newDot = rewriter.create<triton::DotOp>(
        op->getLoc(), dotOp.getResult().getType(), dotOp.getOperand(0),
        dotOp.getOperand(1), _0, dotOp.getAllowTF32());
    auto newCvt = rewriter.create<triton::gpu::ConvertLayoutOp>(
        op->getLoc(), dstTy, newDot.getResult());
    rewriter.replaceOpWithNewOp<arith::AddFOp>(op, newCvt, cvtOp.getOperand());
    return mlir::success();
  }
};

} // namespace

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

class TritonGPURemoveLayoutConversionsPass
    : public TritonGPURemoveLayoutConversionsBase<
          TritonGPURemoveLayoutConversionsPass> {
public:
  TritonGPURemoveLayoutConversionsPass() = default;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    mlir::RewritePatternSet patterns(context);
    PatternSharedInfo sharedInfo;

    patterns.add<SimplifyConversion>(context);
    patterns.add<SimplifyReduceCvt>(context);
    patterns.add<RematerializeBackward>(context, sharedInfo);
    patterns.add<RematerializeForward>(context, sharedInfo);
    patterns.add<MoveConvertOutOfLoop>(context, sharedInfo);
    patterns.add<MoveConvertOutOfIf>(context);
    patterns.add<DecomposeDotOperand>(context);
    patterns.add<ConvertDotConvert>(context);

    if (mlir::applyPatternsAndFoldGreedily(m, std::move(patterns)).failed()) {
      signalPassFailure();
    }

    if (fixupLoops(m).failed()) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<Pass> mlir::createTritonGPURemoveLayoutConversionsPass() {
  return std::make_unique<TritonGPURemoveLayoutConversionsPass>();
}
