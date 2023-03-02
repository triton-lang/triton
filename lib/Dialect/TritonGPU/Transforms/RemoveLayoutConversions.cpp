#include "Utility.h"
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
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"

#include <memory>

using namespace mlir;
namespace {
using triton::DotOp;
using triton::gpu::ConvertLayoutOp;
using triton::gpu::DotOperandEncodingAttr;
using triton::gpu::MmaEncodingAttr;
using triton::gpu::SliceEncodingAttr;

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
          !dstParent.isa<triton::gpu::MmaEncodingAttr>())
        return mlir::failure();
      auto dstParentMma = dstParent.cast<triton::gpu::MmaEncodingAttr>();
      if (dstParentMma.isVolta() || dstParentMma.getWarpsPerCTA()[1] > 1)
        return mlir::failure();
      SetVector<Operation *> bwdSlices;
      mlir::getBackwardSlice(convert.getResult(), &bwdSlices);
      if (llvm::find_if(bwdSlices, [](Operation *op) {
            return isa<triton::DotOp>(op);
          }) == bwdSlices.end())
        return mlir::failure();

      auto tmpType = RankedTensorType::get(
          dstType.getShape(), dstType.getElementType(), dstParentMma);
      auto tmp = rewriter.create<triton::gpu::ConvertLayoutOp>(
          convert.getLoc(), tmpType, convert.getOperand());
      auto newConvert = rewriter.create<triton::gpu::ConvertLayoutOp>(
          convert.getLoc(), dstType, tmp);
      rewriter.replaceOp(op, {newConvert});
      return mlir::success();
    }
    return mlir::failure();
  }
};

class SimplifyReduceCvt : public mlir::RewritePattern {
public:
  explicit SimplifyReduceCvt(mlir::MLIRContext *context)
      : mlir::RewritePattern(triton::ReduceOp::getOperationName(), 2, context) {
  }

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto reduce = cast<triton::ReduceOp>(*op);
    auto reduceArg = dyn_cast_or_null<triton::gpu::ConvertLayoutOp>(
        reduce.getOperand().getDefiningOp());
    if (!reduceArg)
      return mlir::failure();
    // this may generate unsupported conversions in the LLVM codegen
    if (reduceArg.getOperand()
            .getType()
            .cast<RankedTensorType>()
            .getEncoding()
            .isa<triton::gpu::MmaEncodingAttr>())
      return mlir::failure();
    auto newReduce = rewriter.create<triton::ReduceOp>(
        op->getLoc(), reduce.getRedOp(), reduceArg.getOperand(),
        reduce.getAxis());
    if (isa<triton::gpu::ConvertLayoutOp>(
            *reduceArg.getOperand().getDefiningOp()))
      return mlir::failure();
    Value newRet = newReduce.getResult();
    // it's still beneficial to move the conversion
    // to after the reduce if necessary since it will be
    // done on a rank-reduced tensor hence cheaper
    if (newRet.getType() != reduce.getResult().getType())
      newRet = rewriter.create<triton::gpu::ConvertLayoutOp>(
          op->getLoc(), reduce.getResult().getType(), newRet);
    rewriter.replaceOp(op, newRet);

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

// TODO: Interface
LogicalResult invertEncoding(Attribute targetEncoding, Operation *op,
                             Attribute &ret) {
  ret = targetEncoding;
  if (auto expand_dims = dyn_cast<triton::ExpandDimsOp>(op)) {
    ret = triton::gpu::SliceEncodingAttr::get(
        op->getContext(), expand_dims.getAxis(), targetEncoding);
  }
  if (auto reduce = dyn_cast<triton::ReduceOp>(op)) {
    auto sliceEncoding =
        targetEncoding.dyn_cast<triton::gpu::SliceEncodingAttr>();
    if (!sliceEncoding)
      return failure();
    if (sliceEncoding.getDim() != reduce.getAxis())
      return failure();
    ret = sliceEncoding.getParent();
  }
  if (auto view = dyn_cast<triton::ViewOp>(op)) {
    return failure();
  }
  return success();
}

inline bool expensiveLoadOrStore(Operation *op, Attribute &targetEncoding) {
  // Case 1: A size 1 tensor is not expensive since all threads will load the
  // same
  if (isSingleValue(op->getOperand(0)))
    return false;
  // auto ptr = op->getOperand(0);
  //// Case 2: We assume that `evict_last` loads/stores have high hit rate
  // if (auto load = dyn_cast<triton::LoadOp>(op))
  //   if (load.getEvict() == triton::EvictionPolicy::EVICT_LAST)
  //     return false;
  // if (auto store = dyn_cast<triton::StoreOp>(op))
  //   if (store.getEvict() == triton::EvictionPolicy::EVICT_LAST)
  //     return false;
  // if (auto tensorTy = ptr.getType().dyn_cast<RankedTensorType>()) {
  //   auto encoding = tensorTy.getEncoding();
  //   // Case 3: Different type conversion is expensive (e.g., mma <-> block)
  //   if (encoding.getTypeID() != targetEncoding.getTypeID())
  //     return true;
  //   auto sizePerThread = triton::gpu::getSizePerThread(encoding);
  //   auto targetSizePerThread = triton::gpu::getSizePerThread(targetEncoding);
  //   auto order = triton::gpu::getOrder(encoding);
  //   auto targetOrder = triton::gpu::getOrder(targetEncoding);
  //   // Case 4: The targeEncoding may expose more vectorization opportunities
  //   return sizePerThread[order[0]] >= targetSizePerThread[targetOrder[0]];
  // }
  return true;
}

inline bool expensiveToRemat(Operation *op, Attribute &targetEncoding) {
  if (!op)
    return true;
  if (isa<triton::LoadOp, triton::StoreOp>(op))
    return expensiveLoadOrStore(op, targetEncoding);
  if (isa<triton::gpu::ExtractSliceOp, triton::gpu::AllocTensorOp,
          triton::gpu::InsertSliceAsyncOp, triton::AtomicRMWOp,
          triton::AtomicCASOp, triton::DotOp>(op))
    return true;
  if (isa<scf::YieldOp, scf::ForOp, scf::IfOp, scf::WhileOp, scf::ConditionOp>(
          op))
    return true;
  return false;
}

LogicalResult simulateBackwardRematerialization(
    Operation *initOp, SetVector<Operation *> &processed,
    SetVector<Attribute> &layout, llvm::MapVector<Value, Attribute> &toConvert,
    const Attribute &targetEncoding) {
  // DFS
  std::vector<std::pair<Operation *, Attribute>> queue;
  queue.emplace_back(initOp, targetEncoding);
  // We want to see the effect of converting `initOp` to a new layout
  // so we initialize `numCvts = 1`.
  int numCvts = 1;
  while (!queue.empty()) {
    Operation *currOp;
    Attribute currLayout;
    std::tie(currOp, currLayout) = queue.back();
    queue.pop_back();
    // If the current operation is expensive to rematerialize,
    // we stop everything
    if (expensiveToRemat(currOp, currLayout))
      break;
    // A conversion will be removed here (i.e. transferred to operands)
    numCvts -= 1;
    // Done processing
    processed.insert(currOp);
    layout.insert(currLayout);
    // Add all operands to the queue
    for (Value argI : currOp->getOperands()) {
      Attribute newEncoding;
      // Cannot invert the current encoding for this operand
      // we stop everything
      if (failed(invertEncoding(currLayout, currOp, newEncoding)))
        return mlir::failure();
      if (toConvert.count(argI) && toConvert[argI] != newEncoding)
        return mlir::failure();
      Operation *opArgI = argI.getDefiningOp();
      toConvert.insert({argI, newEncoding});
      // 1. Only convert RankedTensorType
      // 2. Skip if there's no defining op
      // 3. Skip if the defining op has already been processed
      // 4. Skip or the defining op is in a different block
      if (!argI.getType().isa<RankedTensorType>() || !opArgI ||
          processed.contains(opArgI) ||
          opArgI->getBlock() != currOp->getBlock())
        continue;
      // If the conversion can be folded into opArgI then
      // we don't count this conversion as expensive
      if (isa<triton::gpu::ConvertLayoutOp, arith::ConstantOp,
              triton::MakeRangeOp, triton::SplatOp>(*opArgI))
        continue;
      if (auto view = dyn_cast<triton::ViewOp>(opArgI))
        continue;

      // We add one expensive conversion for the current operand
      numCvts += 1;
      queue.emplace_back(opArgI, newEncoding);
    }
  }
  // if rematerialization would add more conversions than it removes
  // then we don't do it
  if (numCvts > 0)
    return mlir::failure();
  return mlir::success();
}

//

Operation *cloneWithInferType(mlir::PatternRewriter &rewriter, Operation *op,
                              IRMapping &mapping) {
  Operation *newOp = rewriter.clone(*op, mapping);
  auto origType = op->getResult(0).getType().cast<RankedTensorType>();
  auto argType = newOp->getOperand(0).getType().cast<RankedTensorType>();
  auto newType = RankedTensorType::get(
      origType.getShape(), origType.getElementType(), argType.getEncoding());
  newOp->getResult(0).setType(newType);
  auto typeInfer = dyn_cast<InferTypeOpInterface>(newOp);
  if (typeInfer) {
    SmallVector<Type, 1> newTypes;
    auto success = typeInfer.inferReturnTypes(
        newOp->getContext(), newOp->getLoc(), newOp->getOperands(),
        newOp->getAttrDictionary(), newOp->getRegions(), newTypes);
    if (succeeded(success))
      newOp->getResult(0).setType(newTypes.front());
  }
  return newOp;
}

// op(cvt(arg_0), arg_1, ..., arg_n)
// -> cvt(op(arg_0, cvt(arg_1), ..., cvt(arg_n)))
void pushConversionForward(triton::gpu::ConvertLayoutOp cvt,
                           SetVector<Operation *> &cvtSlices,
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
      auto thenCvt = dyn_cast<triton::gpu::ConvertLayoutOp>(
          thenYield.getOperand(i).getDefiningOp());
      if (hasElse) {
        auto elseYield = ifOp.elseYield();
        auto elseCvt = dyn_cast<triton::gpu::ConvertLayoutOp>(
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
public:
  explicit RematerializeForward(mlir::MLIRContext *context)
      : mlir::RewritePattern(triton::gpu::ConvertLayoutOp::getOperationName(),
                             1, context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *cvtOp,
                  mlir::PatternRewriter &rewriter) const override {
    auto cvt = dyn_cast<triton::gpu::ConvertLayoutOp>(*cvtOp);
    auto srcEncoding =
        cvt.getOperand().getType().cast<RankedTensorType>().getEncoding();
    auto dstEncoding =
        cvt.getResult().getType().cast<RankedTensorType>().getEncoding();
    // XXX: why is this needed?
    if (srcEncoding.isa<triton::gpu::SliceEncodingAttr>())
      return failure();
    SetVector<Operation *> cvtSlices;
    auto filter = [&](Operation *op) {
      return op->getBlock() == cvt->getBlock() &&
             !(isa<triton::ReduceOp>(op) &&
               !op->getResult(0).getType().isa<RankedTensorType>()) &&
             !isa<triton::gpu::ConvertLayoutOp>(op) && !isa<scf::YieldOp>(op);
    };
    mlir::getForwardSlice(cvt.getResult(), &cvtSlices, filter);
    if (cvtSlices.empty()) {
      return failure();
    }

    llvm::MapVector<Value, Attribute> toConvert;
    for (Operation *op : cvtSlices) {
      // don't rematerialize anything expensive
      if (expensiveToRemat(op, dstEncoding)) {
        return failure();
      }
      // don't rematerialize non-element-wise
      if (!isa<triton::ViewOp, triton::CatOp>(op) &&
          !op->hasTrait<mlir::OpTrait::SameOperandsAndResultEncoding>() &&
          !op->hasTrait<mlir::OpTrait::Elementwise>() &&
          !isa<triton::StoreOp>(op)) {
        return failure();
      }
      // don't rematerialize if it adds an extra conversion that can't
      // be removed
      for (Value arg : op->getOperands()) {
        Operation *argOp = arg.getDefiningOp();
        SetVector<Operation *> processed;
        SetVector<Attribute> layout;
        llvm::MapVector<Value, Attribute> toConvert;
        if (argOp && (argOp != cvt) && cvtSlices.count(argOp) == 0 &&
            failed(simulateBackwardRematerialization(argOp, processed, layout,
                                                     toConvert, srcEncoding))) {
          return failure();
        }
      }
    }

    pushConversionForward(cvt, cvtSlices, rewriter);
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
public:
  explicit RematerializeBackward(mlir::MLIRContext *context)
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
    if (isSharedEncoding(cvt->getResults()[0]) ||
        isSharedEncoding(cvt->getOperand(0)))
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
    std::vector<std::pair<Operation *, Attribute>> queue;
    if (failed(simulateBackwardRematerialization(
            cvt, processed, layout, toConvert, targetType.getEncoding())))
      return mlir::failure();

    SmallVector<Value, 4> sortedValues;
    SetVector<Operation *> tmp;
    for (auto &item : toConvert) {
      Value v = item.first;
      if (v.getDefiningOp())
        tmp.insert(v.getDefiningOp());
      else
        sortedValues.push_back(v);
    }
    tmp = mlir::multiRootTopologicalSort(tmp);
    for (Operation *op : tmp)
      sortedValues.push_back(op->getResult(0));

    IRMapping mapping;
    for (Value currOperand : sortedValues) {
      // unpack information
      Attribute targetLayout = toConvert.lookup(currOperand);
      // rematerialize the operand if necessary
      Operation *currOperation = currOperand.getDefiningOp();
      if (processed.contains(currOperation)) {
        Operation *newOperation =
            cloneWithInferType(rewriter, currOperation, mapping);
        newOperation->moveAfter(currOperation);
        currOperation = newOperation;
        currOperand = currOperation->getResult(0);
      }
      // compute target type for the layout cast
      auto currType = currOperand.getType().cast<RankedTensorType>();
      auto newType = RankedTensorType::get(
          currType.getShape(), currType.getElementType(), targetLayout);
      auto newOperand = rewriter.create<triton::gpu::ConvertLayoutOp>(
          currOperand.getLoc(), newType, currOperand);
      if (currOperation)
        newOperand->moveAfter(currOperation);
      else {
        Block *block = currOperand.cast<BlockArgument>().getOwner();
        newOperand->moveAfter(block, block->begin());
      }
      mapping.map(currOperand, newOperand);
    }
    rewriter.replaceOp(cvt, mapping.lookup(cvt->getOperand(0)));
    return mlir::success();
  }
};

// -----------------------------------------------------------------------------
//
// -----------------------------------------------------------------------------

class MoveConvertOutOfLoop : public mlir::RewritePattern {
public:
  explicit MoveConvertOutOfLoop(mlir::MLIRContext *context)
      : mlir::RewritePattern(scf::ForOp::getOperationName(), 1, context) {}

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
    // the iter arg of interest may have other uses than the conversion
    // we're hoisting out of the loop. If that's the case we will
    // need to add extra conversions for all uses... which is only useful
    // if these extra conversions can be removed by another pattern
    auto oldArg = forOp.getRegionIterArgs()[i];
    auto newArg = newForOp.getRegionIterArgs()[i];
    auto newArgFallback = rewriter.create<triton::gpu::ConvertLayoutOp>(
        newForOp.getLoc(), origType, newArg);

    mapping.map(forOp.getInductionVar(), newForOp.getInductionVar());
    for (Operation &op : forOp.getBody()->without_terminator()) {
      if (&op == (Operation *)(&origConversion))
        continue;
      Operation *newOp = rewriter.clone(op, mapping);
      if (find(oldArg.getUsers(), &op) != oldArg.getUsers().end())
        newOp->replaceUsesOfWith(newArg, newArgFallback);
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
    return newResults;
  }

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto forOp = cast<scf::ForOp>(op);
    auto iterArgs = forOp.getRegionIterArgs();
    for (const auto &iterArg : llvm::enumerate(iterArgs)) {
      // if (iterArg.index() != 1)
      //   continue;
      // skip non-tensor types
      if (!iterArg.value().getType().isa<RankedTensorType>())
        continue;
      // we only move `iterArg` out of the loop if
      //   - there is only a single conversion use
      //   - moving this conversion out of the loop will not generate
      //     any extra non-removable conversion
      auto users = iterArg.value().getUsers();
      // check first condition
      SetVector<Type> cvtTargetTypes;
      for (auto user : users) {
        if (isa<triton::gpu::ConvertLayoutOp>(user)) {
          auto newType =
              user->getResults()[0].getType().cast<RankedTensorType>();
          auto oldType = user->getOperand(0).getType().cast<RankedTensorType>();
          if (oldType.getEncoding().isa<triton::gpu::SharedEncodingAttr>() &&
              newType.getEncoding()
                  .isa<triton::gpu::DotOperandEncodingAttr>()) {
            continue;
          }
          if (newType.getEncoding().isa<triton::gpu::SharedEncodingAttr>()) {
            if (newType.getEncoding()
                    .cast<triton::gpu::SharedEncodingAttr>()
                    .getVec() == 1)
              continue;
          }
          cvtTargetTypes.insert(newType);
        }
      }
      if (cvtTargetTypes.size() != 1)
        continue;
      // TODO: check second condition
      for (auto user : users) {
        if (isa<triton::gpu::ConvertLayoutOp>(user))
          continue;
      }
      // check
      for (auto op : iterArg.value().getUsers()) {
        auto cvt = dyn_cast<triton::gpu::ConvertLayoutOp>(op);
        if (!cvt)
          continue;
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
    auto _0f = rewriter.create<arith::ConstantFloatOp>(
        op->getLoc(), APFloat(0.0f), dstTy.getElementType().cast<FloatType>());
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

    patterns.add<SimplifyConversion>(context);
    patterns.add<SimplifyReduceCvt>(context);
    patterns.add<RematerializeBackward>(context);
    patterns.add<RematerializeForward>(context);
    patterns.add<MoveConvertOutOfLoop>(context);
    patterns.add<MoveConvertOutOfIf>(context);
    patterns.add<DecomposeDotOperand>(context);
    patterns.add<ConvertDotConvert>(context);

    if (applyPatternsAndFoldGreedily(m, std::move(patterns)).failed()) {
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
