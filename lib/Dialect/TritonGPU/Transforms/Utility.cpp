#include "Utility.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir {

namespace {

class FixupLoop : public mlir::RewritePattern {

public:
  explicit FixupLoop(mlir::MLIRContext *context)
      : mlir::RewritePattern(scf::ForOp::getOperationName(), 2, context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto forOp = cast<scf::ForOp>(op);

    // Rewrite init argument
    SmallVector<Value, 4> newInitArgs = forOp.getInitArgs();
    bool shouldRematerialize = false;
    for (size_t i = 0; i < newInitArgs.size(); i++) {
      if (newInitArgs[i].getType() != forOp.getRegionIterArgs()[i].getType() ||
          newInitArgs[i].getType() != forOp.getResultTypes()[i]) {
        shouldRematerialize = true;
        break;
      }
    }
    if (!shouldRematerialize)
      return failure();

    scf::ForOp newForOp = rewriter.create<scf::ForOp>(
        forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
        forOp.getStep(), newInitArgs);
    newForOp->moveBefore(forOp);
    rewriter.setInsertionPointToStart(newForOp.getBody());
    IRMapping mapping;
    for (const auto &arg : llvm::enumerate(forOp.getRegionIterArgs()))
      mapping.map(arg.value(), newForOp.getRegionIterArgs()[arg.index()]);
    mapping.map(forOp.getInductionVar(), newForOp.getInductionVar());

    for (Operation &op : forOp.getBody()->getOperations()) {
      rewriter.clone(op, mapping);
    }
    rewriter.replaceOp(forOp, newForOp.getResults());
    return success();
  }
};

} // namespace

LogicalResult fixupLoops(ModuleOp mod) {
  auto *ctx = mod.getContext();
  mlir::RewritePatternSet patterns(ctx);
  patterns.add<FixupLoop>(ctx);
  if (applyPatternsAndFoldGreedily(mod, std::move(patterns)).failed())
    return failure();
  return success();
}

// -------------------------------------------------------------------------- //

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
  if (isa<triton::ViewOp, triton::CatOp>(op)) {
    return failure();
  }
  return success();
}

bool expensiveLoadOrStore(Operation *op, Attribute &targetEncoding) {
  // Case 1: A size 1 tensor is not expensive since all threads will load the
  // same
  if (isSingleValue(op->getOperand(0)))
    return false;
  // Case 2: Tensor of pointers has more threads than elements
  // we can presume a high hit-rate that makes it cheap to load
  auto ptrType = op->getOperand(0).getType().cast<RankedTensorType>();
  IntegerAttr numWarps =
      op->getParentOfType<ModuleOp>()->getAttrOfType<IntegerAttr>(
          "triton_gpu.num-warps");
  if (numWarps) {
    int sizePerThread = triton::gpu::getTotalElemsPerThread(ptrType);
    if (ptrType.getNumElements() < numWarps.getInt() * 32)
      return false;
  }
  return true;
}

bool expensiveToRemat(Operation *op, Attribute &targetEncoding) {
  if (!op)
    return true;
  if (isa<triton::LoadOp, triton::StoreOp>(op))
    return expensiveLoadOrStore(op, targetEncoding);
  if (isa<tensor::ExtractSliceOp, triton::gpu::AllocTensorOp,
          triton::gpu::InsertSliceAsyncOp, triton::AtomicRMWOp,
          triton::AtomicCASOp, triton::DotOp>(op))
    return true;
  if (isa<scf::YieldOp, scf::ForOp, scf::IfOp, scf::WhileOp, scf::ConditionOp>(
          op))
    return true;
  return false;
}

bool canFoldConversion(Operation *op) {
  return isa<triton::gpu::ConvertLayoutOp, arith::ConstantOp,
             triton::MakeRangeOp, triton::SplatOp, triton::ViewOp,
             triton::CatOp>(*op);
}

int simulateBackwardRematerialization(
    Operation *initOp, SetVector<Operation *> &processed,
    SetVector<Attribute> &layout, llvm::MapVector<Value, Attribute> &toConvert,
    Attribute targetEncoding) {
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
        return INT_MAX;
      if (toConvert.count(argI) && toConvert[argI] != newEncoding)
        return INT_MAX;
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
      if (canFoldConversion(opArgI))
        continue;

      // We add one expensive conversion for the current operand
      numCvts += 1;
      queue.emplace_back(opArgI, newEncoding);
    }
  }
  // return net number of conversions
  return numCvts;
}

//

Operation *cloneWithInferType(mlir::OpBuilder &rewriter, Operation *op,
                              IRMapping &mapping) {
  Operation *newOp = rewriter.clone(*op, mapping);
  // if input types haven't changed, we're done
  bool preserveTypes =
      std::all_of(op->operand_begin(), op->operand_end(), [&](Value v) {
        return !mapping.contains(v) ||
               v.getType() == mapping.lookup(v).getType();
      });
  if (preserveTypes)
    return newOp;

  if (newOp->getNumResults() == 0)
    return newOp;
  auto origType = op->getResult(0).getType().dyn_cast<RankedTensorType>();
  auto argType = newOp->getOperand(0).getType().dyn_cast<RankedTensorType>();
  if (!origType || !argType)
    return newOp;
  auto newType = RankedTensorType::get(
      origType.getShape(), origType.getElementType(), argType.getEncoding());
  newOp->getResult(0).setType(newType);
  auto typeInfer = dyn_cast<InferTypeOpInterface>(newOp);
  if (typeInfer) {
    SmallVector<Type, 1> newTypes;
    auto success = typeInfer.inferReturnTypes(
        newOp->getContext(), newOp->getLoc(), newOp->getOperands(),
        newOp->getAttrDictionary(), newOp->getPropertiesStorage(),
        newOp->getRegions(), newTypes);
    if (succeeded(success))
      newOp->getResult(0).setType(newTypes.front());
  }
  return newOp;
}

void rematerializeConversionChain(
    const llvm::MapVector<Value, Attribute> &toConvert,
    mlir::PatternRewriter &rewriter, SetVector<Operation *> &processed,
    IRMapping &mapping) {
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

  for (Value currOperand : sortedValues) {
    Value origOperand = currOperand;
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
      newOperand->moveBefore(block, block->begin());
    }
    mapping.map(origOperand, newOperand);
  }
}

} // namespace mlir
