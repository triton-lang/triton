#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include <memory>
#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

using namespace mlir;

class TritonGPUOptimizeThreadLocalityPass
    : public TritonGPUOptimizeThreadLocalityBase<
          TritonGPUOptimizeThreadLocalityPass> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    DenseSet<triton::ReduceOp> reduceOps;
    mod.walk([&](triton::ReduceOp reduce) -> void {
      auto srcType = reduce.getOperands()[0].getType().cast<RankedTensorType>();
      auto rank = srcType.getShape().size();
      auto srcEncoding = srcType.getEncoding();
      if (!(srcEncoding.isa<triton::gpu::BlockedEncodingAttr>() && rank == 2))
        return;
      if (!reduce->hasOneUse())
        return;
      Operation *user = (reduce->getUses().begin())->getOwner();
      if (!user->hasOneUse())
        return;
      OpOperand &yieldOpOperand = *(user->getUses().begin());
      auto yieldOp = dyn_cast<scf::YieldOp>(yieldOpOperand.getOwner());
      if (!yieldOp)
        return;
      auto operandNumber = yieldOpOperand.getOperandNumber();
      Block *block = reduce->getBlock();
      Operation *parentOp = block->getParentOp();
      auto forOp = dyn_cast<scf::ForOp>(parentOp);
      if (!forOp)
        return;
      auto argNum = yieldOpOperand.getOperandNumber();
      auto oldAccum = forOp.getInitArgs()[argNum];
      auto cstOp = dyn_cast<arith::ConstantOp>(oldAccum.getDefiningOp());
      if (!cstOp)
        return;
      auto oldAccumValue = cstOp.getValue();
      auto denseAttr = oldAccumValue.dyn_cast<DenseFPElementsAttr>();
      // TODO: support non-zero splat by initializing the new accumulator
      // with a neutral value (based on the reduction) then incorporating the
      // splat value to the result (after the loop)
      if (!(denseAttr.isSplat() && denseAttr.getSplatValue<APFloat>().isZero()))
        return;
      reduceOps.insert(reduce);
    });

    for (auto reduce : reduceOps) {
      OpBuilder builder(reduce);
      auto srcType = reduce.getOperands()[0].getType().cast<RankedTensorType>();
      auto srcShape = srcType.getShape();
      auto srcEncoding = srcType.getEncoding();
      assert(srcEncoding.isa<triton::gpu::BlockedEncodingAttr>() &&
             "Thread locality optimization only supports blocked encoding");
      auto blocked = srcEncoding.dyn_cast<triton::gpu::BlockedEncodingAttr>();
      auto elemsPerThread =
          triton::gpu::getElemsPerThread(srcType)[reduce.getAxis()];
      auto rank = srcShape.size();
      // create new layouts
      auto blocked3d = getThreadLocalityOptimizedEncoding(reduce);
      auto viewOpTensorShape = getThreadLocalityOptimizedShape(reduce);
      auto viewOpTensorType = RankedTensorType::get(
          viewOpTensorShape, srcType.getElementType(), blocked3d);
      auto slice2d =
          triton::gpu::SliceEncodingAttr::get(mod.getContext(), 2, blocked3d);
      auto slice1d = triton::gpu::SliceEncodingAttr::get(
          mod.getContext(), reduce.getAxis(), slice2d);
      // Get forOp
      assert(reduce->hasOneUse());
      OpOperand &use = *(reduce->getUses().begin());
      auto operandNumber = use.getOperandNumber();
      auto user = use.getOwner();
      assert(user->getNumOperands() == 2);
      auto accumOperandNumber = (operandNumber == 0) ? 1 : 0;
      auto accumOperand = user->getOperand(accumOperandNumber);
      assert(accumOperand.isa<BlockArgument>());
      auto blockArg = accumOperand.dyn_cast<BlockArgument>();
      auto blockArgNum = blockArg.getArgNumber();
      auto forOp = dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp());
      // get oldAccum
      auto oldAccum =
          forOp.getInitArgs()[blockArgNum - forOp.getNumInductionVars()];
      // get old loop user
      Value loopResult =
          forOp.getResult(blockArgNum - forOp.getNumInductionVars());
      assert(loopResult.hasOneUse());
      OpOperand &loopUse = *(loopResult.getUses().begin());
      Operation *loopUser = loopUse.getOwner();
      // get old loop yield
      auto oldYield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
      // create newAccum initialization
      auto newAccum =
          createAccum(builder, oldAccum, viewOpTensorShape, slice2d);
      // create new loop by copying the old for op signature and appending
      // newAccum to the block arguments
      auto newLoop = replaceForOpWithNewSignature(
          builder, forOp, ValueRange{newAccum->getResult(0)});
      // create thread local reduction (also adds viewOps)
      auto newReduce = createReduce(builder, reduce, viewOpTensorType);

      // create new accum update
      auto newUpdate = createUpdate(builder, newLoop, newReduce, user);
      // create new yield
      auto newYield = createYield(builder, newLoop, oldYield,
                                  newUpdate->getResult(0), blockArgNum);
      // create post loop reduction on the original reduce axis
      auto newReduce2 = createPostLoopReduce(builder, newLoop, reduce);

      // Replace loop user with new reduce
      replaceLoopUser(builder, loopUse, newReduce2);

      oldYield.erase();
      forOp.erase();
    }
  };

private:
  void replaceLoopUser(OpBuilder &builder, OpOperand &loopUse,
                       Operation *newReduce2) const {
    Operation *loopUser = loopUse.getOwner();
    if (auto cvtLayout = dyn_cast<triton::gpu::ConvertLayoutOp>(loopUser)) {
      builder.setInsertionPointAfter(loopUser);
      IRMapping mapping;
      mapping.map(loopUser->getOperands()[0], newReduce2->getResult(0));
      auto newCvt = builder.clone(*loopUser, mapping);
      loopUser->replaceAllUsesWith(newCvt);
      loopUser->erase();
    } else {
      builder.setInsertionPoint(loopUser);
      auto operandIdx = loopUse.getOperandNumber();
      Value operand = loopUser->getOperand(operandIdx);
      Type destType = operand.getType();
      auto newCvt = builder.create<triton::gpu::ConvertLayoutOp>(
          loopUser->getLoc(), destType, newReduce2->getResult(0));
      IRMapping mapping;
      mapping.map(operand, newCvt->getResult(0));
      auto newUser = builder.clone(*loopUser, mapping);
      loopUser->replaceAllUsesWith(newUser);
      loopUser->erase();
    }
  }

  Operation *createPostLoopReduce(OpBuilder &builder, scf::ForOp &loop,
                                  triton::ReduceOp &reduce) const {
    auto resultIndex =
        loop.getBody()->getNumArguments() - 1 - loop.getNumInductionVars();
    auto newLoopResult = loop.getResult(resultIndex);
    builder.setInsertionPointAfter(loop);
    IRMapping mapping;
    mapping.map(*(reduce.getOperands().begin()), newLoopResult);
    auto newReduce2 = cloneWithInferType(builder, &(*reduce), mapping);
    return newReduce2;
  }

  Operation *createYield(OpBuilder &builder, scf::ForOp &loop,
                         scf::YieldOp &oldYield, Value newUpdate,
                         int oldAccumBlockArgNum) const {
    builder.setInsertionPoint(oldYield);
    SmallVector<Value> yieldValues = llvm::to_vector(oldYield.getOperands());
    yieldValues[oldAccumBlockArgNum - 1] =
        loop.getBody()->getArgument(oldAccumBlockArgNum);
    yieldValues.push_back(newUpdate);
    auto newYield =
        builder.create<scf::YieldOp>(oldYield.getLoc(), yieldValues);
    return newYield;
  }

  Operation *createUpdate(OpBuilder &builder, scf::ForOp &loop,
                          Operation *newReduce, Operation *oldUpdate) const {
    auto blockArgNum = loop.getBody()->getNumArguments() - 1;
    auto newArg = loop.getBody()->getArgument(blockArgNum);
    builder.setInsertionPointAfter(newReduce);
    IRMapping mapping;
    mapping.map(oldUpdate->getOperand(0), newArg);
    mapping.map(oldUpdate->getOperand(1), newReduce->getResult(0));
    auto newUpdate = cloneWithInferType(builder, oldUpdate, mapping);
    return newUpdate;
  }

  Operation *createReduce(OpBuilder &builder, triton::ReduceOp reduce,
                          Type viewOpTensorType) const {
    auto srcType = reduce.getOperands()[0].getType().cast<RankedTensorType>();
    auto rank = srcType.getShape().size();
    builder.setInsertionPointAfter(reduce);
    IRMapping mapping;
    for (auto operand : reduce.getOperands()) {
      auto viewOp = builder.create<triton::ViewOp>(reduce.getLoc(),
                                                   viewOpTensorType, operand);
      mapping.map(operand, viewOp);
    }

    auto newReduce = cloneWithInferType(builder, &(*reduce), mapping);
    newReduce->setAttr("axis", builder.getI32IntegerAttr(rank));
    auto typeInfer = dyn_cast<InferTypeOpInterface>(newReduce);
    if (typeInfer) {
      SmallVector<Type, 1> newTypes;
      auto success = typeInfer.inferReturnTypes(
          newReduce->getContext(), newReduce->getLoc(),
          newReduce->getOperands(), newReduce->getAttrDictionary(),
          newReduce->getPropertiesStorage(), newReduce->getRegions(), newTypes);
      if (succeeded(success)) {
        for (size_t i = 0; i < newTypes.size(); i++)
          newReduce->getResult(i).setType(newTypes[i]);
      }
    }
    return newReduce;
  }

  Operation *createAccum(OpBuilder &builder, Value &oldAccum,
                         SmallVector<int64_t> &shape,
                         Attribute &slice2d) const {
    // Drop the last dimension (thread locality dimension)
    SmallVector<int64_t> accumShape(shape.begin(), shape.end() - 1);
    // Create tensor type for the new accumulator
    auto accumType = RankedTensorType::get(
        accumShape,
        oldAccum.getType().cast<RankedTensorType>().getElementType(), slice2d);
    // Create new accumulator
    builder.setInsertionPointAfter(oldAccum.getDefiningOp());
    auto newAccum = builder.create<arith::ConstantOp>(
        oldAccum.getLoc(), accumType, builder.getZeroAttr(accumType));
    return newAccum;
  }

  SmallVector<int64_t>
  getThreadLocalityOptimizedShape(triton::ReduceOp reduce) const {
    auto srcType = reduce.getOperands()[0].getType().cast<RankedTensorType>();
    auto srcShape = srcType.getShape();
    auto rank = srcShape.size();
    auto elemsPerThread =
        triton::gpu::getElemsPerThread(srcType)[reduce.getAxis()];
    auto viewOpTensorShape = insertValue(srcShape, rank, 1);
    viewOpTensorShape[reduce.getAxis()] /= elemsPerThread;
    viewOpTensorShape[rank] = elemsPerThread;
    return viewOpTensorShape;
  }

  Attribute getThreadLocalityOptimizedEncoding(triton::ReduceOp reduce) const {
    auto srcType = reduce.getOperands()[0].getType().cast<RankedTensorType>();
    auto rank = srcType.getShape().size();
    auto srcEncoding = srcType.getEncoding();
    auto blocked = srcEncoding.dyn_cast<triton::gpu::BlockedEncodingAttr>();
    auto sizePerThread3d =
        insertValue(blocked.getSizePerThread(), rank,
                    blocked.getSizePerThread()[reduce.getAxis()]);
    sizePerThread3d[reduce.getAxis()] = 1;
    auto threadsPerWarp3d = insertValue(blocked.getThreadsPerWarp(), rank, 1);
    auto warsPerCTA3d = insertValue(blocked.getWarpsPerCTA(), rank, 1);
    auto order3d = insertValue(blocked.getOrder(), 0, 2);
    auto ctasPerCGA3d =
        insertValue(blocked.getCTALayout().getCTAsPerCGA(), rank, 1);
    auto ctasSplitNum3d =
        insertValue(blocked.getCTALayout().getCTASplitNum(), rank, 1);
    auto ctaOrder3d =
        insertValue(blocked.getCTALayout().getCTAOrder(), rank, 2);
    auto ctaLayout3d = triton::gpu::CTALayoutAttr::get(
        reduce.getContext(), ctasPerCGA3d, ctasSplitNum3d, ctaOrder3d);
    auto blocked3d = triton::gpu::BlockedEncodingAttr::get(
        reduce.getContext(), sizePerThread3d, threadsPerWarp3d, warsPerCTA3d,
        order3d, ctaLayout3d);
    return blocked3d;
  }

  template <typename T>
  SmallVector<T> insertValue(ArrayRef<T> vec, unsigned index, int value) const {
    SmallVector<T> res(vec.begin(), vec.end());
    res.insert(res.begin() + index, static_cast<T>(value));
    return res;
  }
};

std::unique_ptr<Pass> mlir::createTritonGPUOptimizeThreadLocalityPass() {
  return std::make_unique<TritonGPUOptimizeThreadLocalityPass>();
}
