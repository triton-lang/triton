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

// Replace ForOp with a new ForOp with extra operands. The YieldOp is not
// updated and needs to be updated separatly for the loop to be correct.
static scf::ForOp replaceForOpWithNewSignature(OpBuilder &rewriter,
                                               scf::ForOp loop,
                                               ValueRange newIterOperands) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(loop);

  // Create a new loop before the existing one, with the extra operands.
  // rewriter.setInsertionPoint(loop);
  auto operands = llvm::to_vector<4>(loop.getInitArgs());
  operands.append(newIterOperands.begin(), newIterOperands.end());
  scf::ForOp newLoop = rewriter.create<scf::ForOp>(
      loop.getLoc(), loop.getLowerBound(), loop.getUpperBound(), loop.getStep(),
      operands);
  newLoop.getBody()->erase();
  newLoop.getRegion().getBlocks().splice(
      newLoop.getRegion().getBlocks().begin(), loop.getRegion().getBlocks());
  for (Value operand : newIterOperands)
    newLoop.getBody()->addArgument(operand.getType(), operand.getLoc());

  for (auto it : llvm::zip(loop.getResults(), newLoop.getResults().take_front(
                                                  loop.getNumResults())))
    std::get<0>(it).replaceAllUsesWith(std::get<1>(it));
  return newLoop;
}

class TritonGPUOptimizeThreadLocalityPass
    : public TritonGPUOptimizeThreadLocalityBase<
          TritonGPUOptimizeThreadLocalityPass> {
  template <typename T>
  SmallVector<T> insertValue(ArrayRef<T> vec, unsigned index, int value) const {
    SmallVector<T> res(vec.begin(), vec.end());
    res.insert(res.begin() + index, static_cast<T>(value));
    return res;
  }
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    int counter = 0;
    DenseSet<triton::ReduceOp> reduceOps;
    mod.walk([&](triton::ReduceOp reduce) -> void {
      if (counter > 0)
        return;
      counter++;
      std::cout << "reduce op found" << std::endl;
      reduceOps.insert(reduce);
    });
    for (auto reduce : reduceOps) {
      OpBuilder builder(reduce);
      auto srcType = reduce.getOperands()[0].getType().cast<RankedTensorType>();
      auto srcShape = srcType.getShape();
      auto srcEncoding = srcType.getEncoding();
      if (!srcEncoding.isa<triton::gpu::BlockedEncodingAttr>())
        return;
      auto blocked = srcEncoding.dyn_cast<triton::gpu::BlockedEncodingAttr>();
      auto elemsPerThread =
          triton::gpu::getElemsPerThread(srcType)[reduce.getAxis()];
      auto rank = srcShape.size();
      // create new layouts
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
          mod.getContext(), ctasPerCGA3d, ctasSplitNum3d, ctaOrder3d);
      auto blocked3d = triton::gpu::BlockedEncodingAttr::get(
          mod.getContext(), sizePerThread3d, threadsPerWarp3d, warsPerCTA3d,
          order3d, ctaLayout3d);
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
      assert(operandNumber == 1);
      auto firstOperand = user->getOperand(0);
      assert(firstOperand.isa<BlockArgument>());
      auto blockArg = firstOperand.dyn_cast<BlockArgument>();
      auto blockArgNum = blockArg.getArgNumber();
      auto forOp = dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp());

      // get initArg
      auto initArg =
          forOp.getInitArgs()[blockArgNum - forOp.getNumInductionVars()];
      // create newAccum initialization
      SmallVector<int64_t> accumShape(srcShape.begin(), srcShape.end());
      accumShape[reduce.getAxis()] /= elemsPerThread;
      auto accumType = RankedTensorType::get(
          accumShape,
          initArg.getType().cast<RankedTensorType>().getElementType(), slice2d);
      builder.setInsertionPointAfter(initArg.getDefiningOp());
      auto newAccum = builder.create<arith::ConstantOp>(
          initArg.getLoc(), accumType, builder.getZeroAttr(accumType));
      // get old loop user
      OpOperand &loopUse = *(forOp.getResult(0).getUses().begin());
      auto loopUser = loopUse.getOwner();
      // get old loop yield
      auto oldYield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
      // create new loop
      auto newLoop = replaceForOpWithNewSignature(
          builder, forOp, ValueRange{newAccum.getResult()});
      // create new reduce
      auto viewOpTensorShape = insertValue(srcShape, rank, 1);
      viewOpTensorShape[reduce.getAxis()] /= elemsPerThread;
      viewOpTensorShape[rank] = elemsPerThread;
      auto viewOpTensorType = RankedTensorType::get(
          viewOpTensorShape, srcType.getElementType(), blocked3d);
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
            newReduce->getPropertiesStorage(), newReduce->getRegions(),
            newTypes);
        if (succeeded(success)) {
          for (size_t i = 0; i < newTypes.size(); i++)
            newReduce->getResult(i).setType(newTypes[i]);
        }
      }
      // create new accum update
      auto newArg = newLoop.getBody()->getArgument(blockArgNum + 1);
      builder.setInsertionPointAfter(user);
      auto newUpdate = builder.create<arith::AddFOp>(reduce.getLoc(), newArg,
                                                     newReduce->getResult(0));
      // create new yield
      builder.setInsertionPointToEnd(newLoop.getBody());
      auto newLoopBlockArg = newLoop.getBody()->getArgument(blockArgNum);
      SmallVector<Value> yieldValues = {newLoopBlockArg, newUpdate};
      auto newYield =
          builder.create<scf::YieldOp>(newLoop.getLoc(), yieldValues);
      // Add one more reduce after loop
      auto newLoopResult = newLoop.getResult(1);
      builder.setInsertionPointAfter(newLoop);
      IRMapping newMapping;
      newMapping.map(*(reduce.getOperands().begin()), newLoopResult);
      auto newReduce2 = cloneWithInferType(builder, &(*reduce), newMapping);
      // Replace loop user with new reduce
      builder.setInsertionPointAfter(loopUser);
      IRMapping newMapping1;
      newMapping1.map(loopUser->getOperands()[0], newReduce2->getResult(0));
      auto finalOp = builder.clone(*loopUser, newMapping1);
      // Replace uses of loopUser with finalOp
      loopUser->replaceAllUsesWith(finalOp);

      oldYield.erase();
      forOp.erase();
      user->erase();
      reduce.erase();
      loopUser->erase();
    }
    // std::cout << "Printing module after pass" << std::endl;
    // mod.dump();
  };
};

std::unique_ptr<Pass> mlir::createTritonGPUOptimizeThreadLocalityPass() {
  return std::make_unique<TritonGPUOptimizeThreadLocalityPass>();
}
