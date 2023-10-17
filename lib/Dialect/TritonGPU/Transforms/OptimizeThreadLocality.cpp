#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include <memory>
#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

using namespace mlir;

class TritonGPUOptimizeThreadLocalityPass
    : public TritonGPUOptimizeThreadLocalityBase<
          TritonGPUOptimizeThreadLocalityPass> {

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    mod.walk([&](triton::ReduceOp reduce) -> void {
      OpBuilder builder(reduce);
      reduce.dump();
      auto srcType = reduce.getOperands()[0].getType().cast<RankedTensorType>();
      auto srcShape = srcType.getShape();
      auto srcEncoding = srcType.getEncoding();
      if (!srcEncoding.isa<triton::gpu::BlockedEncodingAttr>())
        return;
      auto blocked = srcEncoding.dyn_cast<triton::gpu::BlockedEncodingAttr>();

      auto elemsPerThread = triton::gpu::getTotalElemsPerThread(
          blocked, srcType.getShape(), srcType.getElementType());
      auto axisElems = srcType.getShape()[reduce.getAxis()];
      std::cout << "elemsPerThread: " << elemsPerThread << std::endl;
      std::cout << "axisElems: " << axisElems << std::endl;
      if (axisElems != elemsPerThread)
        return;
      auto rank = srcShape.size();
      SmallVector<int64_t> newShape(srcShape.begin(), srcShape.end());
      newShape[reduce.getAxis()] = 1;
      newShape.push_back(elemsPerThread);
      auto sizePerThread = getSizePerThread(blocked);
      sizePerThread.insert(sizePerThread.begin() + reduce.getAxis(), 1);
      auto threadsPerWarp = getThreadsPerWarp(blocked);
      threadsPerWarp.insert(threadsPerWarp.begin() + reduce.getAxis(), 1);
      auto warpsPerCTA = getWarpsPerCTA(blocked);
      warpsPerCTA.insert(warpsPerCTA.begin() + reduce.getAxis(), 1);
      auto order = getOrder(blocked);
      order.insert(order.begin(), 2);

      auto tensorType = RankedTensorType::get(
          newShape, srcType.getElementType(),
          triton::gpu::BlockedEncodingAttr::get(mod.getContext(), sizePerThread,
                                                threadsPerWarp, warpsPerCTA,
                                                order, blocked.getCTALayout()));
      auto viewOp = builder.create<triton::ViewOp>(reduce.getLoc(), tensorType,
                                                   reduce.getOperands()[0]);
      reduce->setOperands({viewOp.getResult()});
      reduce->setAttr("axis", builder.getI32IntegerAttr(rank));
      // auto newReduce = builder.create<triton::ReduceOp>(
      //     reduce.getLoc(), ValueRange{viewOp.getResult()},
      //     2 /* hardcoded for now*/);
      // addNamedAttrs(newReduce, adaptor.getAttributes());

      // auto &newCombineOp = newReduce.getCombineOp();
      // builder.cloneRegionBefore(op.getCombineOp(), newCombineOp,
      //                           newCombineOp.end());
      // reduce.replaceAllUsesWith(newReduce.getResult());
      // reduce.erase();
    });
  };
};

std::unique_ptr<Pass> mlir::createTritonGPUOptimizeThreadLocalityPass() {
  return std::make_unique<TritonGPUOptimizeThreadLocalityPass>();
}
