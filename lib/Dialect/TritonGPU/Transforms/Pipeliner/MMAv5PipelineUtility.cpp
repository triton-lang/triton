#include "triton/Dialect/TritonGPU/Transforms/MMAv5PipelineUtility.h"

#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

ttng::TMEMAllocOp ttng::createTMemAlloc(OpBuilder &builder,
                                        ttng::TMEMAllocOp oldTMemAllocOp,
                                        bool multiBufferred, int numStages) {
  Location loc = oldTMemAllocOp.getLoc();
  auto oldRetType = oldTMemAllocOp.getType();
  SmallVector<int64_t> shape = {oldRetType.getShape().begin(),
                                oldRetType.getShape().end()};
  if (multiBufferred) {
    shape.insert(shape.begin(), numStages);
  }
  Type accMemDescType = triton::gpu::MemDescType::get(
      shape, oldRetType.getElementType(), oldRetType.getEncoding(),
      oldRetType.getMemorySpace(), /*mutableMemory=*/true);
  return builder.create<ttng::TMEMAllocOp>(oldTMemAllocOp.getLoc(),
                                           accMemDescType, nullptr);
}

std::optional<std::pair<ttng::TMEMAllocOp, ttng::TMEMLoadOp>>
ttng::getTMemAllocAndLoad(ttng::MMAv5OpInterface mmaOp) {
  auto acc = mmaOp->getOperand(2).getDefiningOp<ttng::TMEMAllocOp>();
  if (!acc || acc->getParentRegion() != mmaOp->getParentRegion()) {
    return std::nullopt;
  }
  for (auto user : acc->getUsers()) {
    if (auto load = dyn_cast<ttng::TMEMLoadOp>(user)) {
      if (load->getParentRegion() == mmaOp->getParentRegion()) {
        return std::make_pair(acc, load);
      }
    }
  }
  return std::nullopt;
}
