#include "triton/Conversion/TritonGPUToLLVM/AllocateSharedMemoryUtility.h"

namespace mlir::triton::gpu {

void attachAllocationSizeAndOffsetAttr(ModuleOp mod,
                                       ModuleAllocation &allocation) {
  MLIRContext *ctx = mod.getContext();

  mod.walk<mlir::WalkOrder::PreOrder>([&](FunctionOpInterface funcOp) {
    auto *funcAllocation = allocation.getFuncData(funcOp);
    funcOp.walk([&](Operation *op) {
      auto oBufferId = funcAllocation->getBufferId(op);
      int offset = -1;
      if (oBufferId != Allocation::InvalidBufferId)
        offset = funcAllocation->getOffset(oBufferId);
      else if (op->getNumResults() == 1) {
        Value value = op->getResult(0);
        auto vBufferId = funcAllocation->getBufferId(value);
        if (vBufferId != Allocation::InvalidBufferId)
          offset = funcAllocation->getOffset(vBufferId);
      }
      if (offset == -1)
        return;
      op->setAttr("allocation.offset",
                  IntegerAttr::get(IntegerType::get(ctx, 32), offset));
    });
    return WalkResult::skip();
  });
  mod->setAttr("ttg.shared",
               mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32),
                                      allocation.getSharedMemorySize()));
}

} // namespace mlir::triton::gpu
