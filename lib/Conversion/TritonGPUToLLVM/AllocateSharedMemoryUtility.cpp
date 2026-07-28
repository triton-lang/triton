#include "triton/Conversion/TritonGPUToLLVM/AllocateSharedMemoryUtility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::gpu {

void attachAllocationSizeAndOffsetAttr(ModuleOp mod,
                                       ModuleAllocation &allocation) {
  MLIRContext *ctx = mod.getContext();
  auto i32Ty = IntegerType::get(ctx, 32);

  mod.walk<mlir::WalkOrder::PreOrder>([&](FunctionOpInterface funcOp) {
    auto *funcAllocation = allocation.getFuncData(funcOp);
    funcOp.walk([&](Operation *op) {
      // Handle scratch buffers (from operations like convert_layout)
      auto oBufferId = funcAllocation->getBufferId(op);
      if (oBufferId != Allocation::InvalidBufferId) {
        int offset = funcAllocation->getOffset(oBufferId);
        op->setAttr("allocation.offset", IntegerAttr::get(i32Ty, offset));
        return;
      }

      // Handle explicit buffers (from values like local_alloc results)
      if (op->getNumResults() != 1)
        return;

      Value value = op->getResult(0);
      auto bufferIds = funcAllocation->getBufferIds(value);
      if (bufferIds.empty())
        return;

      // For partitioned tensors, set an array of offsets (one per partition)
      if (bufferIds.size() > 1) {
        SmallVector<Attribute> offsetAttrs;
        for (auto bufferId : bufferIds) {
          int partitionOffset = funcAllocation->getOffset(bufferId);
          offsetAttrs.push_back(IntegerAttr::get(i32Ty, partitionOffset));
        }
        op->setAttr("allocation.offset", ArrayAttr::get(ctx, offsetAttrs));
        return;
      }

      // Standard single offset for non-partitioned tensors
      int offset = funcAllocation->getOffset(bufferIds[0]);
      op->setAttr("allocation.offset", IntegerAttr::get(i32Ty, offset));
    });
    return WalkResult::skip();
  });
  mod->setAttr("ttg.shared",
               mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32),
                                      allocation.getSharedMemorySize()));
}

} // namespace mlir::triton::gpu
