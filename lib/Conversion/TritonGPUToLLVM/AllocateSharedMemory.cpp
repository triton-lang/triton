#include "mlir/Pass/Pass.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_ALLOCATESHAREDMEMORY
#include "triton/Conversion/TritonGPUToLLVM/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

struct AllocateSharedMemory
    : public mlir::triton::impl::AllocateSharedMemoryBase<
          AllocateSharedMemory> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = &getContext();
    ModuleAllocation allocation(mod);

    mod.walk([&](Operation *op) {
      if (op->getNumResults() == 0)
        return;
      Value value = op->getResult(0);
      FunctionOpInterface funcOp =
          value.getParentRegion()
              ->template getParentOfType<FunctionOpInterface>();
      auto *funcAllocation = allocation.getFuncData(funcOp);
      auto smem = allocation.getFunctionSharedMemoryBase(funcOp);
      auto bufferId = funcAllocation->getBufferId(value);
      if (bufferId == Allocation::InvalidBufferId)
        return;
      size_t offset = funcAllocation->getOffset(bufferId);
      op->setAttr("allocation.offset",
                  IntegerAttr::get(IntegerType::get(ctx, 32), offset));
    });
  }
};

} // namespace

namespace mlir {

namespace triton {

namespace gpu {

std::unique_ptr<OperationPass<ModuleOp>> createAllocateSharedMemoryPass() {
  return std::make_unique<AllocateSharedMemory>();
}

} // namespace gpu

} // namespace triton

} // namespace mlir
