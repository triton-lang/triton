#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace ttg = mlir::triton::gpu;

namespace mlir {

#define GEN_PASS_DEF_TRITONAMDGPUSINKLAYOUTCONVERSIONS
#include "TritonAMDGPUTransforms/Passes.h.inc"

namespace {

// Return the first user in the same block of the given op. If the user is in a
// nested block then return the op owning the block. Return nullptr if not
// existing.
static Operation *getFirstUseInSameBlock(Operation *op) {
  SmallVector<Operation *> usersInSameBlock;
  for (auto user : op->getUsers()) {
    if (Operation *ancestor = op->getBlock()->findAncestorOpInBlock(*user))
      usersInSameBlock.push_back(ancestor);
  }
  auto minOpIt =
      llvm::min_element(usersInSameBlock, [](Operation *a, Operation *b) {
        return a->isBeforeInBlock(b);
      });
  return minOpIt != usersInSameBlock.end() ? *minOpIt : nullptr;
}

// Sink conversion after the last dealloc but before the first use in its block.
// This helps to avoid unnecessary shared memory allocation.
static void sinkLayoutConversions(triton::FuncOp funcOp) {
  SmallVector<ttg::ConvertLayoutOp> convertOps;
  funcOp.walk([&](ttg::ConvertLayoutOp op) { convertOps.push_back(op); });
  for (auto op : convertOps) {
    Operation *user = getFirstUseInSameBlock(op);
    for (auto it = Block::iterator(op), ie = op->getBlock()->end();
         it != ie && &*it != user; ++it)
      if (isa<ttg::LocalDeallocOp>(&*it))
        op->moveAfter(&*it);
  }
}

} // namespace

struct TritonAMDGPUSinkLayoutConversionsPass
    : public impl::TritonAMDGPUSinkLayoutConversionsBase<
          TritonAMDGPUSinkLayoutConversionsPass> {

  void runOnOperation() override { sinkLayoutConversions(getOperation()); }
};

} // namespace mlir
