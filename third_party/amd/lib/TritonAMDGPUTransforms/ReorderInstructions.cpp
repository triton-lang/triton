#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include <deque>

#define GEN_PASS_CLASSES
#include "TritonAMDGPUTransforms/Passes.h"

using namespace mlir;
namespace ttg = mlir::triton::gpu;
namespace tt = mlir::triton;

static bool isLocalLoadOrDotLayoutConversion(Operation *op) {
  if (isa<ttg::LocalLoadOp>(op))
    return true;
  if (auto cvt = dyn_cast<ttg::ConvertLayoutOp>(op))
    return isa<ttg::DotOperandEncodingAttr>(cvt.getType().getEncoding());
  return false;
}

class TritonAMDGPUReorderInstructionsPass
    : public TritonAMDGPUReorderInstructionsBase<
          TritonAMDGPUReorderInstructionsPass> {
public:
  TritonAMDGPUReorderInstructionsPass() = default;

  Operation *getFirstUse(Operation *op) {
    std::vector<Operation *> users;
    for (auto user : op->getUsers()) {
      if (Operation *ancestor = op->getBlock()->findAncestorOpInBlock(*user))
        users.push_back(ancestor);
    }
    auto minOpIt = std::min_element(users.begin(), users.end(),
                                    [](mlir::Operation *a, mlir::Operation *b) {
                                      return a->isBeforeInBlock(b);
                                    });
    return minOpIt != users.end() ? *minOpIt : nullptr;
  }

  void runOnOperation() override {
    ModuleOp m = getOperation();

    // Sink shared memory loads and layout conversions into loops to decrease
    // register pressure when possible.
    DenseMap<Operation *, Operation *> opToMove;
    m.walk([&](Operation *op) {
      if (!isLocalLoadOrDotLayoutConversion(op))
        return;
      if (!op->hasOneUse())
        return;
      Operation *user = *op->getUsers().begin();
      if (user->getParentOfType<scf::ForOp>() ==
          op->getParentOfType<scf::ForOp>())
        return;
      opToMove.insert({op, user});
    });
    for (auto &kv : opToMove)
      kv.first->moveBefore(kv.second);
    opToMove.clear();

    // Move writing to LDS and reading from LDS right after the loading of a
    // tensor from global memory. There are 2 possible patterns depending on
    // whether writing to LDS is done using an optional local_alloc argument or
    // a local_store instruction:
    //
    // 1) %1 = load %ptr
    //    %2 = local_alloc %1
    //    %3 = local_load %2
    //
    // 2) %1 = load %ptr
    //    %2 = local_alloc
    //    %3 = local_store %1, %2
    //    %4 = local_load %2
    m.walk([&](ttg::LocalLoadOp localLoad) {
      auto localAlloc = localLoad.getSrc().getDefiningOp<ttg::LocalAllocOp>();
      if (!localAlloc)
        return;

      // Case when localAlloc has operands
      if (localAlloc->getNumOperands() == 1) {
        if (!localAlloc->hasOneUse())
          return;
        auto loadOp = localAlloc->getOperand(0).getDefiningOp<tt::LoadOp>();
        if (!loadOp)
          return;
        localAlloc->moveAfter(loadOp);
        localLoad->moveAfter(localAlloc);
        return;
      }

      // Case when localAlloc has no operands
      assert(localAlloc->getNumOperands() < 1);
      auto allocVal = localAlloc->getResult(0);

      // Check if the localAlloc has exactly two uses (localStore and localLoad)
      int numUses = std::distance(allocVal.use_begin(), allocVal.use_end());
      if (numUses != 2)
        return;

      // localStore comes before localLoad in block.
      Operation *localStore = getFirstUse(localAlloc);
      if (!isa<ttg::LocalStoreOp>(localStore))
        return;

      auto loadOp = localStore->getOperand(0).getDefiningOp<tt::LoadOp>();
      if (!loadOp)
        return;
      localAlloc->moveAfter(loadOp);
      localStore->moveAfter(localAlloc);
      localLoad->moveAfter(localStore);
    });

    // Sink conversion after the last dealloc but before the first use ancestor
    // in its block. This helps to avoid unnecessary shared memory allocation.
    m.walk([&](triton::gpu::ConvertLayoutOp op) {
      auto curr = mlir::Block::iterator(op);
      for (; &*curr != getFirstUse(op); curr++)
        if (isa<triton::gpu::LocalDeallocOp>(&*curr))
          op->moveAfter(&*curr);
    });

    // Move transpositions just after their definition.
    m.walk([&](triton::TransOp op) {
      if (Operation *argOp = op.getSrc().getDefiningOp())
        op->moveAfter(argOp);
    });
  }
};

std::unique_ptr<Pass> mlir::createTritonAMDGPUReorderInstructionsPass() {
  return std::make_unique<TritonAMDGPUReorderInstructionsPass>();
}
