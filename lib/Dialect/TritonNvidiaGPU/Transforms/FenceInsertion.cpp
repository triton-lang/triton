#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/Support/Debug.h"

//===----------------------------------------------------------------------===//
//
// This pass works after all other passes, inserting fences to ensure that
// memory operations are properly ordered across generic and async proxy.
//
//===----------------------------------------------------------------------===//

namespace ttg = mlir::triton::gpu;

namespace mlir {
namespace triton {
namespace nvidia_gpu {

#define GEN_PASS_DEF_TRITONGPUFENCEINSERTION
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

struct FenceInsertionPass
    : public impl::TritonGPUFenceInsertionBase<FenceInsertionPass> {

public:
  using impl::TritonGPUFenceInsertionBase<
      FenceInsertionPass>::TritonGPUFenceInsertionBase;
  // TODO: support more general patterns to insert fences. eg. any op(generic)
  // to shared in use-def chain which refers by async proxy. We have generic(
  // convertlayout with sts/stmatix) + fence + async(wgmma) up to now
  void runOnOperation() override {
    // Only insert fences for compute capability 9.0
    if (computeCapability < 90)
      return;
    ModuleOp mod = getOperation();
    mod.walk([&](DotOpInterface dotOp) {
      Value a = dotOp.getA();
      Value b = dotOp.getB();
      SmallVector<Operation *> copyRegToSharedOpsA = findCopyRegToSharedOps(a);
      SmallVector<Operation *> copyRegToSharedOpsB = findCopyRegToSharedOps(b);
      if (copyRegToSharedOpsA.empty() && copyRegToSharedOpsB.empty())
        return WalkResult::advance();

      OpBuilder builder(dotOp);
      auto fence = builder.create<FenceAsyncSharedOp>(dotOp.getLoc(),
                                                      /*bCluster=*/false);
      // If there is all the dependencies are outside of the loop try to hoist
      // the fence.
      while (auto loopOp = fence->getParentOfType<LoopLikeOpInterface>()) {
        if (!copyRegToSharedOpsA.empty() &&
            llvm::any_of(copyRegToSharedOpsA,
                         [&](Operation *op) { return loopOp->isAncestor(op); }))
          break;
        if (!copyRegToSharedOpsB.empty() &&
            llvm::any_of(copyRegToSharedOpsB,
                         [&](Operation *op) { return loopOp->isAncestor(op); }))
          break;
        loopOp.moveOutOfLoop(fence);
      }

      // If the previous op is already a fence, this one isn't needed.
      if (auto lastFence =
              dyn_cast_or_null<FenceAsyncSharedOp>(fence->getPrevNode())) {
        if (lastFence.getBCluster() == fence.getBCluster())
          fence.erase();
      }

      return WalkResult::advance();
    });
  }

private:
  // Return true if the operand depends on a copy from register to shared.
  SmallVector<Operation *> findCopyRegToSharedOps(Value operand) {
    DenseSet<Value> visited;
    llvm::SetVector<Operation *> result;
    findCopyRegToSharedOps(operand, visited, result);
    return result.takeVector();
  }

  void findCopyRegToSharedOps(Value operand, DenseSet<Value> &visited,
                              llvm::SetVector<Operation *> &result) {
    // If the value has already been visited we can safely return false as we
    // would early return when true.
    if (visited.count(operand))
      return;
    visited.insert(operand);
    if (!isa<triton::gpu::MemDescType>(operand.getType()))
      return;

    auto op = operand.getDefiningOp();
    if (op) {
      // reach an alloc copying from register, we need a fence.
      if (auto localAlloc = dyn_cast<ttg::LocalAllocOp>(op)) {
        if (localAlloc.getSrc()) {
          result.insert(op);
        }
        // Check if there are local_store ops that write to that buffer.
        for (auto user : localAlloc.getResult().getUsers()) {
          while (user->hasOneUse() &&
                 user->hasTrait<OpTrait::MemDescViewTrait>()) {
            user = *user->getUsers().begin();
          }
          if (isa<ttg::LocalStoreOp>(user)) {
            result.insert(user);
            return;
          }
        }
      }
      // if it is not an alloc, iterate over the operands.
      for (auto v : op->getOperands()) {
        findCopyRegToSharedOps(v, visited, result);
      }
      return;
    }

    // reach BlockArgument
    BlockArgument arg = cast<BlockArgument>(operand);
    unsigned argNum = arg.getArgNumber();
    Operation *argOwner = arg.getOwner()->getParentOp();
    // look through ForOp iter argument
    if (auto forOp = dyn_cast<scf::ForOp>(argOwner)) {
      assert(argNum != 0 && "induction var cannot be memdesc type");
      --argNum;
      // prologue
      findCopyRegToSharedOps(forOp.getInitArgs()[argNum], visited, result);
      // yield
      auto yieldOp = forOp.getBody()->getTerminator();
      Value v = yieldOp->getOperand(argNum);
      findCopyRegToSharedOps(v, visited, result);
      return;
    }

    // look through `ttg.warp_specialize`.
    if (auto wsOp = dyn_cast<ttg::WarpSpecializePartitionsOp>(argOwner)) {
      findCopyRegToSharedOps(wsOp.getParentOp().getExplicitCaptures()[argNum],
                             visited, result);
      return;
    }

    // Conservatively return true for other ops
    result.insert(argOwner);
  }
};

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
