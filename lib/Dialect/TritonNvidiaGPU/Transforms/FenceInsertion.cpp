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

using namespace mlir;
namespace tt = ::mlir::triton;
namespace ttg = ::mlir::triton::gpu;
namespace ttng = ::mlir::triton::nvidia_gpu;

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {

struct FenceInsertionPass
    : public TritonGPUFenceInsertionBase<FenceInsertionPass> {

public:
  FenceInsertionPass() = default;
  FenceInsertionPass(int computeCapability) {
    this->computeCapability = computeCapability;
  }
  // TODO: support more general patterns to insert fences. eg. any op(generic)
  // to shared in use-def chain which refers by async proxy. We have generic(
  // convertlayout with sts/stmatix) + fence + async(wgmma) up to now
  void runOnOperation() override {
    // Only insert fences for compute capability 9.0
    if (computeCapability < 90)
      return;
    ModuleOp mod = getOperation();
    mod.walk([&](tt::DotOpInterface dotOp) {
      Value a = dotOp.getA();
      Value b = dotOp.getB();
      bool aDependsOnShared = dependOnCopyRegToShared(a);
      bool bDependsOnShared = dependOnCopyRegToShared(b);
      if (!aDependsOnShared && !bDependsOnShared)
        return WalkResult::advance();

      OpBuilder builder(dotOp);
      auto fence = builder.create<ttng::FenceAsyncSharedOp>(dotOp.getLoc(),
                                                            /*bCluster=*/false);
      // If there is all the dependencies are outside of the loop try to hoist
      // the fence.
      while (auto loopOp = fence->getParentOfType<LoopLikeOpInterface>()) {
        if (aDependsOnShared &&
            loopOp->isAncestor(a.getParentBlock()->getParentOp()))
          break;
        if (bDependsOnShared &&
            loopOp->isAncestor(b.getParentBlock()->getParentOp()))
          break;
        loopOp.moveOutOfLoop(fence);
      }

      // If the previous op is already a fence, this one isn't needed.
      if (auto lastFence = dyn_cast_or_null<ttng::FenceAsyncSharedOp>(
              fence->getPrevNode())) {
        if (lastFence.getBCluster() == fence.getBCluster())
          fence.erase();
      }

      return WalkResult::advance();
    });
  }

private:
  // Return true if the operand depends on a copy from register to shared.
  bool dependOnCopyRegToShared(Value operand) {
    DenseSet<Value> visited;
    return dependOnCopyRegToShared(operand, visited);
  }

  bool dependOnCopyRegToShared(Value operand, DenseSet<Value> &visited) {
    // If the value has already been visited we can safely return false as we
    // would early return when true.
    if (visited.count(operand))
      return false;
    visited.insert(operand);
    if (!isa<triton::gpu::MemDescType>(operand.getType()))
      return false;

    auto op = operand.getDefiningOp();
    if (op) {
      // reach an alloc copying from register, we need a fence.
      if (isa<ttg::LocalAllocOp>(op) && cast<ttg::LocalAllocOp>(op).getSrc())
        return true;
      // if it is not an alloc, iterate over the operands.
      for (auto v : op->getOperands()) {
        if (dependOnCopyRegToShared(v))
          return true;
      }
      return false;
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
      if (dependOnCopyRegToShared(forOp.getInitArgs()[argNum], visited))
        return true;
      // yield
      auto yieldOp = forOp.getBody()->getTerminator();
      Value v = yieldOp->getOperand(argNum);
      return dependOnCopyRegToShared(v, visited);
    }

    // look through `ttg.warp_specialize`.
    if (auto wsOp = dyn_cast<ttg::WarpSpecializePartitionsOp>(argOwner)) {
      return dependOnCopyRegToShared(
          wsOp.getParentOp().getExplicitCaptures()[argNum]);
    }

    // Conservatively return true for other ops
    return true;
  }
};
} // namespace

std::unique_ptr<Pass>
mlir::createTritonNvidiaGPUFenceInsertionPass(int computeCapability) {
  return std::make_unique<FenceInsertionPass>(computeCapability);
}
