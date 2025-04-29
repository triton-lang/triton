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
    mod.walk([&](Operation *op) {
      bool isMMAv3 = isa<ttng::WarpGroupDotOp>(op);
      if (!isMMAv3 && !isa<ttng::MMAv5OpInterface>(op))
        return WalkResult::advance();
      OpBuilder builder(op);
      auto a = op->getOperand(0);
      auto b = op->getOperand(1);
      if (isMMAv3) {
        auto mmaEncoding = dyn_cast<ttg::NvidiaMmaEncodingAttr>(
            cast<RankedTensorType>(op->getResult(0).getType()).getEncoding());
        if (!mmaEncoding || !mmaEncoding.isHopper())
          return WalkResult::advance();
      }
      bool aDependsOnShared = dependOnCopyRegToShared(a);
      bool bDependsOnShared = dependOnCopyRegToShared(b);
      if (!aDependsOnShared && !bDependsOnShared)
        return WalkResult::advance();
      Operation *fence = builder.create<ttng::FenceAsyncSharedOp>(
          op->getLoc(), /*bCluster=*/false);
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
    // support ForOp only
    if (auto forOp = dyn_cast<scf::ForOp>(argOwner)) {
      // prologue
      auto iterOperands = forOp.getInitArgs();
      if (argNum == 0)
        return false;
      if (dependOnCopyRegToShared(iterOperands[argNum - 1], visited))
        return true;
      // yield
      auto yieldOp = forOp.getBody()->getTerminator();
      Value v = yieldOp->getOperand(argNum - 1);
      auto entry = std::make_pair<Operation *, unsigned>(std::move(yieldOp),
                                                         std::move(argNum));
      if (dependOnCopyRegToShared(v, visited))
        return true;
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
