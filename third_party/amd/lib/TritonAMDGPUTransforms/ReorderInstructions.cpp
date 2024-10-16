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

// Search through block to find earliest insertion point for move op. This can
// be either an atomic op or last usage of source pointer. Search ends when move
// op is encountered.
static llvm::ilist<Operation>::iterator
findEarlyInsertionPoint(Block *block, Operation *move) {
  Value src;
  if (auto ld = dyn_cast<triton::LoadOp>(move))
    src = ld.getPtr();

  auto ipnt = block->end();
  for (auto bi = block->begin(); bi != block->end(); ++bi) {
    auto *op = &*bi;
    if (op == move) // Don't move later than current location
      break;

    op->walk([&](Operation *wop) {
      if (src) {
        // Check for ops accessing src value.
        for (auto opr : wop->getOperands()) {
          if (opr == src)
            ipnt = bi;
        }
      }
      // Atomics used for global synchronization.
      if (isa<triton::AtomicRMWOp, triton::AtomicCASOp>(wop))
        ipnt = bi;
      // Break at barrier
      if (isa<gpu::BarrierOp>(wop))
        ipnt = bi;
      // Break at loops.
      if (isa<scf::ForOp, scf::WhileOp>(wop))
        ipnt = bi;
    });
  }
  return ipnt;
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

    SmallVector<Operation *> moveOps;
    // Move global loads early to prefetch. This may increase register pressure
    // but it enables issuing global loads early.
    m.walk([&](triton::LoadOp op) { moveOps.push_back(op); });
    // Move local_stores early if dependence distance greater than
    // one iteration.
    // Best perf on GEMM when these precede global loads.
    m.walk([&](ttg::LocalStoreOp op) { moveOps.push_back(op); });

    for (auto op : llvm::reverse(moveOps)) {
      // Gather use-def chain in block.
      Block *block = op->getBlock();
      bool leadsToLoad = false;
      SetVector<Operation *> backwardSet;

      BackwardSliceOptions options;
      options.omitBlockArguments = true;
      options.inclusive = false;
      options.filter = [&](Operation *defOp) -> bool {
        Block *defBlock = defOp->getBlock();
        if (!block->findAncestorOpInBlock(*defOp))
          return false;
        // Check for a `load` dependent path.
        leadsToLoad |= isa<triton::LoadOp>(defOp);
        // Only move ops residing in the same block.
        return defBlock == block;
      };
      mlir::getBackwardSlice(op, &backwardSet, options);
      backwardSet.insert(op);

      // Don't move a local_store if its source is a load from
      // the same iteration.
      if (isa<ttg::LocalStoreOp>(op) && leadsToLoad)
        continue;

      auto ipoint = findEarlyInsertionPoint(block, op);
      // Remove ops that already precede the insertion point. This is done
      // before moves happen to avoid `Operation::isBeforeInBlock` N^2
      // complexity.

      SmallVector<Operation *> dfg = backwardSet.takeVector();
      if (ipoint != block->end()) {
        // Move ops to insertion point.
        llvm::erase_if(
            dfg, [&](Operation *op) { return !ipoint->isBeforeInBlock(op); });
        for (auto *dfgop : llvm::reverse(dfg))
          dfgop->moveAfter(block, ipoint);
      } else {
        // Move ops to block begin.
        for (auto *dfgop : llvm::reverse(dfg))
          dfgop->moveBefore(block, block->begin());
      }
    }

    /**
     * Sched-load optimization for matmul kernels with large tile sizes
     * The basic idea of sched-load optimization is to sink the 2nd tt.load
     * after local_load so that global_load instructions can be interleaved with
     * mfma's. This can help hide the issue latency of global_load instructions
     * and improve performance on MI300X.
     *
     * It's assumed that the IR before this optimization has the following
     * structure:
     * ```mlir
     * scf.for ..
     * {
     *   tileA = tt.load a_ptr
     *   tileB = tt.load b_ptr
     *   opA = local_load bufferA
     *   opB = local_load bufferB
     *   res = tt.dot opA, opB
     *   local_store tileA, bufferA
     *   local_store tileB, bufferB
     * }
     * ```
     * After this optimization, the IR is transformed to
     * ```mlir
     * scf.for ..
     * {
     *   tileA = tt.load a_ptr
     *   opA = local_load bufferA
     *   opB = local_load bufferB
     *   tileB = tt.load b_ptr  <-- 2nd tt.load is sinked here
     *   res = tt.dot opA, opB
     *   local_store tileA, bufferA
     *   local_store tileB, bufferB
     * }
     * ```
     * For now, we don't have a perfect hueristic about when should this
     * optimization be applied. Therefore, we implement a simple hueristic that
     * this is applied when the tile size of A and B are large enough, i.e.
     * nonKDim >= 128  and kDim >= 64. And also this is only applied for typical
     * matmul kernels, i.e. only two tt.load's and one dotOp inside the loop. We
     * are experimenting how to better control instruction scheduling and enable
     * such optimizations.
     */
    m.walk([&](scf::ForOp forOp) -> void {
      SetVector<Operation *> loadOps;
      triton::DotOp dotOp;
      int nDotOps = 0;
      for (Operation &op : forOp) {
        if (auto loadOp = dyn_cast<triton::LoadOp>(&op))
          loadOps.insert(loadOp);
        if (auto curOp = dyn_cast<triton::DotOp>(&op)) {
          nDotOps++;
          dotOp = curOp;
        }
      }
      // Only apply the optimization when there are 2 load's and 1 dot in the
      // loop
      if (loadOps.size() != 2 || nDotOps != 1)
        return;
      // Only apply the optimization when tile size is large enough
      // 1. nonKDim >= 128
      // 2. kDim >= 64
      auto ldAOp = dyn_cast<triton::LoadOp>(loadOps[0]);
      auto tileAShape = cast<RankedTensorType>(ldAOp.getType()).getShape();
      auto ldBOp = dyn_cast<triton::LoadOp>(loadOps[1]);
      auto tileBShape = cast<RankedTensorType>(ldBOp.getType()).getShape();
      if (!(tileAShape[0] >= 128 && tileAShape[1] >= 64 &&
            tileBShape[1] >= 128))
        return;
      // move ldBOp right before tt.dot
      loadOps[1]->moveBefore(dotOp);
    });
  }
};

std::unique_ptr<Pass> mlir::createTritonAMDGPUReorderInstructionsPass() {
  return std::make_unique<TritonAMDGPUReorderInstructionsPass>();
}
