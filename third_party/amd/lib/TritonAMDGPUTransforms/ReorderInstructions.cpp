#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/RegionUtils.h"
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

// Copy of mlir::getBackwardSlice with changes to handle nested regions.
// This is a temporary local fix until these changes are upstreamed to mlir.
static void getDeepBackwardSlice(Operation *op,
                                 SetVector<Operation *> *backwardSlice,
                                 const BackwardSliceOptions &options) {
  if (!op || op->hasTrait<OpTrait::IsIsolatedFromAbove>())
    return;

  // Evaluate whether we should keep this def.
  // This is useful in particular to implement scoping; i.e. return the
  // transitive backwardSlice in the current scope.
  if (options.filter && !options.filter(op))
    return;

  SetVector<Value> usedValues;
  Block *opBlock = op->getBlock();
  auto f = [&](OpOperand *nestedValue) {
    // Filter out values that are not defined in the block
    // that contains 'op'. This is to avoid including values
    // that are defined in the nested regions of 'op'.
    if (auto *nestedOp = nestedValue->get().getDefiningOp()) {
      if (opBlock == nestedOp->getBlock()) {
        usedValues.insert(nestedValue->get());
      }
    }
  };

  // collect all the values used in the nested regions of this op
  // SetVector<Region*> nestedRegions;
  for (auto &region : op->getRegions()) {
    region.walk([&](Region *nestedRegion) {
      mlir::visitUsedValuesDefinedAbove(*nestedRegion, *nestedRegion, f);
    });
  }

  // collect all the values used in the op
  for (const auto &en : llvm::enumerate(op->getOperands())) {
    usedValues.insert(en.value());
  }

  for (const auto &en : llvm::enumerate(usedValues)) {
    auto operand = en.value();
    if (auto *definingOp = operand.getDefiningOp()) {
      if (backwardSlice->count(definingOp) == 0)
        getDeepBackwardSlice(definingOp, backwardSlice, options);
    } else if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
      if (options.omitBlockArguments)
        continue;

      Block *block = blockArg.getOwner();
      Operation *parentOp = block->getParentOp();
      // TODO: determine whether we want to recurse backward into the other
      // blocks of parentOp, which are not technically backward unless they flow
      // into us. For now, just bail.
      if (parentOp && backwardSlice->count(parentOp) == 0) {
        assert(parentOp->getNumRegions() == 1 &&
               parentOp->getRegion(0).getBlocks().size() == 1);
        getDeepBackwardSlice(parentOp, backwardSlice, options);
      }
    } else {
      llvm_unreachable("No definingOp and not a block argument.");
    }
  }

  backwardSlice->insert(op);
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

// Check if the operation opInsideLoop is inside any scf::ForOp and
// opOutsideLoop is not inside the same loop.
bool isCrossLoopBoundary(mlir::Operation *opInsideLoop,
                         mlir::Operation *opOutsideLoop) {
  scf::ForOp parentForOp = opInsideLoop->getParentOfType<scf::ForOp>();
  return parentForOp && !parentForOp->isAncestor(opOutsideLoop);
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

    // Adjust the placement of LDS writes and reads to immediately follow the
    // definition of their operands in case where LDS write is in the
    // loop but it's operand is not. This is a heuristic for optimizing fused
    // attention by hoisting Q tensor LDS read/write operations outside of the
    // loop, as Q is a loop invariant and can be loaded once before entering the
    // loop.
    // There are two possible patterns for this adjustment depending on
    // whether the write to LDS is performed using an optional `local_alloc`
    // argument or a `local_store` instruction.
    //
    // clang-format off
    //
    // 1) %1 = some_op ... (typically a load or an operation that scales the tensor after loading)
    //    %2 = local_alloc %1
    //    %3 = local_load %2
    //
    // 2) %1 = some_op ...
    //    %2 = local_alloc
    //    %3 = local_store %1, %2
    //    %4 = local_load %2
    //
    // clang-format on
    m.walk([&](ttg::LocalLoadOp localLoad) {
      auto localAlloc = localLoad.getSrc().getDefiningOp<ttg::LocalAllocOp>();
      if (!localAlloc)
        return;

      // Case when localAlloc has operands
      if (localAlloc->getNumOperands() == 1) {
        if (!localAlloc->hasOneUse())
          return;

        auto srcTensorOp = localAlloc->getOperand(0).getDefiningOp();
        // Check if localAlloc is in the loop but it's src tensor defining op is
        // outside of it.
        if (!srcTensorOp || !isCrossLoopBoundary(localAlloc, srcTensorOp)) {
          return;
        }

        localAlloc->moveAfter(srcTensorOp);
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

      auto srcTensorOp = localStore->getOperand(0).getDefiningOp();
      // Check if localStore is in the loop but it's src tensor defining op is
      // outside of it.
      if (!srcTensorOp || !isCrossLoopBoundary(localStore, srcTensorOp)) {
        return;
      }

      localAlloc->moveAfter(srcTensorOp);
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
      getDeepBackwardSlice(op, &backwardSet, options);

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
  }
};

std::unique_ptr<Pass> mlir::createTritonAMDGPUReorderInstructionsPass() {
  return std::make_unique<TritonAMDGPUReorderInstructionsPass>();
}
