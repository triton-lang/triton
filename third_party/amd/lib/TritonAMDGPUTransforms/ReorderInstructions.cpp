#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
namespace ttg = mlir::triton::gpu;

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

// Return true if the given moduleOp contains a pure matmul problem; i.e.,
// single dot in the main loop.
static bool isPureMatmulProblem(ModuleOp moduleOp) {
  for (auto forOp : moduleOp.getOps<scf::ForOp>()) {
    int counter = 0;
    forOp.walk([&counter](triton::DotOp dotOp) { ++counter; });
    if (counter != 1)
      return false;
  }
  return true;
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

// Check if the operation opInsideLoop is inside any scf::ForOp and
// opOutsideLoop is not inside the same loop.
static bool isCrossLoopBoundary(mlir::Operation *opInsideLoop,
                                mlir::Operation *opOutsideLoop) {
  scf::ForOp parentForOp = opInsideLoop->getParentOfType<scf::ForOp>();
  return parentForOp && !parentForOp->isAncestor(opOutsideLoop);
}

//===----------------------------------------------------------------------===//
// Reorder mechanisms
//===----------------------------------------------------------------------===//

// Sink dot layout conversions into loops to decrease register pressure when
// possible.
static void sinkDotConversion(ModuleOp moduleOp) {
  DenseMap<Operation *, Operation *> opToMove;
  moduleOp.walk([&](ttg::ConvertLayoutOp op) {
    Attribute encoding = op.getType().getEncoding();
    if (!isa_and_nonnull<ttg::DotOperandEncodingAttr>(encoding))
      return;
    if (!op->hasOneUse())
      return;
    Operation *user = *op->getUsers().begin();
    if (user->getParentOfType<scf::ForOp>() ==
        op->getParentOfType<scf::ForOp>())
      return;
    opToMove[op] = user;
  });

  for (auto &kv : opToMove)
    kv.first->moveBefore(kv.second);
}

// Adjust the placement of shared memory writes and reads to immediately follow
// the definition of their operands in case where shared memory write is in the
// loop but its operand is not.
//
// This is a heuristic driven by optimizing fused attention by hoisting Q tensor
// shared memory read/write operations outside of the loop, as Q is a loop
// invariant and can be loaded once before entering the loop. But it should be
// generally applicable.
//
// There are two possible patterns for this adjustment depending on whether the
// write to shared memory is performed using an optional `local_alloc` argument
// or a `local_store` instruction.
//
// 1) %1 = some_op ... (typically a load or an operation that scales the tensor
//                      after loading)
//    %2 = local_alloc %1
//    %3 = local_load %2
//
// 2) %1 = some_op ...
//    %2 = local_alloc
//    %3 = local_store %1, %2
//    %4 = local_load %2
static void hoistLocalLoad(ModuleOp moduleOp) {
  moduleOp.walk([&](ttg::LocalLoadOp localLoad) {
    auto localAlloc = localLoad.getSrc().getDefiningOp<ttg::LocalAllocOp>();
    if (!localAlloc)
      return;

    // Case when localAlloc has operands
    if (localAlloc->getNumOperands() == 1) {
      if (!localAlloc->hasOneUse())
        return;

      auto srcTensorOp = localAlloc.getSrc().getDefiningOp();
      // Check if localAlloc is in the loop but it's src tensor defining op is
      // outside of it.
      if (!srcTensorOp || !isCrossLoopBoundary(localAlloc, srcTensorOp))
        return;

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
    Operation *localStore = getFirstUseInSameBlock(localAlloc);
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
}

// Sink conversion after the last dealloc but before the first use in its block.
// This helps to avoid unnecessary shared memory allocation.
static void moveDownCoversion(ModuleOp moduleOp) {
  SmallVector<ttg::ConvertLayoutOp> convertOps;
  moduleOp.walk([&](ttg::ConvertLayoutOp op) { convertOps.push_back(op); });

  for (auto op : convertOps) {
    Operation *user = getFirstUseInSameBlock(op);
    for (auto it = Block::iterator(op), ie = op->getBlock()->end();
         it != ie && &*it != user; ++it)
      if (isa<ttg::LocalDeallocOp>(&*it))
        op->moveAfter(&*it);
  }
}

// Move transpositions just after their definition.
static void moveUpTranspose(ModuleOp moduleOp) {
  SmallVector<triton::TransOp> transOps;
  moduleOp.walk([&](triton::TransOp op) { transOps.push_back(op); });

  for (auto op : transOps)
    if (Operation *argOp = op.getSrc().getDefiningOp())
      op->moveAfter(argOp);
}

// Schedule global load and local store ops for better GEMM performance.
static void scheduleGlobalLoadLocalStore(ModuleOp m) {
  SmallVector<Operation *> moveOps;
  // Move global loads early to prefetch. This may increase register pressure
  // but it enables issuing global loads early.
  m.walk([&](triton::LoadOp op) { moveOps.push_back(op); });
  // Move local_stores early if dependence distance greater than one iteration.
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
}

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

#define GEN_PASS_CLASSES
#include "TritonAMDGPUTransforms/Passes.h"

namespace {
struct TritonAMDGPUReorderInstructionsPass
    : public TritonAMDGPUReorderInstructionsBase<
          TritonAMDGPUReorderInstructionsPass> {
  void runOnOperation() override {
    ModuleOp m = getOperation();

    hoistLocalLoad(m);

    sinkDotConversion(m);
    moveDownCoversion(m);

    moveUpTranspose(m);

    if (isPureMatmulProblem(m))
      scheduleGlobalLoadLocalStore(m);
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createTritonAMDGPUReorderInstructionsPass() {
  return std::make_unique<TritonAMDGPUReorderInstructionsPass>();
}
