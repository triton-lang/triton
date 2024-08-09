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
findEarlyInsertionPoint(Block *block, Operation *move, bool &found) {
  Value src;
  if (auto ld = dyn_cast<triton::LoadOp>(move))
    src = ld.getPtr();

  found = false;
  auto ipnt = block->begin();
  for (auto bi = ipnt; bi != block->end(); ++bi) {
    auto *op = &*bi;
    if (op == move) // Don't move later than current location
      break;

    op->walk([&](Operation *wop) {
      if (src) {
        // Check for ops accessing src value.
        for (auto opr : wop->getOperands()) {
          if (opr == src) {
            ipnt = bi;
            found = true;
          }
        }
      }
      // Atomics used for global synchronization.
      if (isa<triton::AtomicRMWOp, triton::AtomicCASOp>(wop)) {
        ipnt = bi;
        found = true;
      }
      // Break at loops.
      if (isa<scf::ForOp, scf::WhileOp>(wop)) {
        ipnt = bi;
        found = true;
      }
    });
  }
  return ipnt;
}

class TritonAMDGPUReorderInstructionsPass
    : public TritonAMDGPUReorderInstructionsBase<
          TritonAMDGPUReorderInstructionsPass> {
public:
  TritonAMDGPUReorderInstructionsPass() = default;

  void runOnOperation() override {
    ModuleOp m = getOperation();
    mlir::DominanceInfo dom(m);

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

    for (auto op : moveOps) {
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

      bool found;
      auto ipoint = findEarlyInsertionPoint(block, op, found);
      // Remove ops that already precede the insertion point. This is done
      // before moves happen to avoid `Operation::isBeforeInBlock` N^2
      // complexity.
      if (!found && backwardSet.contains(&*block->begin()))
        found = true;

      SmallVector<Operation *> dfg = backwardSet.takeVector();
      // llvm::erase_if(
      //     dfg, [&](Operation *op) { return !ipoint->isBeforeInBlock(op); });
      // Move ops to insertion point.
      for (auto *dfgop : llvm::reverse(dfg))
        dfgop->moveAfter(block, ipoint);
      if (!found)
        block->begin()->moveAfter(op);
    }
  }
};

std::unique_ptr<Pass> mlir::createTritonAMDGPUReorderInstructionsPass() {
  return std::make_unique<TritonAMDGPUReorderInstructionsPass>();
}
