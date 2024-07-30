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

// Gather cone of data flow graph (DFG) from the op's basic block.
// - Collect dfg breadth first to keep relative order and reverse order for
//   insertion after. An op may be captured multiple times if DFG reconverges
//   and it will be moved multiple times to keep dominance correctness.
// - Returns bool if this DFG leads to a load op. This condition is not
//   desirable for moving ttg.local_store ops early.
static bool gatherDFG(Operation *seedOp, Block *block,
                      SmallVector<Operation *> &dfg) {
  bool leadsToLoad = false;

  std::deque<Operation *> ops = {seedOp};
  auto checkOperands = [&](Operation *op) {
    for (auto operand : op->getOperands()) {
      if (Operation *defOp = operand.getDefiningOp()) {
        Block *defBlock = defOp->getBlock();
        if (block->findAncestorOpInBlock(*defOp)) {
          // We only move ops residing in the same block.
          if (defBlock == block)
            dfg.push_back(defOp);
          ops.push_back(defOp);
          leadsToLoad |= isa<triton::LoadOp>(defOp);
        } else {
          // Otherwise it should always be in the parent block.
          assert(defBlock->findAncestorOpInBlock(*block->getParentOp()));
        }
      }
    }
  };

  while (!ops.empty()) {
    Operation *op = ops.front();
    ops.pop_front();
    // check next op and sub-regions
    op->walk(checkOperands);
  }
  return leadsToLoad;
}

// Search through block to find earliest insertion point for move op. This can
// be either an atomic op or last usage of source pointer. Search ends when move
// op is encountered.
static llvm::ilist<Operation>::iterator
findEarlyInsertionPoint(Block *block, Operation *move, Value src) {
  auto loc = block->begin();
  for (auto bi = block->begin(); bi != block->end(); ++bi) {
    auto *op = &*bi;
    if (op == move) // Don't move later than current location
      break;
    if (src) {
      // Check for ops accessing src value.
      for (auto opr : op->getOperands()) {
        if (opr == src)
          loc = bi;
      }
    }
    // Atomics used for syncronization?
    op->walk([&](Operation *wop) {
      if (isa<triton::AtomicRMWOp, triton::AtomicCASOp>(wop))
        loc = bi;
      if (isa<scf::ForOp, scf::WhileOp>(wop))
        loc = bi;
    });
  }
  return loc;
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

    // Move LocalLoadOp and LocalAllocOp immediately after their operands.
    // This enables issuing them as early as possible.
    m.walk([&](Operation *op) {
      if (!isa<ttg::LocalLoadOp, ttg::LocalAllocOp>(op) ||
          op->getNumOperands() < 1)
        return;
      if (Operation *argOp = op->getOperand(0).getDefiningOp())
        op->moveAfter(argOp);
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
    // one iteration. Best perf on GEMM when these precede global loads.
    m.walk([&](ttg::LocalStoreOp op) { moveOps.push_back(op); });

    for (auto op : moveOps) {
      // Gather use-def chain in block.
      Block *block = op->getBlock();
      SmallVector<Operation *> dfg = {op};
      bool leadsToLoad = gatherDFG(op, block, dfg);
      if (!isa<ttg::LocalStoreOp>(op) || !leadsToLoad) {
        Value src;
        if (auto ld = dyn_cast<triton::LoadOp>(op))
          src = ld.getPtr();
        auto ip = findEarlyInsertionPoint(block, op, src);
        // Remove ops that already precede the insertion point. This is done
        // before moves happen to avoid `Operation::isBeforeInBlock` N^2
        // complexity.
        llvm::erase_if(dfg,
                       [&](Operation *op) { return !ip->isBeforeInBlock(op); });
        // Move ops to insertion point.
        for (auto *op : dfg)
          op->moveAfter(block, ip);
      }
    }
  }
};

std::unique_ptr<Pass> mlir::createTritonAMDGPUReorderInstructionsPass() {
  return std::make_unique<TritonAMDGPUReorderInstructionsPass>();
}
