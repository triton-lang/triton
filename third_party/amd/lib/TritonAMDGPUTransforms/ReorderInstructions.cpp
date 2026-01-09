#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/STLExtras.h"

namespace ttg = mlir::triton::gpu;

namespace mlir {

#define GEN_PASS_DEF_TRITONAMDGPUREORDERINSTRUCTIONS
#include "TritonAMDGPUTransforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

// Search through block to find earliest insertion point for move op. This can
// be either an atomic op or the defining op of source pointer. Search ends when
// move op is encountered.
static llvm::ilist<Operation>::iterator
findEarlyInsertionPoint(Block *block, triton::LoadOp move) {
  Value src = move.getPtr();

  auto ipnt = block->end();
  for (auto bi = block->begin(); bi != block->end(); ++bi) {
    auto *op = &*bi;
    if (op == move) // Don't move later than current location
      break;

    // Check for ops defining the source ptr
    for (auto opr : op->getResults()) {
      if (opr == src) {
        ipnt = bi;
        break;
      }
    }

    // Break at:
    // - Atomics used for global synchronization.
    // - barriers
    // - loops
    if (isa<triton::AtomicRMWOp, triton::AtomicCASOp, gpu::BarrierOp,
            scf::ForOp, scf::WhileOp>(op)) {
      ipnt = bi;
    }
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

//===----------------------------------------------------------------------===//
// Reorder mechanisms
//===----------------------------------------------------------------------===//

// Sink conversion after the last dealloc but before the first use in its block.
// This helps to avoid unnecessary shared memory allocation.
static void moveDownCoversion(triton::FuncOp funcOp) {
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

// Move transpositions just after their definition.
static void moveUpTranspose(triton::FuncOp funcOp) {
  SmallVector<triton::TransposeOpInterface> transOps;
  funcOp.walk([&](triton::TransposeOpInterface op) { transOps.push_back(op); });

  for (auto op : transOps)
    if (Operation *argOp = op.getSrc().getDefiningOp())
      op->moveAfter(argOp);
}

// Schedule global load ops in prologue for better GEMM performance.
static void moveUpGlobalLoadInPrologue(triton::FuncOp funcOp) {
  // Move global_load ops early to prefetch. This may increase
  // register pressure but it enables issuing global loads early.
  auto globalLoadOps =
      llvm::to_vector(funcOp.getBody().getOps<triton::LoadOp>());

  // Avoid moving up global_load ops that don't belong to any prologue to avoid
  // extra register pressure.
  llvm::erase_if(globalLoadOps, [](triton::LoadOp op) {
    return !op->getAttr("amd.pipeliner_part");
  });

  for (auto op : llvm::reverse(globalLoadOps)) {
    // Gather use-def chain in block.
    Block *block = op->getBlock();
    SetVector<Operation *> backwardSet;

    BackwardSliceOptions options;
    options.omitBlockArguments = true;
    options.inclusive = false;
    // Slice should include values flowing into op regions
    options.omitUsesFromAbove = false;
    options.filter = [&](Operation *defOp) -> bool {
      Block *defBlock = defOp->getBlock();
      if (!block->findAncestorOpInBlock(*defOp))
        return false;

      // Only move ops residing in the same block.
      return defBlock == block;
    };
    (void)mlir::getBackwardSlice(op.getOperation(), &backwardSet, options);
    backwardSet.insert(op);

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

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

struct TritonAMDGPUReorderInstructionsPass
    : public impl::TritonAMDGPUReorderInstructionsBase<
          TritonAMDGPUReorderInstructionsPass> {
  void runOnOperation() override {
    ModuleOp m = getOperation();
    for (auto funcOp : m.getOps<triton::FuncOp>()) {
      moveDownCoversion(funcOp);
      moveUpTranspose(funcOp);
      moveUpGlobalLoadInPrologue(funcOp);
    }
  }
};

} // namespace mlir
