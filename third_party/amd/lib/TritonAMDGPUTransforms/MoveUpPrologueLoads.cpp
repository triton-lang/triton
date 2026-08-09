//===----------------------------------------------------------------------===//
// This pass moves global load ops early in the prologue for prefetching to
// improve GEMM performance. It only affects non-volatile loads marked with
// the "amd.pipeliner_part" attribute.
//
// The pass moves each load op and its dependencies (backward slice) to the
// earliest valid insertion point, which is after:
//   - The defining op of the source pointer
//   - Any operation that may modify global memory
//   - Any atomic ops
//   - Any barriers
//   - Any loop ops (scf.for, scf.while)
//
// This may increase register pressure but enables issuing global loads early.
//===----------------------------------------------------------------------===//

#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace mlir {

#define GEN_PASS_DEF_TRITONAMDGPUMOVEUPPROLOGUELOADS
#include "TritonAMDGPUTransforms/Passes.h.inc"

namespace {

// Return true when moving a global load above `op` may change the value it
// observes. Unknown effects and effects on the default resource may alias
// global memory; explicitly disjoint resources, such as shared memory, do not.
static bool mayModifyGlobalMemory(Operation *op) {
  auto effects = getEffectsRecursively(op);
  if (!effects)
    return true;

  return llvm::any_of(
      *effects, [](const MemoryEffects::EffectInstance &effect) {
        if (!isa<MemoryEffects::Write, MemoryEffects::Free>(effect.getEffect()))
          return false;
        if (effect.getResource() == SideEffects::DefaultResource::get())
          return true;
        return !effect.getResource()->isDisjointFrom(tt::GlobalMemory::get());
      });
}

// Find the earliest valid insertion point for the load op in its block.
static Block::iterator findEarlyInsertionPoint(Block *block, tt::LoadOp load) {
  Block::iterator insertPoint = block->end();
  Value ptr = load.getPtr();
  for (Operation &op : *block) {
    if (&op == load)
      break; // Don't move past current location

    // Update insertion point if this op defines the source pointer.
    if (llvm::is_contained(op.getResults(), ptr)) {
      insertPoint = Block::iterator(&op);
      continue;
    }

    // Break at global memory modifications, atomics, barriers and loops.
    if (mayModifyGlobalMemory(&op) ||
        isa<tt::AtomicRMWOp, tt::AtomicCASOp, tt::AtomicPollOp, gpu::BarrierOp,
            ttg::BarrierOp, scf::ForOp, scf::WhileOp>(&op)) {
      insertPoint = Block::iterator(&op);
    }
  }
  return insertPoint;
}

static void moveUpLoad(tt::LoadOp load) {
  // Gather backward slice of the load op within the same block.
  Block *block = load->getBlock();
  SetVector<Operation *> slice;
  BackwardSliceOptions options;
  options.omitBlockArguments = true;
  options.inclusive = false;
  options.omitUsesFromAbove = false;
  options.filter = [block](Operation *op) { return op->getBlock() == block; };
  (void)getBackwardSlice(load.getOperation(), &slice, options);
  slice.insert(load);

  // Find the earliest valid insertion point for the load op.
  Block::iterator insertPoint = findEarlyInsertionPoint(block, load);
  SmallVector<Operation *> opsToMove = slice.takeVector();
  if (insertPoint != block->end()) {
    // Filter out ops already at or before the insertion point.
    llvm::erase_if(opsToMove, [&](Operation *op) {
      return !insertPoint->isBeforeInBlock(op);
    });
  }

  // Move the ops to the insertion point.
  for (Operation *op : llvm::reverse(opsToMove)) {
    if (insertPoint != block->end())
      op->moveAfter(block, insertPoint);
    else
      op->moveBefore(block, block->begin());
  }
}

} // namespace

struct TritonAMDGPUMoveUpPrologueLoadsPass
    : public impl::TritonAMDGPUMoveUpPrologueLoadsBase<
          TritonAMDGPUMoveUpPrologueLoadsPass> {
  void runOnOperation() override {
    // Collect load ops with "amd.pipeliner_part" attribute.
    SmallVector<tt::LoadOp> prologueLoads;
    getOperation().walk([&](tt::LoadOp load) {
      if (load->hasAttr("amd.pipeliner_part") && !load.getIsVolatile())
        prologueLoads.push_back(load);
    });
    // Process in reverse order to maintain relative order of moved ops.
    for (tt::LoadOp load : llvm::reverse(prologueLoads))
      moveUpLoad(load);
  }
};

} // namespace mlir
