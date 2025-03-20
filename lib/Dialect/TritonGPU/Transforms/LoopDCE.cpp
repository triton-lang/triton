#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

using namespace mlir;
using namespace triton;

//===----------------------------------------------------------------------===//
// recursiveLoopDCE
//===----------------------------------------------------------------------===//

// Recursively remove dead operations in the loop body. These are subgraphs of
// memory-effect-free ops in the loop whose results are transitively only used
// by themselves.
static void recursiveLoopDCE(scf::ForOp loop) {
  SetVector<Value> liveSet;
  DenseSet<Operation *> liveOps;
  auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  for (auto [i, result] : llvm::enumerate(loop.getResults())) {
    if (!result.use_empty())
      liveSet.insert(yield.getOperand(i));
  }
  for (Operation &op : loop.getBody()->without_terminator()) {
    if (isMemoryEffectFree(&op))
      continue;
    liveOps.insert(&op);
    for (Value operand : getNestedOperands(&op))
      liveSet.insert(operand);
  }

  for (unsigned i = 0; i < liveSet.size(); ++i) {
    Value value = liveSet[i];
    // Skip values defined outside the loop body.
    if (value.getParentRegion()->isProperAncestor(&loop.getBodyRegion()))
      continue;

    auto arg = dyn_cast<BlockArgument>(value);
    if (arg && arg.getOwner() == loop.getBody()) {
      // Skip the induction variable.
      if (arg == loop.getInductionVar())
        continue;
      liveSet.insert(yield.getOperand(arg.getArgNumber() - 1));
      continue;
    }
    Operation *op = arg ? arg.getOwner()->getParentOp() : value.getDefiningOp();
    liveOps.insert(op);
    for (Value operand : getNestedOperands(op))
      liveSet.insert(operand);
  }

  llvm::BitVector toErase(loop.getNumRegionIterArgs());
  for (auto [i, operand] : llvm::enumerate(yield.getOperands())) {
    if (!liveSet.count(operand))
      toErase.set(i);
  }
  for (Operation &op :
       llvm::make_early_inc_range(loop.getBody()->without_terminator())) {
    if (!liveOps.count(&op)) {
      op.dropAllUses();
      op.erase();
    }
  }
  eraseLoopCarriedValues(loop, std::move(toErase));
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace mlir::triton::gpu {
#define GEN_PASS_DEF_TRITONGPULOOPDCE
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu

namespace {
struct LoopDCE : triton::gpu::impl::TritonGPULoopDCEBase<LoopDCE> {
  using TritonGPULoopDCEBase::TritonGPULoopDCEBase;

  void runOnOperation() override {
    getOperation().walk([&](scf::ForOp loop) { recursiveLoopDCE(loop); });
  }
};
} // namespace
