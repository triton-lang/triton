#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

namespace mlir::triton {

#define GEN_PASS_DEF_TRITONLOOPUNSWITCH
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

#define DEBUG_TYPE "triton-loop-unswitch"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {


void inlineResolvedIf(scf::IfOp ifOp, Region &region) {
  if (region.empty()) {
    ifOp.erase();
    return;
  }
  Block &block = region.front();
  auto yield = cast<scf::YieldOp>(block.getTerminator());
  ifOp.replaceAllUsesWith(yield.getOperands());
  ifOp->getBlock()->getOperations().splice(ifOp->getIterator(),
                                           block.getOperations(), block.begin(),
                                           std::prev(block.end()));
  ifOp.erase();
}


bool isInvariant(scf::ForOp forOp, Value value,
                 SmallVectorImpl<Operation *> &toHoist) {
  if (!forOp.getBodyRegion().isAncestor(value.getParentRegion()))
    return true;
  // Induction variable or iter_args.
  if (isa<BlockArgument>(value))
    return false;
  Operation *def = value.getDefiningOp();
  if (def->getBlock() != forOp.getBody())
    return false;
  if (!isMemoryEffectFree(def) || !isSpeculatable(def))
    return false;
  for (Value operand : def->getOperands())
    if (!isInvariant(forOp, operand, toHoist))
      return false;
  if (!llvm::is_contained(toHoist, def))
    toHoist.push_back(def);
  return true;
}

// Return the first scf.if directly in the body of \p forOp whose condition is
// loop-invariant, hoisting the condition computation out of the loop if
// needed.
scf::IfOp findUnswitchCandidate(scf::ForOp forOp) {
  for (Operation &op : forOp.getBody()->without_terminator()) {
    auto ifOp = dyn_cast<scf::IfOp>(&op);
    if (!ifOp)
      continue;
    SmallVector<Operation *> toHoist;
    if (!isInvariant(forOp, ifOp.getCondition(), toHoist))
      continue;
    for (Operation *def : toHoist)
      def->moveBefore(forOp);
    return ifOp;
  }
  return nullptr;
}

bool tryUnswitch(scf::ForOp forOp, unsigned maxBodySize) {
  unsigned bodySize = 0;
  forOp.getBody()->walk([&](Operation *) { ++bodySize; });
  if (bodySize > maxBodySize) {
    LDBG("skipping loop with body size " << bodySize);
    return false;
  }

  scf::IfOp candidate = findUnswitchCandidate(forOp);
  if (!candidate)
    return false;

  OpBuilder b(forOp);
  Location loc = forOp.getLoc();
  auto newIf = scf::IfOp::create(b, loc, forOp.getResultTypes(),
                                 candidate.getCondition(),
                                 /*withElseRegion=*/true);

  // Clone the loop into one branch of the new if and resolve the candidate to
  // the corresponding region.
  auto specialize = [&](Block *dest, bool takeThen) {
    // Builders of scf.if may add a terminator for ifs with no results , the clone
    // supplies its own yield below.
    if (!dest->empty() && dest->back().hasTrait<OpTrait::IsTerminator>())
      dest->back().erase();
    OpBuilder bb = OpBuilder::atBlockEnd(dest);
    IRMapping map;
    auto clonedFor = cast<scf::ForOp>(bb.clone(*forOp.getOperation(), map));
    auto clonedIf = cast<scf::IfOp>(map.lookup(candidate.getOperation()));
    inlineResolvedIf(clonedIf, takeThen ? clonedIf.getThenRegion()
                                        : clonedIf.getElseRegion());
    scf::YieldOp::create(bb, loc, clonedFor.getResults());
  };
  specialize(newIf.thenBlock(), /*takeThen=*/true);
  specialize(newIf.elseBlock(), /*takeThen=*/false);

  forOp.replaceAllUsesWith(newIf.getResults());
  forOp.erase();
  return true;
}

} // anonymous namespace

class LoopUnswitchPass : public impl::TritonLoopUnswitchBase<LoopUnswitchPass> {
public:
  using TritonLoopUnswitchBase::TritonLoopUnswitchBase;

  void runOnOperation() override {
    // Collect loops post-order so that unswitching an inner
    // loop is visible to the size check of its enclosing loops, and process
    // each loop at most once to bound duplication.
    SmallVector<scf::ForOp> loops;
    getOperation()->walk([&](scf::ForOp forOp) { loops.push_back(forOp); });
    for (scf::ForOp forOp : loops)
      if (tryUnswitch(forOp, maxBodySize))
        LDBG("unswitched loop");
  }
};

} // namespace mlir::triton
