#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

//===----------------------------------------------------------------------===//
// This file will examine uses of induction variables to determine if a loop
// should be split into consecutive loops of [lo..midp) and [midp..hi].
// If the induction var is `<` or `>` a loop invariant value, it should be split.
//===----------------------------------------------------------------------===//

#define GEN_PASS_CLASSES
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

#define DEBUG_TYPE "triton-loop-split"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
namespace tt = mlir::triton;

namespace {

/// @brief Struct to collect characteristics of CmpI predicates.
struct CmpType {
  bool greater;
  bool equal;
  arith::CmpIPredicate inverted;
};

/// @brief Map to quickly determine if a supported predicate is present.
///        If present, quickly query characteristics.
static DenseMap<arith::CmpIPredicate, CmpType> CmpTypeMap = {
    {arith::CmpIPredicate::sge, {true, true, arith::CmpIPredicate::sle}},
    {arith::CmpIPredicate::sgt, {true, false, arith::CmpIPredicate::slt}},
    {arith::CmpIPredicate::sle, {false, true, arith::CmpIPredicate::sge}},
    {arith::CmpIPredicate::slt, {false, false, arith::CmpIPredicate::sgt}},
    {arith::CmpIPredicate::uge, {true, true, arith::CmpIPredicate::ule}},
    {arith::CmpIPredicate::ugt, {true, false, arith::CmpIPredicate::ult}},
    {arith::CmpIPredicate::ule, {false, true, arith::CmpIPredicate::uge}},
    {arith::CmpIPredicate::ult, {false, false, arith::CmpIPredicate::ugt}},
};

/// @brief  Capture the cmpi op and canonicalize (induction var on LHS).
class CmpCanon {
public:
  CmpCanon(arith::CmpIOp cmp, OpOperand &iter) : predicate(cmp.getPredicate()) {
    if (isValid()) {
      if (iter.getOperandNumber() == 1)
        predicate = CmpTypeMap[predicate].inverted;
      value = cmp.getOperand(iter.getOperandNumber() ^ 1);
    }
  }

  bool isValid() const {
    return CmpTypeMap.find(predicate) != CmpTypeMap.end();
  }

  bool isEqual() const {
    return isValid() ? CmpTypeMap[predicate].equal : false;
  }
  bool isGreater() const {
    return isValid() ? CmpTypeMap[predicate].greater : false;
  }

  Value getValue() const {
    return value;
  }

private:
  arith::CmpIPredicate predicate;
  Value value;
};

//===----------------------------------------------------------------------===//
/// @brief LoopBisect class to process an scf.for induction variable, and 
///        split the loop when the right conditions are met.
class LoopBisect {
public:
  LoopBisect(scf::ForOp _forOp) : forOp(_forOp) {}

  LogicalResult bisect();

private:
  void getCmp(OpOperand &opr);

private:
  // Data members
  scf::ForOp forOp;

  DenseMap<Operation *, CmpCanon> cmpMap;
};

/// Test the use for:
///  1. Is a CmpI
///  2. Is >=, <=, >, <
///  3. The comparison is loop-invariant
/// Note: poor man's SCEV
/// TODO: add support for mask logic
void LoopBisect::getCmp(OpOperand &opr) {
  if (auto cmp = dyn_cast<arith::CmpIOp>(opr.getOwner())) {
    LDBG("CMP: " << cmp);

    CmpCanon ccmp(cmp, opr);
    if (ccmp.isValid()) {
      // Other most be loop invariant, needs full SCEV analysis
      if (auto *defOther = ccmp.getValue().getDefiningOp()) {
        if (forOp->isAncestor(defOther))
          return;
      } else {
        auto blockArg = dyn_cast<BlockArgument>(ccmp.getValue()); // test block arg
        if (forOp->isAncestor(blockArg.getOwner()->getParentOp()))
          return;
      }
      cmpMap.insert(std::make_pair(cmp, ccmp));
    }
  }
}

LogicalResult LoopBisect::bisect() {
  auto lo = forOp.getLowerBound();
  auto hi = forOp.getUpperBound();
  auto step = forOp.getConstantStep();

  // LDBG("Loop: " << forOp);
  if (!step) {
    LDBG("Non-constant step");
    return failure();
  }

  // Collect comparators
  auto iter = forOp.getInductionVar();
  for (OpOperand &use : iter.getUses()) {
    getCmp(use);
  }

  // TODO: for multiple points, determine if they can be sorted, then split each
  // - for now just pick the first one
  if (cmpMap.size() >= 1) {
    auto [cmp, ccmp] = *cmpMap.begin();
 
    auto loc = cmp->getLoc();
    OpBuilder b(forOp);
 
    // make all unsigned?? assume I is lo..hi, increasing
    Value midp = ccmp.getValue();
    if (ccmp.isEqual()) {
      // return i >= c ? c - 1 : c + 1
      auto incr = b.create<arith::ConstantIntOp>(loc, ccmp.isGreater() ? -1 : 1, 32);
      midp = b.create<arith::AddIOp>(loc, midp, incr);
    }

    /// Handle midp not a multiple of step
    if (*step != 1) {
      // midp += (midp % step) ? 1 : 0
      auto zero = b.create<arith::ConstantIntOp>(loc, 0, 32);
      auto one = b.create<arith::ConstantIntOp>(loc, 1, 32);
      auto rem = b.create<arith::RemUIOp>(loc, midp, forOp.getStep());
      auto rcmp =
          b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, rem, zero);
      auto sel = b.create<arith::SelectOp>(loc, rcmp, one, zero);
      midp = b.create<arith::AddIOp>(loc, midp, sel);
    }

    /// TODO(sjw): update upstream peelForLoop
    /// bisect loop (lo .. midp)
    /// bisect loop (midp .. hi)
    IRMapping mapping;
    b.setInsertionPointAfter(forOp);
    scf::ForOp newForOp = cast<scf::ForOp>(b.clone(*forOp, mapping));
    newForOp.setLowerBound(midp);
    forOp.replaceAllUsesWith(newForOp.getResults());
    newForOp.getInitArgsMutable().assign(forOp->getResults());
    forOp.setUpperBound(midp);

    // replace cmp with constant True/False for each loop
    b.setInsertionPoint(forOp);
    cmp->replaceAllUsesWith(b.create<arith::ConstantIntOp>(loc, !ccmp.isGreater(), 1));
    auto *newCmp = mapping.lookup(cmp);
    newCmp->replaceAllUsesWith(b.create<arith::ConstantIntOp>(loc, ccmp.isGreater(), 1));
  }

  return success();
}

struct LoopBisectPass : public TritonLoopSplitBase<LoopBisectPass> {
  LoopBisectPass() = default;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    SmallVector<scf::ForOp> loops;
    getOperation()->walk<WalkOrder::PostOrder>(
        [&](scf::ForOp forOp) { loops.push_back(forOp); });

    for (scf::ForOp forOp : loops) {
      LoopBisect sp(forOp);
      if (failed(sp.bisect()))
        continue;
    }
  }

private:
};
} // namespace

std::unique_ptr<Pass> mlir::triton::createLoopSplitPass() {
  return std::make_unique<LoopBisectPass>();
}
