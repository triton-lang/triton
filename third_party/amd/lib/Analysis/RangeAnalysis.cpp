#include "third_party/amd/include/Analysis/RangeAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/Support/Debug.h"

#include <numeric>
#include <optional>

#undef DEBUG_TYPE
#define DEBUG_TYPE "tritonamdgpu-range-analysis"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;

namespace tt = mlir::triton;

// TODO(max): remove after we catch up to
// https://github.com/llvm/llvm-project/pull/127888
namespace mlir::triton::AMD {
LogicalResult staticallyNonNegative(DataFlowSolver &solver, Value v) {
  auto *result = solver.lookupState<dataflow::IntegerValueRangeLattice>(v);
  if (!result || result->getValue().isUninitialized())
    return failure();
  const ConstantIntRanges &range = result->getValue().getValue();
  return success(range.smin().isNonNegative());
}

LogicalResult staticallyNonNegative(DataFlowSolver &solver, Operation *op) {
  auto nonNegativePred = [&solver](Value v) -> bool {
    return succeeded(staticallyNonNegative(solver, v));
  };
  return success(llvm::all_of(op->getOperands(), nonNegativePred) &&
                 llvm::all_of(op->getResults(), nonNegativePred));
}
} // namespace mlir::triton::AMD

namespace {

constexpr int64_t kDefaultMaxTripCount = 1024;

std::optional<int64_t> maybeGetTripCount(LoopLikeOpInterface loop) {
  std::optional<OpFoldResult> lowerBound = loop.getSingleLowerBound();
  std::optional<OpFoldResult> upperBound = loop.getSingleUpperBound();
  std::optional<OpFoldResult> step = loop.getSingleStep();
  if (lowerBound && upperBound && step)
    return constantTripCount(*lowerBound, *upperBound, *step);
  return {};
}

void getEnclosingLoops(Operation &op, SmallVector<LoopLikeOpInterface> &ops) {
  Operation *currOp = op.getParentOp();
  while (currOp) {
    if (isa<LoopLikeOpInterface>(currOp))
      ops.push_back(llvm::cast<LoopLikeOpInterface>(currOp));
    currOp = currOp->getParentOp();
  }
}

void inferResultRanges(tt::GetProgramIdOp *op, SetIntRangeFn setResultRange) {
  constexpr int64_t kDefaultMaxProgramID = 2 << 15; // 65536
  setResultRange(
      op->getResult(),
      ConstantIntRanges::range(
          /*min*/ {/*numBits*/ 32, /*val*/ 0, /*isSigned*/ true},
          /*max*/
          {/*numBits*/ 32, /*val*/ kDefaultMaxProgramID, /*isSigned*/ true},
          /*isSigned*/ true));
}

void inferResultRanges(tt::MakeRangeOp *op, SetIntRangeFn setResultRange) {
  setResultRange(
      op->getResult(),
      ConstantIntRanges::range(
          /*min*/ {/*numBits*/ 32, /*val*/ op->getStart(), /*isSigned*/ true},
          /*max*/ {/*numBits*/ 32, /*val*/ op->getEnd(), /*isSigned*/ true},
          /*isSigned*/ true));
}

void inferResultRanges(tt::SplatOp *op, ArrayRef<ConstantIntRanges> argRanges,
                       SetIntRangeFn setResultRange) {
  setResultRange(op->getResult(), argRanges[0]);
}

void inferResultRanges(tt::ExpandDimsOp *op,
                       ArrayRef<ConstantIntRanges> argRanges,
                       SetIntRangeFn setResultRange) {
  setResultRange(op->getResult(), argRanges[0]);
}

} // namespace

namespace mlir::triton::AMD {

void TritonIntegerRangeAnalysis::setToEntryState(
    dataflow::IntegerValueRangeLattice *lattice) {
  propagateIfChanged(lattice, lattice->join(IntegerValueRange::getMaxRange(
                                  lattice->getAnchor())));
}

LogicalResult TritonIntegerRangeAnalysis::visitOperation(
    Operation *op,
    ArrayRef<const dataflow::IntegerValueRangeLattice *> operands,
    ArrayRef<dataflow::IntegerValueRangeLattice *> results) {
  LDBG("  Inferring ranges for " << *op << "\n");
  // This callback is almost exactly like the callback in
  // IntegerRangeAnalysis::visitOperation except we do not "short-cicruit" the
  // analysis by inferring a maximum range for loop results (instead we
  // perform a check based on visit counts in visitRegionSuccessors).
  auto joinCallback = [&op, &results,
                       this](Value v, const IntegerValueRange &incomingRange) {
    auto result = dyn_cast<OpResult>(v);
    if (!result)
      return;
    assert(llvm::is_contained(op->getResults(), result));

    LDBG("  Inferred range " << incomingRange << "\n");
    dataflow::IntegerValueRangeLattice *lattice =
        results[result.getResultNumber()];
    IntegerValueRange oldRange = lattice->getValue();
    ChangeResult changed = lattice->join(incomingRange);
    propagateIfChanged(lattice, changed);
  };

  if (llvm::isa<GetProgramIdOp, MakeRangeOp, SplatOp, ExpandDimsOp>(op)) {
    SmallVector<ConstantIntRanges> argRanges;
    for (auto lattice : operands) {
      if (lattice->getValue().isUninitialized()) {
        setAllToEntryStates(results);
        return success();
      }
      argRanges.push_back(lattice->getValue().getValue());
    }
    if (auto op_ = llvm::dyn_cast<GetProgramIdOp>(op))
      inferResultRanges(&op_, joinCallback);
    else if (auto op_ = llvm::dyn_cast<SplatOp>(op))
      inferResultRanges(&op_, argRanges, joinCallback);
    else if (auto op_ = llvm::dyn_cast<ExpandDimsOp>(op))
      inferResultRanges(&op_, argRanges, joinCallback);
    else if (auto op_ = llvm::dyn_cast<MakeRangeOp>(op))
      inferResultRanges(&op_, joinCallback);
    else {
      llvm_unreachable("Unsupported operation");
    }
    return success();
  }

  auto inferrable = dyn_cast<InferIntRangeInterface>(op);
  if (!inferrable) {
    setAllToEntryStates(results);
    return success();
  }
  SmallVector<IntegerValueRange> argRanges = llvm::map_to_vector(
      operands, [](const dataflow::IntegerValueRangeLattice *lattice) {
        return lattice->getValue();
      });

  inferrable.inferResultRangesFromOptional(argRanges, joinCallback);
  return success();
}

void TritonIntegerRangeAnalysis::visitRegionSuccessors(
    ProgramPoint *point, RegionBranchOpInterface branch,
    RegionBranchPoint successor,
    ArrayRef<dataflow::AbstractSparseLattice *> abstractLattices) {
  SmallVector<dataflow::IntegerValueRangeLattice *> lattices;
  for (auto abstractLat : abstractLattices) {
    lattices.push_back(
        static_cast<dataflow::IntegerValueRangeLattice *>(abstractLat));
  }
  // Initialize loop trip counts
  LoopLikeOpInterface loop =
      llvm::dyn_cast<LoopLikeOpInterface>(branch.getOperation());
  if (loop) {
    if (!loopTripCounts.contains(loop)) {
      SmallVector loops{loop};
      getEnclosingLoops(*loop, loops);
      int loopTripCount =
          std::accumulate(loops.begin(), loops.end(), 1,
                          [](int accum, LoopLikeOpInterface loop) {
                            return accum * maybeGetTripCount(loop).value_or(
                                               kDefaultMaxTripCount + 1);
                          });
      loopTripCounts[loop] = loopTripCount;
    }
    for (auto argLat : lattices) {
      if (!loopVisits.contains({loop, argLat})) {
        loopVisits[{loop, argLat}] = 0;
      }
    }
  }

  const auto *predecessors =
      getOrCreateFor<dataflow::PredecessorState>(point, point);
  assert(predecessors->allPredecessorsKnown() &&
         "unexpected unresolved region successors");
  for (Operation *op : predecessors->getKnownPredecessors()) {
    std::optional<OperandRange> operands;
    if (op == branch) {
      operands = branch.getEntrySuccessorOperands(successor);
    } else if (auto regionTerminator =
                   dyn_cast<RegionBranchTerminatorOpInterface>(op)) {
      operands = regionTerminator.getSuccessorOperands(successor);
    }
    if (!operands)
      return setAllToEntryStates(lattices);

    ValueRange inputs = predecessors->getSuccessorInputs(op);
    assert(inputs.size() == operands->size() &&
           "expected the same number of successor inputs as operands");

    unsigned firstIndex = 0;
    if (inputs.size() != lattices.size()) {
      if (!point->isBlockStart()) {
        if (!inputs.empty()) {
          firstIndex = cast<OpResult>(inputs.front()).getResultNumber();
        }
        visitNonControlFlowArguments(branch,
                                     RegionSuccessor(branch->getResults().slice(
                                         firstIndex, inputs.size())),
                                     lattices, firstIndex);
      } else {
        if (!inputs.empty()) {
          firstIndex = cast<BlockArgument>(inputs.front()).getArgNumber();
        }
        Region *region = point->getBlock()->getParent();
        visitNonControlFlowArguments(
            branch,
            RegionSuccessor(region, region->getArguments().slice(
                                        firstIndex, inputs.size())),
            lattices, firstIndex);
      }
    }

    for (auto [oper, argLat] :
         llvm::zip(*operands, ArrayRef(lattices).drop_front(firstIndex))) {
      std::pair loopArgLat = {loop, argLat};
      // If we've "run the loop" #tripcount times, stop propagating.
      if (loop && loopVisits[loopArgLat] >= loopTripCounts[loop])
        continue;
      ChangeResult changed;
      if (loop && loopTripCounts[loop] > kDefaultMaxTripCount) {
        // If the loop's tripcount is too large, infer the maximum range for
        // the arg lattices. This will have the effect that all users will
        // also be inferred to have maximum range and end the analysis will
        // end (the maximum range is the "top" of the lattice and thus no
        // further changes/updates are possible).
        changed = argLat->join(IntegerValueRange::getMaxRange(oper));
      } else {
        // Else, propagate pred operands.
        changed = argLat->join(*getLatticeElementFor(point, oper));
      }
      propagateIfChanged(argLat, changed);
      // Only increase the loop visitation count if have actually update the
      // lattice because otherwise we will over count the number of visits
      // (since not all iter_arg lattices are updated/propagated on each
      // visit).
      if (loop && changed == ChangeResult::Change)
        ++loopVisits[loopArgLat];
    }
  }
}
} // namespace mlir::triton::AMD
