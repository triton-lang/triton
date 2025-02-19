#ifndef TRITONAMD_ANALYSIS_RANGE_ANALYSIS_H
#define TRITONAMD_ANALYSIS_RANGE_ANALYSIS_H

#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Interfaces/LoopLikeInterface.h"

namespace mlir::triton::AMD {
/// This struct (analysis) adapt's upstream's IntegerRangeAnalysis (inferring
/// lower/upperbounds on integer constants) to our needs.
/// Specifically there are 2 points of extension:
///
/// 1. Support for GetProgramIdOp, MakeRangeOp, SplatOp, ExpandDimsOp. *Note*,
/// upstream already supports range inference for shaped types such as tensors
/// (here we just implement effectively implement the interfaces for our ops).
///    * Upstream's semantics for "range of shape type" is union over ranges of
///    elements.
///    * We do not use tablegen to implement
///    DeclareOpInterfaceMethods<InferIntRangeInterface, ["inferResultRanges"]>
///    in order to keep the entire implementation contained/encapsulated.
///
/// 2. Support for inference "through loops". Upstream's analysis conservatively
/// inferences [min_int, max_int] for loop carried values (and therefore loop
/// body values). Here we attempt to do better by analysis the loop bounds and
/// "abstractly interpreting" the loop when loop bounds are statically known.
/// See visitRegionSuccessors.
struct TritonIntegerRangeAnalysis : dataflow::IntegerRangeAnalysis {
  using dataflow::IntegerRangeAnalysis::IntegerRangeAnalysis;

  llvm::SmallDenseMap<LoopLikeOpInterface, int64_t> loopTripCounts;
  llvm::SmallDenseMap<
      std::pair<LoopLikeOpInterface, dataflow::IntegerValueRangeLattice *>,
      int64_t>
      loopVisits;

  void setToEntryState(dataflow::IntegerValueRangeLattice *lattice) override;

  LogicalResult visitOperation(
      Operation *op,
      ArrayRef<const dataflow::IntegerValueRangeLattice *> operands,
      ArrayRef<dataflow::IntegerValueRangeLattice *> results) override;

  /// This method (which overloads
  /// AbstractSparseForwardDataFlowAnalysis::visitRegionSuccessors) implements
  /// "abstract interpretation" of loops with statically known bounds in order
  /// to infer tight ranges for loop carried values (and therefore loop body
  /// values). By "abstract interpretation" we mean lattice states are
  /// propagated to all region successors N times, where N is the total trip
  /// count of the loop. Recall for scf.for, both the loop itself and the users
  /// of the loop successors. Thus, after N propagations both loop body values
  /// and users of loop results will have accurate ranges (assuming we have
  /// implemented support for range analysis on the ops).
  /// *Note*, this implementation is majority similar to
  /// AbstractSparseForwardDataFlowAnalysis::visitRegionSuccessors (so check
  /// there for more explanation/insight) and basically only does two things
  /// differently:
  ///
  /// 1. If the branch op is a loop (LoopLikeOpInterface) then we attempt to
  /// compute its total trip count (nested loop trip counts multiply) and
  /// initialize a visit count to 0. Note, due to how Dataflow analysis works we
  /// have to actually visit the loop N times for each iter_arg (each argument
  /// lattice) so we actually track visit count for (loop, arg) not just (loop).
  ///
  /// 2. Before propagating, we check if we have propagated for (loop, arg) >= N
  /// times. If so, we do not propagate (and thus the traversal converges/ends).
  ///
  /// Note, for loops where the trip count cannot be inferred *and* loops with a
  /// total trip count larger than `kDefaultMaxTripCount`, fallback to
  /// upstream's conservative inference (i.e., we infer [min_int, max_int]) for
  /// the loop operands and all users and all users of the results of the loop.
  void visitRegionSuccessors(
      ProgramPoint *point, RegionBranchOpInterface branch,
      RegionBranchPoint successor,
      ArrayRef<dataflow::AbstractSparseLattice *> abstractLattices) override;
};

// TODO(max): remove after we catch up to
// https://github.com/llvm/llvm-project/pull/127888
LogicalResult staticallyNonNegative(DataFlowSolver &solver, Value v);
LogicalResult staticallyNonNegative(DataFlowSolver &solver, Operation *op);

} // namespace mlir::triton::AMD

#endif
