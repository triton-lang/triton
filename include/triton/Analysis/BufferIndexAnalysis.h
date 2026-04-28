#ifndef TRITON_ANALYSIS_BUFFER_INDEX_ANALYSIS_H
#define TRITON_ANALYSIS_BUFFER_INDEX_ANALYSIS_H

#include "triton/Analysis/Membar.h"

#include "mlir/IR/Dominance.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include <memory>
#include <vector>

namespace mlir {

class Block;
class Operation;
struct BufferIndexExpr;

/// Extends membar's slice disjointness check for multi-buffered shared-memory
/// allocations selected by `ttg.memdesc_index`.
///
/// The analysis is intentionally narrow. It decomposes the dynamic slot index
/// into `base + constantOffset`, optionally under a positive constant modulus,
/// and proves two accesses disjoint only when the expressions have the same
/// base and different offsets modulo the same modulus.
///
/// Recognized index shapes:
///   1. integer constants,
///   2. `arith.addi` with one constant operand,
///   3. `arith.remsi` with a positive constant modulus,
///   4. the pipeliner's select/cmpi one-step wrap:
///        select(cmpi sge(base + 1, N), 0, base + 1)
///        select(cmpi slt(base + 1, N), base + 1, 0)
///      only when -1 <= base < N is proven.
///
/// This is a per-function analysis: it owns dominance information for bounded
/// index proofs and interns expressions whose pointers are stored on
/// `AllocationSlice`. In scf form it checks the `scf.for` iter_arg init/yield
/// pair; in cf form it checks incoming block-argument operands and uses
/// dominance to distinguish loop backedges from non-backedge initial values.
///
/// Unknown arithmetic, dynamic or non-positive moduli, nested moduli, different
/// SSA bases, and loop-carried slices whose index was invalidated all fail the
/// disjointness proof and fall back to normal membar aliasing.
///
/// The class owns and interns the analyzed expressions attached to
/// `AllocationSlice`; callers must not keep those payloads beyond the lifetime
/// of this analysis.
class BufferIndexAnalysis {
public:
  explicit BufferIndexAnalysis(FunctionOpInterface funcOp);
  ~BufferIndexAnalysis();

  /// Builds an `AllocationSlice` for `value` and attaches a buffer-index
  /// expression to it. Callers in membar should use this rather than the raw
  /// `AllocationSlice` constructor so index expressions are attached
  /// consistently.
  AllocationSlice makeSlice(Value value, Interval<size_t> allocationInterval,
                            Allocation::BufferId bufferId);

  /// Returns true if `successor` is reached by a loop backedge from
  /// `terminator`. Region-form loops are handled structurally; cf-form loops
  /// use the standard dominance rule.
  bool isBackedgeSuccessor(Operation *terminator, Block *successor) const;

  /// Clears the buffer index of every slice in `info`, rebuilding both maps.
  /// Used at loop backedges where the same SSA value can denote a value from a
  /// different dynamic iteration, and before storing function summaries where
  /// per-function SSA index identity is no longer meaningful.
  void invalidateBufferIndices(BlockInfo &info) const;

private:
  void attachBufferIndex(AllocationSlice &slice, Value value);
  const BufferIndexExpr *intern(BufferIndexExpr expr);

  DominanceInfo dominanceInfo;
  std::vector<std::unique_ptr<BufferIndexExpr>> expressions;
};

/// Returns true only if the buffer-index expressions attached to `a` and `b`
/// provably denote different buffer slots in the same dynamic iteration.
/// Stateless; accesses only the payload pointers.
bool areBufferIndicesProvablyDifferent(const AllocationSlice &a,
                                       const AllocationSlice &b);

} // namespace mlir

#endif // TRITON_ANALYSIS_BUFFER_INDEX_ANALYSIS_H
