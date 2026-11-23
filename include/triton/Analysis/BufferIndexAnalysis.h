#ifndef TRITON_ANALYSIS_BUFFER_INDEX_ANALYSIS_H
#define TRITON_ANALYSIS_BUFFER_INDEX_ANALYSIS_H

#include "mlir/IR/Dominance.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include <compare>
#include <memory>
#include <vector>

namespace mlir {

class Block;
class CFGLoopInfo;
class Operation;
struct BufferIndexExpr;

namespace triton::membar {

struct Subslice;

/// Relative coordinates of a descriptor. Two views can be disjoint even
/// when their sets of possible physical addresses overlap.
struct RelativeAddress {
  const BufferIndexExpr *index = nullptr;
  const Subslice *subslice = nullptr;

  /// Compare the recorded coordinates within a single SSA binding.
  /// Subslice bounds are relative to their common source descriptor.
  /// Slot comparisons also require the same allocator buffer.
  bool disjoint(RelativeAddress other, bool sameBuffer) const;
  auto operator<=>(const RelativeAddress &) const = default;
};

} // namespace triton::membar

/// Extends membar's slice disjointness check for multi-buffered shared-memory
/// allocations selected by `ttg.memdesc_index`.
///
/// The analysis is intentionally narrow. It decomposes the dynamic slot index
/// into `base + constantOffset`, optionally under a positive constant modulus,
/// and proves two accesses disjoint when the expressions have the same base
/// and different offsets modulo the same modulus. Cursors of the same block
/// also stay disjoint if their slots start at distinct residues and advance
/// by the same constant step on every backedge.
/// Scalar indices follow Triton's block-programming semantics. Comparisons
/// require the same binding of the indexed source and the index base. A
/// source change can change both the origin and the byte stride of a slot.
///
/// Recognized index shapes:
///   1. integer constants,
///   2. `arith.addi` with one constant operand,
///   3. `arith.remsi` with a positive constant modulus, or `arith.remui`
///      with a power-of-two modulus,
///   4. the pipeliner's select/cmpi one-step wrap:
///        select(cmpi sge(base + 1, N), 0, base + 1)
///        select(cmpi slt(base + 1, N), base + 1, 0)
///      only when -1 <= base < N is proven.
///
/// This is a per-function analysis. It owns dominance information used to
/// detect cf-form loop backedges and interns the analyzed expressions whose
/// pointers are stored in `RelativeAddress`. The select/cmpi recurrence base
/// must be provably in `[-1, N)`, which is checked from the loop-header block
/// argument's incoming operands.
///
/// Buffer-index reasoning is disabled for functions with irreducible control
/// flow because not every cyclic edge is a dominance backedge in such CFGs.
///
/// Unknown arithmetic and dynamic, non-positive or nested moduli become
/// opaque SSA bases. Unrelated bases or a change of binding leave Membar
/// to check physical overlap.
///
/// `RelativeAddress` borrows slot expressions and subslice coordinates.
/// Both are valid for the lifetime of this analysis.
class BufferIndexAnalysis {
public:
  explicit BufferIndexAnalysis(FunctionOpInterface funcOp);
  ~BufferIndexAnalysis();

  /// Obtain direct subslice coordinates and the slot selected through views.
  triton::membar::RelativeAddress getRelativeAddress(Value value,
                                                     bool useIndex = true);

  /// Returns true if `successor` may revisit definitions from `terminator`'s
  /// dynamic iteration. Uses dominance for reducible cf-form loops and
  /// conservatively invalidates region-to-region, nested and irreducible CFG
  /// edges, whose reducibility is not established by this analysis.
  bool isBackedgeSuccessor(Operation *terminator, Block *successor) const;

  /// Whether a branch enters a natural loop from outside its header's
  /// dominance region. Restricted to the analyzed function-level CFG.
  bool entersLoop(Operation *terminator) const;

  /// Natural loops in a multi-block region. Single-block regions have no
  /// dominance tree.
  const CFGLoopInfo *getLoopInfo(Region *region) const;

private:
  const BufferIndexExpr *getIndexExpr(Value value);

  DominanceInfo dominanceInfo;
  mutable DenseMap<Region *, std::unique_ptr<CFGLoopInfo>> loops;
  bool hasReducibleCFG;
  std::vector<std::unique_ptr<BufferIndexExpr>> expressions;
  DenseMap<Value, std::unique_ptr<triton::membar::Subslice>> subslices;
};

} // namespace mlir

#endif // TRITON_ANALYSIS_BUFFER_INDEX_ANALYSIS_H
