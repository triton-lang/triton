#ifndef TRITON_ANALYSIS_BUFFER_INDEX_ANALYSIS_H
#define TRITON_ANALYSIS_BUFFER_INDEX_ANALYSIS_H

#include "mlir/IR/Dominance.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include <cstdint>
#include <memory>
#include <vector>

namespace mlir {

class Block;
class CFGLoopInfo;
class Operation;
struct BufferIndexExpr;

namespace triton::membar {

struct Subslice {
  // SSA identity of the descriptor defining the coordinate system.
  const void *source;
  // Bounds borrowed from the immutable offset attribute and result type.
  ArrayRef<int32_t> offsets;
  ArrayRef<int64_t> shape;
};

/// Compare two known subviews within one binding of their common source.
bool disjointSubslices(const Subslice *a, const Subslice *b);

} // namespace triton::membar

/// Describes buffer slots selected by `ttg.memdesc_index` and the control-flow
/// edges that can rebind the indexed descriptor or its index.
class BufferIndexAnalysis {
public:
  explicit BufferIndexAnalysis(FunctionOpInterface funcOp);
  ~BufferIndexAnalysis();

  /// Return the direct subslice coordinates, if any, owned by this analysis.
  const triton::membar::Subslice *getSubslice(Value value);

  /// Follow slot-preserving views of shared-memory descriptors. Return null
  /// for unknown slots or irreducible control flow. Owns the expressions.
  const BufferIndexExpr *getIndexExpr(Value value);

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
  DominanceInfo dominanceInfo;
  mutable DenseMap<Region *, std::unique_ptr<CFGLoopInfo>> loops;
  bool hasReducibleCFG;
  std::vector<std::unique_ptr<BufferIndexExpr>> expressions;
  DenseMap<Value, std::unique_ptr<triton::membar::Subslice>> subslices;
};

/// Prove that two slots of the same descriptor differ within one SSA binding.
/// A null expression supplies no proof.
bool areBufferIndicesProvablyDifferent(const BufferIndexExpr *a,
                                       const BufferIndexExpr *b);

} // namespace mlir

#endif // TRITON_ANALYSIS_BUFFER_INDEX_ANALYSIS_H
