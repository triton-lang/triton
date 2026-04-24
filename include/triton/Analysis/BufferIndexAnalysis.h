#ifndef TRITON_ANALYSIS_BUFFER_INDEX_ANALYSIS_H
#define TRITON_ANALYSIS_BUFFER_INDEX_ANALYSIS_H

#include "triton/Analysis/Membar.h"

#include "mlir/IR/Value.h"

namespace mlir {

class Block;
class Operation;

/// Returns true only if `a` and `b` provably denote different buffer slots
/// within a single execution of the enclosing region. Used by the membar
/// analysis to skip barriers between multi-buffered shared-memory accesses
/// whose dynamic slot indices are different.
bool areIndicesProvablyDifferent(Value a, Value b);

/// Returns the dynamic slot index of a multi-buffered shared-memory access,
/// or a null Value if one cannot be recovered. Walks through
/// MemDescViewTrait producers to the underlying MemDescIndexOp.
Value extractBufferIndex(Value value);

/// Buffer-index payload stored in AllocationSlice::extensionKey.

inline Value getBufferIndex(const AllocationSlice &slice) {
  return Value::getFromOpaquePointer(slice.extensionKey);
}

inline void setBufferIndex(AllocationSlice &slice, Value value) {
  slice.extensionKey = value.getAsOpaquePointer();
}

/// Returns a copy of `slice` with its buffer index cleared. Used when
/// propagating BlockInfo across a CFG backedge: the underlying SSA
/// value denotes a different runtime integer on the carried side vs.
/// the next iteration, so the analysis must not compare the two. A
/// null index fails areIndicesProvablyDifferent and falls back to
/// conservative aliasing.
inline AllocationSlice
withInvalidatedBufferIndex(const AllocationSlice &slice) {
  AllocationSlice copy = slice;
  copy.extensionKey = nullptr;
  return copy;
}

/// Returns true if `successor` is the entry block of a CFG edge from
/// `terminator` that re-enters an earlier region (e.g. scf.for yield ->
/// body, scf.while after -> before). Used by membar as a post-join hook
/// to invalidate buffer indices at loop-carried merges.
bool isBackedgeSuccessor(Operation *terminator, Block *successor);

/// Clears the buffer index of every slice in `info`, rebuilding both
/// slice maps. Intended as a post-join step at loop-carried merges: we
/// cannot evaluate index disjointness over loop iterations, so any
/// index attached to a slice reaching a loop header is unreliable.
void invalidateBufferIndices(BlockInfo &info);

} // namespace mlir

#endif // TRITON_ANALYSIS_BUFFER_INDEX_ANALYSIS_H
