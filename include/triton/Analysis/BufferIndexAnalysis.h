#ifndef TRITON_ANALYSIS_BUFFER_INDEX_ANALYSIS_H
#define TRITON_ANALYSIS_BUFFER_INDEX_ANALYSIS_H

#include "mlir/IR/Value.h"

namespace mlir {

struct BlockInfo;

/// Returns true only if `a` and `b` provably denote different buffer slots
/// within a single execution of the enclosing region. Used by the membar
/// analysis to skip barriers between multi-buffered shared-memory accesses
/// whose dynamic slot indices are different.
bool areIndicesProvablyDifferent(Value a, Value b);

/// Returns the dynamic slot index of a multi-buffered shared-memory access,
/// or a null Value if one cannot be recovered. Walks through
/// MemDescViewTrait producers to the underlying MemDescIndexOp.
Value extractBufferIndex(Value value);

/// Like BlockInfo::join, but for slices propagated across a CFG backedge:
/// each incoming slice has its bufferIndex cleared before being merged
/// into `dest`, since the SSA value denotes a different runtime integer
/// on the carried side vs. the next iteration. A null index fails
/// areIndicesProvablyDifferent and falls back to conservative aliasing.
void joinFromBackedge(BlockInfo &dest, const BlockInfo &src);

} // namespace mlir

#endif // TRITON_ANALYSIS_BUFFER_INDEX_ANALYSIS_H
