#ifndef TRITON_ANALYSIS_LIVEBUFFERS_H
#define TRITON_ANALYSIS_LIVEBUFFERS_H

#include "triton/Analysis/RegionPredecessor.h"
#include "llvm/ADT/BitVector.h"

namespace mlir::triton {
class FuncOp;

class LiveBuffersAnalysis {
public:
  LiveBuffersAnalysis(FuncOp func, RegionPredecessorAnalysis &preds);

  // Get the buffer ID for the given operation.
  size_t getBufferId(Operation *op) const;
  // Get the operation for the given buffer ID.
  Operation *getBufferOp(size_t id) const;
  // Get an iterator range over all of the buffers.
  ArrayRef<Operation *> getBuffers() const { return buffers; }
  // Get the live buffers before the operation.
  const BitVector &getLiveBuffersBefore(Operation *op) const;
  // Get the live buffers after the operation.
  const BitVector &getLiveBuffersAfter(Operation *op) const;
  // Get the live buffers before the block iterator.
  const BitVector &getLiveBuffersBefore(BlockIter it) const;

  const BitVector &getLiveBufferMask(size_t id) const;
  BitVector getLiveBufferMask(ArrayRef<size_t> ids) const;

private:
  using BufferStates = std::optional<BitVector>;

  // Join two buffer states, taking the union of the live buffers.
  static bool join(BufferStates &lhs, const BitVector &rhs);
  // Initialize the analysis by finding all buffers to track.
  void initialize(FuncOp func);
  // Run fixed-point iteration of the live buffers.
  void run(FuncOp func, RegionPredecessorAnalysis &preds);

  // Buffer ID to operation map.
  SmallVector<Operation *> buffers;
  // Operation to buffer ID map.
  llvm::MapVector<Operation *, size_t> bufferIds;
  // The live buffers at each block iterator.
  llvm::MapVector<BlockIter, BufferStates> bufferStates;
  // Set of unique live states for the whole function.
  SetVector<BitVector> uniqueStates;
  // Live buffer masks for each buffer.
  SmallVector<BitVector> liveBufferMasks;
};
} // namespace mlir::triton

#endif // TRITON_ANALYSIS_LIVEBUFFERS_H
