#ifndef TRITON_ANALYSIS_LIVEBUFFERS_H
#define TRITON_ANALYSIS_LIVEBUFFERS_H

#include "triton/Analysis/RegionPredecessor.h"
#include "llvm/ADT/BitVector.h"

namespace mlir::triton {
class FuncOp;

struct BufferLiveRangeAnalysis {
  using BufferStates = std::optional<BitVector>;

  static bool join(BufferStates &lhs, const BitVector &rhs);
  void initialize(FuncOp func);
  void run(FuncOp func, RegionPredecessorAnalysis &preds);

  SmallVector<Operation *> buffers;
  llvm::MapVector<Operation *, size_t> bufferIds;
  llvm::MapVector<BlockIter, BufferStates> bufferStates;
};
} // namespace mlir::triton

#endif // TRITON_ANALYSIS_LIVEBUFFERS_H
