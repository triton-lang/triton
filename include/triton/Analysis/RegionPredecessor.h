#ifndef TRITON_ANALYSIS_REGIONPREDECESSOR_H
#define TRITON_ANALYSIS_REGIONPREDECESSOR_H

#include "mlir/IR/Block.h"

namespace mlir {
class Operation;
class Region;

using BlockIter = std::pair<Block *, Block::iterator>;

// This class is a simple analysis that precomputes the predecessors for region
// branch operation for O(1) lookup. This is useful for other analyses that need
// to look up the predecessors of a region or a region's parent operation.
class RegionPredecessorAnalysis {
public:
  // Constructor that initializes the analysis at the given operation.
  RegionPredecessorAnalysis(Operation *op);

  // Get the region predecessors of the operation.
  ArrayRef<BlockIter> getPredecessors(Operation *op);
  // Get the region predecessors of the region.
  ArrayRef<BlockIter> getPredecessors(Region *region);

private:
  DenseMap<llvm::PointerUnion<Operation *, Region *>, SetVector<BlockIter>>
      predecessors;
};
} // namespace mlir

#endif // TRITON_ANALYSIS_REGIONPREDECESSOR_H
