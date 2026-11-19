#ifndef TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTRANSFORMS_SUBTILEFUSION_H_
#define TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTRANSFORMS_SUBTILEFUSION_H_

#include "PipelineUtility.h"
#include "StreamOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

// Sub-tile fusion
//
// A sliced kernel loads one operand tile as several descriptor loads, one per
// slice. When the slices tile one larger tile, a single whole-tile TDM copy
// fetches all of them: every slice is then read back out of the shared buffer
// through memdesc_subslice, but the copy and its wait are issued once.

namespace mlir {

// Descriptor loads that read the cells of one regular grid over a larger
// tile: they share a descriptor and a result type, and their indices place
// each at a distinct constant multiple of the child shape.
struct SubtileLoadGroup {
  // In program order.
  SmallVector<triton::DescriptorLoadOp> children;
  // Element offset of each child inside the parent tile.
  SmallVector<SmallVector<int64_t>> offsets;
  // The child at offset zero; its indices address the parent tile.
  triton::DescriptorLoadOp origin;
  SmallVector<int64_t> parentShape;
};

// The pipelined descriptor loads that can share one whole-tile TDM copy.
// See SubtileFusion.cpp for the conditions a group must meet.
SmallVector<SubtileLoadGroup>
findSubtileLoadGroups(const LoadToInfoMap &loadToInfo);

// Replaces every child of `group` with a slice of one whole-tile TDM copy into
// a `numBuffers`-deep ring. The children keep one chain each, all sharing the
// copy and the wait, so every slice is still read beside the dot it feeds.
void createFusedTDMAsyncCopy(const SubtileLoadGroup &group,
                             triton::gpu::SharedEncodingTrait childSharedEnc,
                             scf::ForOp forOp, Value extractIdx, int numBuffers,
                             LoadToStreamOpMap &loadToStreamOp);

} // namespace mlir

#endif // TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTRANSFORMS_SUBTILEFUSION_H_
