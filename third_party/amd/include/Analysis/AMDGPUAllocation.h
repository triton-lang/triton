#ifndef TRITONAMD_ANALYSIS_AMDGPU_ALLOCATION_H
#define TRITONAMD_ANALYSIS_AMDGPU_ALLOCATION_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"

namespace mlir::triton::AMD {

unsigned getConvertLayoutScratchInBytes(RankedTensorType srcTy,
                                        RankedTensorType dstTy);

unsigned AMDAllocationAnalysisScratchSizeFn(Operation *op);

// For a layout conversion between `srcTy` and `dstTy`, return the vector length
// that can be used for the stores to and loads from shared memory,
// respectively.
std::pair</*inVec*/ unsigned, /*outVec*/ unsigned>
getScratchCvtInOutVecLengths(RankedTensorType srcTy, RankedTensorType dstTy);

} // namespace mlir::triton::AMD

#endif // TRITONAMD_ANALYSIS_AMDGPU_ALLOCATION_H
