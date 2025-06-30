#ifndef TRITONAMD_ANALYSIS_AMDGPU_ALLOCATION_H
#define TRITONAMD_ANALYSIS_AMDGPU_ALLOCATION_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"

namespace mlir::triton::AMD {

constexpr char AttrSharedMemPadded[] = "amdgpu.use_padded_scratch_shmem";

unsigned getConvertLayoutScratchInBytes(RankedTensorType srcTy,
                                        RankedTensorType dstTy,
                                        bool usePadding);

unsigned AMDAllocationAnalysisScratchSizeFn(Operation *op);

} // namespace mlir::triton::AMD

#endif // TRITONAMD_ANALYSIS_AMDGPU_ALLOCATION_H
