#ifndef TRITONAMD_ANALYSIS_AMDGPU_ALLOCATION_H
#define TRITONAMD_ANALYSIS_AMDGPU_ALLOCATION_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"

#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"

namespace mlir::triton::gpu {
class ConvertLayoutOp;
}

namespace mlir::triton::AMD {

unsigned getConvertLayoutScratchInBytes(gpu::ConvertLayoutOp op,
                                        TargetInfoBase &targetInfo);

unsigned AMDAllocationAnalysisScratchSizeFn(Operation *op,
                                            TargetInfoBase &targetInfo);

} // namespace mlir::triton::AMD

#endif // TRITONAMD_ANALYSIS_AMDGPU_ALLOCATION_H
