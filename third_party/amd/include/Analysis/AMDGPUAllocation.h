#ifndef TRITON_THIRD_PARTY_AMD_INCLUDE_ANALYSIS_AMDGPUALLOCATION_H_
#define TRITON_THIRD_PARTY_AMD_INCLUDE_ANALYSIS_AMDGPUALLOCATION_H_

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

#endif // TRITON_THIRD_PARTY_AMD_INCLUDE_ANALYSIS_AMDGPUALLOCATION_H_
