#ifndef TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTRANSFORMS_LDSENCODING_H_
#define TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTRANSFORMS_LDSENCODING_H_

#include "Dialect/TritonAMDGPU/IR/TargetFeatures.h"
#include "amd/lib/TritonAMDGPUToLLVM/TargetInfo.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

// Choosing how a pipelined load is staged through LDS: the shared encoding of
// its buffer and whether the buffer can be filled by a direct-to-LDS copy.

namespace mlir {

// Adapted from
// lib/Dialect/TritonGPU/Transforms/Utility.cpp::getSharedEncIfAllUsersAreDotEnc
// to support AMDMfmaEncodingAttr.
// TODO(max): figure out how to refactor to use upstream
//
// If all the transitive uses of the given value have are used by a convert to
// the same dot operand encoding, return true and get the shared encoding that
// needs to be used to be compatible with users' layouts.
std::optional<triton::gpu::SharedEncodingTrait> getSharedEncIfAllUsersAreDotEnc(
    Operation *loadOp, triton::ModuleAxisInfoAnalysis &axisInfoAnalysis,
    const triton::amdgpu::TargetFeatures &targetFeatures,
    const triton::AMD::TargetInfo &targetInfo, bool useAsyncCopy);

// Whether `loadOp` can be lowered to an async copy into a `numBuffers`-deep
// ring of `sharedEnc` buffers. A null `sharedEnc` checks only the global side.
bool canBeConvertedToAsyncLoad(unsigned numBuffers, triton::LoadOp loadOp,
                               triton::gpu::SharedEncodingTrait sharedEnc,
                               triton::ModuleAxisInfoAnalysis &axisInfoAnalysis,
                               const triton::AMD::TargetInfo &targetInfo);

} // namespace mlir

#endif // TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTRANSFORMS_LDSENCODING_H_
