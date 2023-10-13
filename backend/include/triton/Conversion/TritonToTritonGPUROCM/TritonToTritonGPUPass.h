#ifndef TRITON_CONVERSION_ROCM_TRITONTOTRITONGPU_TRITONTOTRITONGPUPASS_H
#define TRITON_CONVERSION_ROCM_TRITONTOTRITONGPU_TRITONTOTRITONGPUPASS_H

#include <memory>

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

namespace triton {

constexpr static char AttrNumWarpsName[] = "triton_gpu_rocm.num-warps";
constexpr static char AttrNumCTAsName[] = "triton_gpu_rocm.num-ctas";
constexpr static char AttrComputeCapabilityName[] =
    "triton_gpu_rocm.compute-capability";

constexpr static char AttrNumThreadsPerWarp[] = "triton_gpu_rocm.threads-per-warp";

// Create the pass with numWarps passed from cl::opt.
std::unique_ptr<OperationPass<ModuleOp>> createConvertTritonToTritonGPUROCMPass();

// Create the pass with numWarps set explicitly.
std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonToTritonGPUROCMPass(int numWarps, int threadsPerWarp = 64,
                                   int numCTAs = 1, int computeCapability = 80);

} // namespace triton
} // namespace mlir

#endif
