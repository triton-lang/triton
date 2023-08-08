#ifndef TRITON_CONVERSION_TRITONTOTRITONGPU_TRITONTOTRITONGPUPASS_H
#define TRITON_CONVERSION_TRITONTOTRITONGPU_TRITONTOTRITONGPUPASS_H

#include <memory>

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

namespace triton {

constexpr static char AttrNumWarpsName[] = "triton_gpu.num-warps";
constexpr static char AttrNumCTAsName[] = "triton_gpu.num-ctas";
constexpr static char AttrComputeCapabilityName[] =
    "triton_gpu.compute-capability";

constexpr static char AttrNumThreadsPerWarp[] = "triton_gpu.threads-per-warp";

// Create the pass with numWarps passed from cl::opt.
std::unique_ptr<OperationPass<ModuleOp>> createConvertTritonToTritonGPUPass();

// Create the pass with numWarps set explicitly.
std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonToTritonGPUPass(int numWarps, int threadsPerWarp = 32,
                                   int numCTAs = 1, int computeCapability = 80);

} // namespace triton
} // namespace mlir

#endif
