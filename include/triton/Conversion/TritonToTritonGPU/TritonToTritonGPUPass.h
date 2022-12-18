#ifndef TRITON_CONVERSION_TRITONTOTRITONGPU_TRITONTOTRITONGPUPASS_H_
#define TRITON_CONVERSION_TRITONTOTRITONGPU_TRITONTOTRITONGPUPASS_H_

#include <memory>

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

namespace triton {

constexpr static char AttrNumWarpsName[] = "triton_gpu.num-warps";

// Create the pass with numWarps passed from cl::opt.
std::unique_ptr<OperationPass<ModuleOp>> createConvertTritonToTritonGPUPass();

// Create the pass with numWarps set explicitly.
std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonToTritonGPUPass(int numWarps);

} // namespace triton
} // namespace mlir

#endif
