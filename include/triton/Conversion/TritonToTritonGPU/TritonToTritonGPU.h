#ifndef TRITON_CONVERSION_TRITONTOTRITONGPU_TRITONTOTRITONGPUPASS_H_
#define TRITON_CONVERSION_TRITONTOTRITONGPU_TRITONTOTRITONGPUPASS_H_

#include <memory>

namespace mlir{

class ModuleOp;
template <typename T> class OperationPass;

namespace triton{

std::unique_ptr<OperationPass<ModuleOp>> 
createConvertTritonToTritonGPUPass(int numWarps = 4);

}
} // namespace mlir


#endif