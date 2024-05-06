#ifndef TRITON_CONVERSION_TRITONTOTRITONCPU_TRITONTOTRITONCPUPASS_H
#define TRITON_CONVERSION_TRITONTOTRITONCPU_TRITONTOTRITONCPUPASS_H

#include <memory>

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createConvertTritonToTritonCPUPass();

} // namespace triton
} // namespace mlir

#endif
