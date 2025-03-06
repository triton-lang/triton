#ifndef PROTONGPU_CONVERSION_PROTONGPUTOLLVM_PROTONNVIDIAGPUTOLLVM_PASSES_H
#define PROTONGPU_CONVERSION_PROTONGPUTOLLVM_PROTONNVIDIAGPUTOLLVM_PASSES_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include <memory>

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

namespace triton::proton {

#define GEN_PASS_DECL
#include "proton/dialect/include/Conversion/ProtonGPUToLLVM/ProtonNvidiaGPUToLLVM/Passes.h.inc"

namespace gpu {
std::unique_ptr<OperationPass<ModuleOp>>
createConvertProtonNvidiaGPUToLLVMPass(int32_t computeCapability,
                                       int32_t ptxVersion);

} // namespace gpu

#define GEN_PASS_REGISTRATION
#include "proton/dialect/include/Conversion/ProtonGPUToLLVM/ProtonNvidiaGPUToLLVM/Passes.h.inc"

} // namespace triton::proton

} // namespace mlir

#endif
