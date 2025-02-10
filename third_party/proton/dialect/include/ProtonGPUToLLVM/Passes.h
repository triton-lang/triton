#ifndef TRITON_THIRD_PARTY_PROTON_INCLUDE_PROTON_GPU_TO_LLVM_PASSES_H_
#define TRITON_THIRD_PARTY_PROTON_INCLUDE_PROTON_GPU_TO_LLVM_PASSES_H_

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

} // namespace mlir

namespace mlir::triton {
std::unique_ptr<OperationPass<ModuleOp>> createProtonLoweringPass();

#define GEN_PASS_REGISTRATION
#include "../third_party/proton/dialect/include/ProtonGPUToLLVM/Passes.h.inc"

} // namespace mlir::triton

#endif
