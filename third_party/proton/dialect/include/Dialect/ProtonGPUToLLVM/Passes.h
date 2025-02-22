#ifndef TRITON_THIRD_PARTY_PROTON_INCLUDE_TRITONPROTONGPUTOLLVM_PASSES_H_
#define TRITON_THIRD_PARTY_PROTON_INCLUDE_TRITONPROTONGPUTOLLVM_PASSES_H_

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

} // namespace mlir

namespace mlir::triton {

#define GEN_PASS_DECL
#include "third_party/proton/dialect/include/Dialect/ProtonGPUToLLVM/Passes.h"

} // namespace mlir::triton


namespace mlir::triton {

std::unique_ptr<OperationPass<ModuleOp>>
createConvertProtonGPUToLLVMPass(StringRef targetArch);

#define GEN_PASS_REGISTRATION
#include "third_party/proton/dialect/include/Dialect/ProtonGPUToLLVM/Passes.h"
} // namespace mlir::triton

#endif // TRITON_THIRD_PARTY_PROTON_INCLUDE_TRITONPROTONGPUTOLLVM_PASSES_H_
