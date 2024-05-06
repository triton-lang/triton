#ifndef TRITONCPU_CONVERSION_TRITONCPUTOLLVM_PASSES_H
#define TRITONCPU_CONVERSION_TRITONCPUTOLLVM_PASSES_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include <memory>

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

namespace triton {

#define GEN_PASS_DECL
#include "triton/Conversion/TritonCPUToLLVM/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createConvertTritonCPUToLLVMPass();

#define GEN_PASS_REGISTRATION
#include "triton/Conversion/TritonCPUToLLVM/Passes.h.inc"

} // namespace triton

} // namespace mlir

#endif
