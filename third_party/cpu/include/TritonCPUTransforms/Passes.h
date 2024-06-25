#ifndef TritonCPUTransforms_CONVERSION_PASSES_H
#define TritonCPUTransforms_CONVERSION_PASSES_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include <memory>

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

namespace triton {
namespace cpu {

#define GEN_PASS_DECL
#include "cpu/include/TritonCPUTransforms/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createConvertUnsupportedOps();
std::unique_ptr<OperationPass<ModuleOp>> createDecomposeFpConversions();

#define GEN_PASS_REGISTRATION
#include "cpu/include/TritonCPUTransforms/Passes.h.inc"

} // namespace cpu
} // namespace triton

} // namespace mlir

#endif
