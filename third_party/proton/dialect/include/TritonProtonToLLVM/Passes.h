#ifndef TRITON_CONVERSION_TRITONPROTONTOLLVM_PASSES_H
#define TRITON_CONVERSION_TRITONPROTONTOLLVM_PASSES_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include <memory>

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

namespace triton {

#define GEN_PASS_DECL
#include "third_party/proton/dialect/include/TritonProtonToLLVM/Passes.h.inc"

namespace proton {

} // namespace proton

#define GEN_PASS_REGISTRATION
#include "third_party/proton/dialect/include/TritonProtonToLLVM/Passes.h.inc"

} // namespace triton

} // namespace mlir

#endif
