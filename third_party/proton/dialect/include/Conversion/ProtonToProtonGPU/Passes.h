#ifndef CONVERSION_PROTON_TO_PROTON_GPU_PASSES_H
#define CONVERSION_PROTON_TO_PROTON_GPU_PASSES_H

#include "mlir/Pass/Pass.h"
#include "proton/dialect/include/Dialect/Proton/IR/Dialect.h"
#include "proton/dialect/include/Dialect/ProtonGPU/IR/Dialect.h"

namespace mlir {
namespace triton {
namespace proton {

// Generate the pass class declarations.
#define GEN_PASS_DECL
#include "proton/dialect/include/Conversion/ProtonToProtonGPU/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>>
createConvertProtonToProtonGPUPass(std::string metric, std::string granularity,
                                   int32_t maxSharedMem, int32_t scratchMem,
                                   int32_t alignment, std::string strategy,
                                   std::string bufferType, int32_t bufferSize);

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "proton/dialect/include/Conversion/ProtonToProtonGPU/Passes.h.inc"

} // namespace proton
} // namespace triton
} // namespace mlir

#endif
