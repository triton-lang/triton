#ifndef PROTON_TO_PROTONGPU_PASSES_H
#define PROTON_TO_PROTONGPU_PASSES_H

#include "mlir/Pass/Pass.h"
#include "proton/dialect/include/Dialect/Proton/IR/Dialect.h"
#include "proton/dialect/include/Dialect/ProtonGPU/IR/Dialect.h"

namespace mlir::triton::proton {

// Generate the pass class declarations.
#define GEN_PASS_DECL
#include "proton/dialect/include/Conversion/ProtonToProtonGPU/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createConvertProtonToProtonGPUPass(
    std::string metric = "cycle", std::string granularity = "warp",
    std::string selectIds = "", int32_t maxSharedMem = 0,
    int32_t scratchMem = 32768, int32_t alignment = 128,
    std::string strategy = "circular", std::string bufferType = "shared_mem",
    int32_t bufferSize = 0);

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "proton/dialect/include/Conversion/ProtonToProtonGPU/Passes.h.inc"

} // namespace mlir::triton::proton

#endif // PROTON_TO_PROTONGPU_PASSES_H
