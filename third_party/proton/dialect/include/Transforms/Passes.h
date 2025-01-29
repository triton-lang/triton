#ifndef TRANSFORMS_PASSES_H_
#define TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"
#include "proton/dialect/include/Dialect/Proton/IR/Dialect.h"

namespace mlir {
namespace triton {
namespace proton {

// Generate the pass class declarations.
#define GEN_PASS_DECL
#include "proton/dialect/include/Transforms/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>>
createProtonLowering(int32_t maxSharedMem, int32_t scratchMem,
                     int32_t alignment);

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "proton/dialect/include/Transforms/Passes.h.inc"

} // namespace proton
} // namespace triton
} // namespace mlir
#endif
