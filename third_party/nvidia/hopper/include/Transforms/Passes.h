
#ifndef DIALECT_NV_TRANSFORMS_PASSES_H_
#define DIALECT_NV_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir {

// Generate the pass class declarations.
#define GEN_PASS_DECL
#include "nvidia/hopper/include/Transforms/Passes.h.inc"

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "nvidia/hopper/include/Transforms/Passes.h.inc"

} // namespace mlir
#endif // DIALECT_NV_TRANSFORMS_PASSES_H_
