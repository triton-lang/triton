#ifndef PROTONGPU_TRANSFORMS_PASSES_H_
#define PROTONGPU_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir::triton::proton::gpu {

// Generate the pass class declarations.
#define GEN_PASS_DECL
#include "proton/Dialect/include/Dialect/ProtonGPU/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "proton/Dialect/include/Dialect/ProtonGPU/Transforms/Passes.h.inc"

} // namespace mlir::triton::proton::gpu

#endif // PROTONGPU_TRANSFORMS_PASSES_H_
