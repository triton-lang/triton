#ifndef TRITON_DIALECT_TRITON_TRANSFORMS_PASSES_H_
#define TRITON_DIALECT_TRITON_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
class ModuleOp;
namespace triton {

// Group independent interleaved loads before layout assignment. The thread
// count is used only for profitability; address and memory-order checks are
// internal to this transformation.
void coalesceAdjacentLoads(ModuleOp module, int numThreads);

// Generate the pass class declarations.
#define GEN_PASS_DECL
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
