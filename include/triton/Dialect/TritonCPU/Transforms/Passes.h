#ifndef TRITON_DIALECT_TRITONCPU_TRANSFORMS_PASSES_H_
#define TRITON_DIALECT_TRITONCPU_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace triton {
namespace cpu {} // namespace cpu
} // namespace triton

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "triton/Dialect/TritonCPU/Transforms/Passes.h.inc"

} // namespace mlir
#endif
