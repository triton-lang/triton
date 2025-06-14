#ifndef TRITON_DIALECT_TRITONINSTRUMENT_TRANSFORMS_PASSES_H_
#define TRITON_DIALECT_TRITONINSTRUMENT_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"
#include "triton/Dialect/TritonInstrument/IR/Dialect.h"

namespace mlir {
namespace triton {
namespace instrument {

// Generate the pass class declarations.
#define GEN_PASS_DECL
#include "triton/Dialect/TritonInstrument/Transforms/Passes.h.inc"

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "triton/Dialect/TritonInstrument/Transforms/Passes.h.inc"

} // namespace instrument
} // namespace triton
} // namespace mlir
#endif
