#ifndef TRITON_DIALECT_TRITONGPU_TRANSFORMS_PASSES_H_
#define TRITON_DIALECT_TRITONGPU_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
std::unique_ptr<Pass> createTritonGPUPipelinePass();

namespace triton {
namespace gpu {
std::unique_ptr<Pass> createCombineOpsPass();
}
}

// /// Generate the code for registering passes.
// #define GEN_PASS_REGISTRATION
// #include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

} // namespace mlir
#endif
