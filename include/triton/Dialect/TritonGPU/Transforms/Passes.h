#ifndef TRITON_DIALECT_TRITONGPU_TRANSFORMS_PASSES_H_
#define TRITON_DIALECT_TRITONGPU_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
std::unique_ptr<Pass> createTritonGPUPipelinePass(int numStages = 2);

// TODO(Keren): prefetch pass not working yet
std::unique_ptr<Pass> createTritonGPUPrefetchPass();

std::unique_ptr<Pass> createTritonGPUCanonicalizeLoopsPass();

std::unique_ptr<Pass> createTritonGPUCoalescePass();

std::unique_ptr<Pass> createTritonGPUCombineOpsPass();

std::unique_ptr<Pass> createTritonGPUVerifier();

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

} // namespace mlir
#endif
