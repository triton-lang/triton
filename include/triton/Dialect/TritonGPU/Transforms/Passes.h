#ifndef TRITON_DIALECT_TRITONGPU_TRANSFORMS_PASSES_H_
#define TRITON_DIALECT_TRITONGPU_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
std::unique_ptr<Pass> createTritonGPUPipelinePass(int numStages = 2);

std::unique_ptr<Pass>
createTritonGPUAccelerateMatmulPass(int computeCapability = 80);

std::unique_ptr<Pass> createTritonGPUPrefetchPass();

std::unique_ptr<Pass> createTritonGPUCanonicalizeLoopsPass();

std::unique_ptr<Pass> createTritonGPUCoalescePass();

std::unique_ptr<Pass> createTritonGPUReorderInstructionsPass();

std::unique_ptr<Pass> createTritonGPUDecomposeConversionsPass();

std::unique_ptr<Pass> createTritonGPURemoveLayoutConversionsPass();

std::unique_ptr<Pass> createTritonGPUVerifier();

std::unique_ptr<Pass> createTritonGPUOptimizeDotOperandsPass();

std::unique_ptr<Pass> createTritonGPUUpdateMmaForVoltaPass();

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

} // namespace mlir
#endif
