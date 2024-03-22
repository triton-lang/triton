#ifndef TRITON_DIALECT_TRITONAMDGPU_TRANSFORMS_PASSES_H_
#define TRITON_DIALECT_TRITONAMDGPU_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

namespace mlir {

std::unique_ptr<Pass> createTritonAMDGPUPipelinePass(int numStages = 3,
                                                  int numWarps = 4,
                                                  int numCTAs = 1,
                                                  int computeCapability = 80);

std::unique_ptr<Pass> createTritonAMDGPUStreamPipelinePass();

std::unique_ptr<Pass>
createTritonAMDGPUAccelerateMatmulPass(std::string archGenName = std::string(),
                                       int matrixInstructionSize = 0);

std::unique_ptr<Pass> createTritonAMDGPUPrefetchPass();

std::unique_ptr<Pass> createTritonAMDGPUCanonicalizeLoopsPass();

std::unique_ptr<Pass> createTritonAMDGPUCoalescePass();

std::unique_ptr<Pass> createTritonAMDGPUReorderInstructionsPass();

std::unique_ptr<Pass> createTritonAMDGPUDecomposeConversionsPass();

std::unique_ptr<Pass> createTritonAMDGPURemoveLayoutConversionsPass();

std::unique_ptr<Pass> createTritonAMDGPUVerifier();

std::unique_ptr<Pass> createTritonAMDGPUOptimizeDotOperandsPass();

std::unique_ptr<Pass> createTritonAMDGPUOptimizeEpiloguePass();

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "TritonAMDGPUTransforms/Passes.h.inc"

} // namespace mlir
#endif
