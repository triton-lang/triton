#ifndef TRITON_DIALECT_TRITONGPUROCM_TRANSFORMS_PASSES_H_
#define TRITON_DIALECT_TRITONGPUROCM_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

namespace mlir {

std::unique_ptr<Pass> createTritonGPUROCMPipelinePass(int numStages = 3,
                                                  int numWarps = 4,
                                                  int numCTAs = 1,
                                                  int computeCapability = 80);

std::unique_ptr<Pass> createTritonGPUROCMStreamPipelinePass();

std::unique_ptr<Pass>
createTritonGPUROCMAccelerateMatmulPass(int computeCapability = 80);

std::unique_ptr<Pass> createTritonGPUROCMPrefetchPass();

std::unique_ptr<Pass> createTritonGPUROCMCanonicalizeLoopsPass();

std::unique_ptr<Pass> createTritonGPUROCMCoalescePass();

std::unique_ptr<Pass> createTritonGPUROCMReorderInstructionsPass();

std::unique_ptr<Pass> createTritonGPUROCMDecomposeConversionsPass();

std::unique_ptr<Pass> createTritonGPUROCMRemoveLayoutConversionsPass();

std::unique_ptr<Pass> createTritonGPUROCMVerifier();

std::unique_ptr<Pass> createTritonGPUROCMOptimizeDotOperandsPass();

std::unique_ptr<Pass> createTritonGPUROCMOptimizeEpiloguePass();

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "triton/Dialect/TritonGPUROCM/Transforms/Passes.h.inc"

} // namespace mlir
#endif
