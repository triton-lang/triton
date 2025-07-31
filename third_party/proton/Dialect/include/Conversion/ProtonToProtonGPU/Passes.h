#ifndef PROTON_TO_PROTONGPU_PASSES_H
#define PROTON_TO_PROTONGPU_PASSES_H

#include "mlir/Pass/Pass.h"
#include "proton/Dialect/include/Dialect/Proton/IR/Dialect.h"
#include "proton/Dialect/include/Dialect/ProtonGPU/IR/Dialect.h"

namespace mlir::triton::proton {

// Generate the pass class declarations.
#define GEN_PASS_DECL
#include "proton/Dialect/include/Conversion/ProtonToProtonGPU/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createConvertProtonToProtonGPUPass(
    MetricType metricType = MetricType::CYCLE,
    SamplingStrategy samplingStrategy = SamplingStrategy::NONE,
    llvm::StringRef samplingOptions = "",
    gpu::Granularity granularity = gpu::Granularity::WARP,
    gpu::BufferStrategy bufferStrategy = gpu::BufferStrategy::CIRCULAR,
    gpu::BufferType bufferType = gpu::BufferType::SHARED,
    int32_t bufferSize = 0, int32_t maxSharedMemSize = 32768,
    int64_t profileScratchSize = 32768, int32_t profileScratchAlignment = 128,
    bool clkExt = false);

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "proton/Dialect/include/Conversion/ProtonToProtonGPU/Passes.h.inc"

} // namespace mlir::triton::proton

#endif // PROTON_TO_PROTONGPU_PASSES_H
