#include "Conversion/ProtonGPUToLLVM/Passes.h"
#include "Dialect/Proton/IR/Dialect.h"
#include "Dialect/ProtonGPU/IR/Dialect.h"
#include "mlir/Pass/PassManager.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"

namespace mlir::triton::proton::gpu {

#define GEN_PASS_DEF_ALLOCATEPROTONGLOBALSCRATCHBUFFERPASS
#include "Conversion/ProtonGPUToLLVM/Passes.h.inc"

struct AllocateProtonGlobalScratchBufferPass
    : public impl::AllocateProtonGlobalScratchBufferPassBase<
          AllocateProtonGlobalScratchBufferPass> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    OpPassManager pm("builtin.module");
    pm.addPass(mlir::triton::gpu::createTritonGPUGlobalScratchAllocationPass());
    if (failed(runPipeline(pm, mod)))
      signalPassFailure();
  }
};

} // namespace mlir::triton::proton::gpu
